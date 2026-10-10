#!/usr/bin/env python3
"""Architecture comparison, aimed at the one decision that is actually hard.

Decomposing the count as c = 4h + r (h = which half of the 8-count, r =
position within it) shows the baseline has effectively solved r (0.898 vs
0.250 chance) and is near a coinflip on h (0.740 vs 0.500). Perfect h would
take overall accuracy from 0.686 to 0.898, so every variant here is a
different way of attacking h.

  flatten   baseline: flatten the time axis into an MLP
  gru       recurrent over time instead; far fewer parameters
  factor    separate heads for h and r, combined as log p(h) + log p(r),
            with auxiliary losses so h gets gradient of its own
  gruf      gru trunk + factorised head
  halves    encode the two halves of the window separately and compare them,
            since telling half 1 from half 2 is inherently a comparison

Those share one conv trunk, so only the head differs. Two later variants
change the body, aimed at what the Amor y Control investigation found
(NOTES 13):

  bands     one small trunk + GRU per frequency band, each predicting alone,
            combined by a learned per-window gate (readable per song)
  beatseq   per-beat encoder, then a transformer across beats; memory per
            beat, so long windows are cheap

Same fixed val split for every variant and seed, and per-song results
recorded, because song difficulty dwarfs the effects being measured. The
headline is song level (NOTES 10): batch-decoded accuracy, flips, and mean e,
the continuous evidence for the truth over the 1<->5 flip.

    python arch_compare.py --window 24 --epochs 4 --kinds bands,beatseq,beatseq@48
    python arch_compare.py --window 24 --kinds gru-saved     # re-score amor_compare models
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_phase import load_all, load_input, make_index, batches, DEV, FPB, N_MELS

ROOT = Path(__file__).resolve().parent
CLIP = 0.0          # >0 enables gradient-norm clipping


# Trunk variants. Two things were changed at once earlier -- BatchNorm ->
# GroupNorm and pool-after -> stride-2 conv -- and the result got worse and
# less stable, with no way to tell which change did it. Both are now flags so
# they can be varied one at a time.
TRUNK_NORM = "batch"        # "batch" | "group"
TRUNK_DOWN = "pool"         # "pool"  | "stride"


class Trunk(nn.Module):
    """3 conv blocks, then average away frequency. Keeps the time axis.

    `down="stride"` downsamples inside the first conv instead of pooling after
    it, which cuts that layer's activation memory 4x (1.6GB -> 0.4GB at W=48,
    batch 128) for the same output resolution.

    `norm="group"` removes BatchNorm's dependence on batch composition, which
    otherwise makes any batch-splitting workaround change the computation.
    """

    def __init__(self, ch=(1, 32, 64, 128), groups=8, norm=None, down=None):
        super().__init__()
        norm = norm or TRUNK_NORM
        down = down or TRUNK_DOWN
        stride1 = 2 if down == "stride" else 1
        self.convs = nn.ModuleList([
            nn.Conv2d(ch[0], ch[1], 3, stride=stride1, padding=1),
            nn.Conv2d(ch[1], ch[2], 3, padding=1),
            nn.Conv2d(ch[2], ch[3], 3, padding=1)])
        mk = ((lambda c: nn.GroupNorm(groups, c)) if norm == "group"
              else (lambda c: nn.BatchNorm2d(c)))
        self.norms = nn.ModuleList([mk(c) for c in ch[1:]])
        # A strided first conv has already halved both axes.
        self.pool_after = (down != "stride", True, True)
        self.out = ch[-1]

    def forward(self, x):
        for c, n, pool in zip(self.convs, self.norms, self.pool_after):
            x = F.relu(n(c(x)))
            if pool:
                x = F.max_pool2d(x, 2)
        return x.mean(dim=3)                       # [B, C, T]


class Combined(nn.Module):
    """Factorised head: logit[c] = log p(h=c//4) + log p(r=c%4).

    Structural, not merely auxiliary -- the 8-way distribution is *built* from
    the two sub-decisions, so the model cannot get 8-way credit without
    committing on h.
    """

    def __init__(self, d):
        super().__init__()
        self.h = nn.Linear(d, 2)
        self.r = nn.Linear(d, 4)

    def forward(self, z):
        lh = F.log_softmax(self.h(z), -1)          # [B,2]
        lr = F.log_softmax(self.r(z), -1)          # [B,4]
        return (lh[:, :, None] + lr[:, None, :]).reshape(-1, 8), lh, lr


class Model(nn.Module):
    def __init__(self, kind, w_frames):
        super().__init__()
        self.kind = kind
        self.trunk = Trunk()
        C, T = self.trunk.out, max(w_frames // 8, 1)
        if kind in ("flatten", "factor"):
            self.mlp = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                     nn.Linear(C * T, 256), nn.ReLU(), nn.Dropout(0.3))
            d = 256
        elif kind in ("gru", "gruf"):
            self.rnn = nn.GRU(C, 128, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            d = 256
        elif kind == "halves":
            self.rnn = nn.GRU(C, 128, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3)
            d = 256 * 3                            # first half, second half, diff
        self.head = (Combined(d) if kind in ("factor", "gruf", "halves")
                     else nn.Linear(d, 8))

    def embed(self, x):
        z = self.trunk(x)                          # [B,C,T]
        if self.kind in ("flatten", "factor"):
            return self.mlp(z)
        seq = z.transpose(1, 2)                    # [B,T,C]
        if self.kind in ("gru", "gruf"):
            o, _ = self.rnn(seq)
            return self.drop(o[:, -1])
        half = seq.shape[1] // 2
        a, _ = self.rnn(seq[:, :half]); b, _ = self.rnn(seq[:, half:])
        a, b = a[:, -1], b[:, -1]
        return self.drop(torch.cat([a, b, a - b], -1))

    def forward(self, x):
        z = self.embed(x)
        if isinstance(self.head, Combined):
            return self.head(z)
        return self.head(z), None, None


# Mel-bin ranges of the five ablation bands (ablation.BANDS), inclusive:
# bass, low-mid, mid, high-mid, high. Duplicated rather than imported because
# ablation.py imports train_phase's Net and would pull it in here.
BAND_BINS = [(0, 11), (12, 33), (34, 69), (70, 105), (106, 127)]


class Bands(nn.Module):
    """Per-band experts, combined by a learned gate.

    Each band gets its own small trunk and GRU and predicts the count alone.
    The combined prediction is sum_b g_b * log p_b, renormalised -- a weighted
    product of experts -- where the gate g (one weight per band, per window,
    mean 1) is computed from all the bands' embeddings. On Amor y Control the
    high-mid band argues for the 1<->5 flip while the bass carries the truth
    (NOTES 13a/13c); a single trunk can only average them, this can learn to
    trust the bass. Each expert also gets its own loss so it stays a usable
    predictor alone, and the gate weights are readable per song.
    """

    def __init__(self, ch=(1, 16, 32, 64)):
        super().__init__()
        nb, c = len(BAND_BINS), ch[-1]
        self.trunks = nn.ModuleList([Trunk(ch=ch) for _ in range(nb)])
        self.rnns = nn.ModuleList([nn.GRU(c, c, batch_first=True, bidirectional=True)
                                   for _ in range(nb)])
        self.heads = nn.ModuleList([nn.Linear(2 * c, 8) for _ in range(nb)])
        self.gate = nn.Sequential(nn.Linear(2 * c * nb, 64), nn.ReLU(), nn.Linear(64, nb))
        self.drop = nn.Dropout(0.3)
        self.experts = None        # per-band log-probs from the last forward, for aux loss
        self.last_gate = None      # [B, nb] gate weights from the last forward

    def forward(self, x):
        zs, lps = [], []
        for (lo, hi), trunk, rnn, head in zip(BAND_BINS, self.trunks, self.rnns, self.heads):
            o, _ = rnn(trunk(x[..., lo:hi + 1]).transpose(1, 2))
            z = self.drop(o[:, -1])
            zs.append(z)
            lps.append(F.log_softmax(head(z), -1))
        g = F.softmax(self.gate(torch.cat(zs, -1)), -1) * len(BAND_BINS)
        self.experts, self.last_gate = lps, g.detach()
        mixed = (g[..., None] * torch.stack(lps, 1)).sum(1)
        return F.log_softmax(mixed, -1), None, None


class BeatSeq(nn.Module):
    """Per-beat encoder, then a transformer across beats.

    The shared trunk keeps full frame resolution across the whole window, so
    its memory grows with W*16 frames and W>24 needed micro-batching (NOTES
    9). Here each beat's 16 frames are encoded on their own -- keeping two
    sub-beat time steps, so syncopation survives -- and only the W beat
    vectors meet, in a small transformer. Memory is per beat, so 48-64 beat
    windows are affordable. GroupNorm throughout, so micro-batching would not
    change the computation either.
    """

    def __init__(self, W, d=128, layers=2, heads=4, n_chroma=0):
        super().__init__()

        def blk(i, o):
            return [nn.Conv2d(i, o, 3, stride=2, padding=1), nn.GroupNorm(4, o), nn.ReLU()]

        self.W, self.n_chroma = W, n_chroma
        self.enc = nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 64))   # [16,128] -> [2,16]
        # Optional chroma branch (input = 128 mel bins + n_chroma pitch classes
        # per frame). Chroma is not a spectrum -- adjacent bins are not
        # neighbouring frequencies -- so it gets its own small dense encoder
        # per beat rather than sharing the convolution.
        self.cenc = (nn.Sequential(nn.Linear(FPB * n_chroma, 64), nn.ReLU())
                     if n_chroma else None)
        self.proj = nn.Linear(64 * 2 + (64 if n_chroma else 0), d)
        self.pos = nn.Parameter(torch.randn(W, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(d, 8)

    def forward(self, x):
        B = x.shape[0]
        beats = x.reshape(B * self.W, 1, FPB, x.shape[-1])       # one image per beat
        if self.cenc is not None:
            mel, chroma = beats[..., :-self.n_chroma], beats[..., -self.n_chroma:]
            z = torch.cat([self.enc(mel).mean(3).flatten(1),
                           self.cenc(chroma.flatten(1))], 1)      # [B*W, 128 + 64]
        else:
            z = self.enc(beats).mean(3).flatten(1)              # [B*W, 128]
        z = self.tf(self.proj(z).reshape(B, self.W, -1) + self.pos)
        return self.head(self.drop(z[:, -1])), None, None       # target is the last beat


def make_model(kind, W):
    if kind == "bands":
        return Bands()
    if kind == "beatseq":
        return BeatSeq(W)
    if kind == "beatseqc":
        return BeatSeq(W, n_chroma=12)                           # needs --input mel+chroma
    return Model(kind, W * FPB)


def song_level(logp, beats, truth):
    """Batch decoding: one phase for the whole song, chosen by every window.

    Per-window accuracy is a proxy (NOTES section 10): phase advances
    deterministically, so the song's labelling is fixed by one of 8
    hypotheses. Scored against the actual beat index rather than list
    position, because untrusted windows leave gaps in the index.

    song   accuracy of the decoded labelling
    flip   1 if the winner is the hypothesis four away -- the confident
           1<->5 inversion that aggregation amplifies instead of cancelling
    margin winning minus runner-up log-score per window; how decisive it was
    e      mean ln p(true) - ln p(true+4): evidence for the truth over the
           1<->5 flip, in nats. Continuous where `song` is nearly binary, so
           it separates models that decode the same songs (NOTES 13a)
    """
    j = np.arange(len(beats))
    scores = np.array([logp[j, (h + beats) % 8].sum() for h in range(8)])
    phi = int(scores.argmax())
    pred = (phi + beats) % 8
    # Truth is almost always (phi* + beat) % 8; take its mode as phi*.
    phi_true = int(np.bincount((truth - beats) % 8, minlength=8).argmax())
    top2 = np.sort(scores)[-2:]
    return {"song": float((pred == truth).mean()),
            "flip": int(phi == (phi_true + 4) % 8),
            "margin": float((top2[1] - top2[0]) / len(beats)),
            "e": float((logp[j, truth] - logp[j, (truth + 4) % 8]).mean())}


def run(kind, W, songs, epochs, seed, tr, va_index, aux=0.3, micro=None,
        return_model=False, raw=None, train_index=None, train_out=None):
    """micro: activation-memory cap via gradient accumulation.

    The first conv keeps full time x mel resolution at 32 channels, so one
    activation is batch*32*(W*16)*128*4 bytes -- 268MB at W=8 but 1.6GB at
    W=48, before autograd stores intermediates. Splitting the batch and
    accumulating gradients is mathematically identical to the full batch and
    caps that. Default keeps batch*W constant at the W=8 setting.
    """
    if micro is None:
        # Default to NO chunking. Auto-shrinking with W was a trap: with
        # BatchNorm it silently changes the computation, so two runs that
        # differ only in window size also differed in how they normalise,
        # and a 3-hour ablation ended up measuring the wrong variable.
        # Pass micro explicitly when memory forces it, and say so in results.
        micro = 128
    rng = np.random.default_rng(100 + seed)
    torch.manual_seed(100 + seed)
    model = make_model(kind, W).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train(); nb = 0
        for X, y in batches(songs, tr, W, 128, rng, 0.0, 1.0):
            opt.zero_grad()
            chunks = max(1, (len(X) + micro - 1) // micro)
            for ci in range(chunks):
                xs, ys = X[ci * micro:(ci + 1) * micro], y[ci * micro:(ci + 1) * micro]
                if not len(xs):
                    continue
                logits, lh, lr = model(xs)
                loss = F.cross_entropy(logits, ys)
                if lh is not None and aux:
                    # Give h its own gradient, not only the 8-way signal.
                    loss = loss + aux * (F.nll_loss(lh, ys // 4) + F.nll_loss(lr, ys % 4))
                if getattr(model, "experts", None) is not None and aux:
                    # Each band expert also predicts alone, so it stays usable.
                    loss = loss + aux * sum(F.nll_loss(e, ys) for e in model.experts) / len(model.experts)
                (loss / chunks).backward()
            if CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); nb += 1
            if nb >= 500:
                break
    out = evaluate(model, songs, W, va_index, micro, raw=raw)
    if train_index is not None and train_out is not None:
        # Same metrics on songs the model trained on: the train/val gap says
        # whether it is overfitting (data-limited) or underfitting (capacity).
        train_out.update(evaluate(model, songs, W, train_index, micro))
    n = sum(p.numel() for p in model.parameters())
    return (out, n, model) if return_model else (out, n)


def evaluate(model, songs, W, va_index, bs=128, raw=None):
    """Per-song window and song-level metrics; band gate weights if present.

    raw: if a dict, also filled with {song: (logp, beat index, truth)} so
    seeds can be ensembled later without retraining (ensemble.py).
    """
    model.eval()
    out = {}
    with torch.no_grad():
        for si, idx in va_index.items():
            L, Y, G = [], [], []
            r2 = np.random.default_rng(1)
            for X, y in batches(songs, idx, W, bs, r2, 0.0, 1.0, shuffle=False):
                # log_softmax is a no-op on the factorised and band heads'
                # output, which is already normalised log-probability.
                L.append(F.log_softmax(model(X)[0], 1).cpu().numpy())
                Y.append(y.cpu().numpy())
                if getattr(model, "last_gate", None) is not None:
                    G.append(model.last_gate.cpu().numpy())
            if not L:
                continue
            logp, t = np.concatenate(L), np.concatenate(Y)
            p = logp.argmax(1)
            out[si] = {"acc": float((p == t).mean()),
                       "q": float((p == (t + 4) % 8).mean()),
                       "h": float((p // 4 == t // 4).mean()),
                       "r": float((p % 4 == t % 4).mean()),
                       **song_level(logp, idx[:len(t), 1], t)}
            if G:
                out[si]["gate"] = np.concatenate(G).mean(0).round(3).tolist()
            if raw is not None:
                raw[si] = (logp.astype(np.float32), idx[:len(t), 1], t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--kinds", default="flatten,gru,factor,gruf,halves",
                    help="comma list of kind[@W][*EPOCHS]; '@W' and '*E' override the window "
                         "and epochs for that kind; 'gru-saved' re-scores data/amor_w24_s*.pt "
                         "(gru, W=24, split 0) without training")
    ap.add_argument("--split", type=int, default=0,
                    help="which 16 songs are held out: perm[16*k:16*(k+1)]. Split 0 is the "
                         "one every earlier result used; others are disjoint from it")
    ap.add_argument("--folds", type=int, default=0,
                    help="cross-validation: split all songs into this many folds and hold out "
                         "--fold; overrides --split")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--train-songs", type=int, default=0,
                    help="train on only the first N training songs (learning curve); 0 = all")
    ap.add_argument("--train-eval", type=int, default=16,
                    help="also score this many training songs, for the train/val gap; 0 = off")
    ap.add_argument("--seed0", type=int, default=0, help="first seed, so new seeds can be added")
    ap.add_argument("--out", default="data/arch_compare.json")
    ap.add_argument("--input", default="mel", choices=["mel", "chroma", "mel+chroma"],
                    help="model input features (chroma from build_chroma.py)")
    ap.add_argument("--save-logp", default=None, metavar="DIR",
                    help="also save each seed's per-song log-probs to DIR/<spec>_split<k>_s<seed>.npz "
                         "for ensemble.py")
    args = ap.parse_args()

    songs = load_input(args.input)
    perm = np.random.default_rng(0).permutation(len(songs))
    if args.folds:
        k = f"cv{args.folds}.{args.fold}"
        parts = np.array_split(perm, args.folds)
        val_ids = parts[args.fold]
        tr_ids = np.concatenate([p for i, p in enumerate(parts) if i != args.fold])
    else:
        k = args.split
        val_ids = perm[16 * k:16 * (k + 1)]
        tr_ids = np.concatenate([perm[:16 * k], perm[16 * (k + 1):]])
    if args.train_songs:
        tr_ids = tr_ids[:args.train_songs]
    # Fixed subset of the training songs, scored like held-out songs.
    tr_eval_ids = tr_ids[:args.train_eval] if args.train_eval else []
    seeds_used = list(range(args.seed0, args.seed0 + args.seeds))
    print(f"split {k}  seeds {seeds_used}  val songs: {len(val_ids)}  "
          f"train songs: {len(tr_ids)}\n")
    print(f"{'arch':<12} {'W':>3} {'params':>9} {'acc':>6} {'h':>6} {'song':>6} "
          f"{'exact':>6} {'flips':>6} {'e':>6}   per-seed song / e")
    print("-" * 96)
    res = {}
    for spec in args.kinds.split(","):
        m = re.fullmatch(r"([\w-]+)(?:@(\d+))?(?:\*(\d+))?", spec)
        assert m, f"bad kind spec {spec!r}"
        kind = m.group(1)
        W = int(m.group(2)) if m.group(2) else args.window
        epochs = int(m.group(3)) if m.group(3) else args.epochs
        va_index = {int(si): make_index(songs, [si], W) for si in val_ids}
        va_index = {i: v for i, v in va_index.items() if len(v) >= 64}
        per_seed, train_seeds = [], []
        t0 = time.time()

        def save(s, raw):
            if not args.save_logp:
                return
            d = Path(ROOT / args.save_logp); d.mkdir(parents=True, exist_ok=True)
            safe = spec.replace("*", "x").replace("@", "_w")
            np.savez_compressed(d / f"{safe}_split{k}_s{s}.npz", **{
                f"{si}_{part}": arr for si, trio in raw.items()
                for part, arr in zip(("logp", "beats", "truth"), trio)})

        if kind == "gru-saved":
            assert W == 24 and k == 0, "the saved gru checkpoints are W=24, split 0"
            for s in range(4):
                path = ROOT / f"data/amor_w24_s{s}.pt"
                if not path.exists():
                    continue
                model = Model("gru", W * FPB).to(DEV)
                model.load_state_dict(torch.load(path, map_location=DEV))
                raw = {}
                per_seed.append(evaluate(model, songs, W, va_index, raw=raw))
                save(s, raw)
            npar = sum(p.numel() for p in model.parameters())
        else:
            tr = make_index(songs, tr_ids, W)
            tr_eval = {int(si): make_index(songs, [si], W) for si in tr_eval_ids}
            tr_eval = {i: v for i, v in tr_eval.items() if len(v) >= 64} or None
            for s in seeds_used:
                raw, tr_out = {}, {}
                out, npar = run(kind, W, songs, epochs, s, tr, va_index, raw=raw,
                                train_index=tr_eval, train_out=tr_out)
                per_seed.append(out)
                train_seeds.append(tr_out)
                save(s, raw)
        mins = (time.time() - t0) / 60
        res[spec] = {"W": W, "epochs": epochs, "split": k, "params": npar,
                     "train_songs": len(tr_ids),
                     "seed_ids": list(range(4)) if kind == "gru-saved" else seeds_used,
                     "minutes": round(mins, 1), "seeds": per_seed,
                     "train_seeds": train_seeds}
        M = lambda k: np.mean([[v[k] for v in o.values()] for o in per_seed])
        Cnt = lambda f: np.mean([sum(f(v) for v in o.values()) for o in per_seed])
        seeds = " ".join(f"{np.mean([v['song'] for v in o.values()]):.3f}/"
                         f"{np.mean([v['e'] for v in o.values()]):+.2f}" for o in per_seed)
        print(f"{spec:<12} {W:3d} {npar:9,} {M('acc'):6.3f} {M('h'):6.3f} {M('song'):6.3f} "
              f"{Cnt(lambda v: v['song'] > 0.95):6.1f} {Cnt(lambda v: v['flip']):6.1f} "
              f"{M('e'):+6.2f}   {seeds}   [{mins:.0f} min]", flush=True)
        if any(train_seeds):
            TM = lambda k: np.mean([[v[k] for v in o.values()] for o in train_seeds if o])
            print(f"{'  (train)':<12} {'':>13} {TM('acc'):6.3f} {TM('h'):6.3f} {TM('song'):6.3f} "
                  f"{'':>13} {TM('e'):+6.2f}   scored on {len(next(o for o in train_seeds if o))} "
                  f"training songs", flush=True)
        Path(ROOT / args.out).write_text(json.dumps(res, indent=1))
    print(f"\nexact: songs >0.95 of {len(va_index)}; flips: songs decoded to the 1<->5 "
          "inversion; e: mean evidence for the truth over the flip (nats)")


if __name__ == "__main__":
    main()
