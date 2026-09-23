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

Shared conv trunk throughout, so only the head differs. Same fixed val split
for every variant and seed, and per-song accuracy recorded, because song
difficulty (sd ~0.19) dwarfs the effects being measured.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_phase import load_all, make_index, batches, DEV, FPB, N_MELS

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


def run(kind, W, songs, epochs, seed, tr, va_index, aux=0.3, micro=None):
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
    model = Model(kind, W * FPB).to(DEV)
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
                (loss / chunks).backward()
            if CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); nb += 1
            if nb >= 500:
                break
    model.eval()
    out = {}
    with torch.no_grad():
        for si, idx in va_index.items():
            P, Y = [], []
            r2 = np.random.default_rng(1)
            for X, y in batches(songs, idx, W, micro, r2, 0.0, 1.0, shuffle=False):
                P.append(model(X)[0].argmax(1).cpu().numpy()); Y.append(y.cpu().numpy())
            if not P:
                continue
            p, t = np.concatenate(P), np.concatenate(Y)
            out[si] = {"acc": float((p == t).mean()),
                       "q": float((p == (t + 4) % 8).mean()),
                       "h": float((p // 4 == t // 4).mean()),
                       "r": float((p % 4 == t % 4).mean())}
    n = sum(p.numel() for p in model.parameters())
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--kinds", default="flatten,gru,factor,gruf,halves")
    args = ap.parse_args()

    songs = load_all()
    W = args.window
    perm = np.random.default_rng(0).permutation(len(songs))
    val_ids, tr_ids = perm[:16], perm[16:]
    tr = make_index(songs, tr_ids, W)
    va_index = {int(si): make_index(songs, [si], W) for si in val_ids}
    va_index = {k: v for k, v in va_index.items() if len(v) >= 64}
    print(f"W={W}  {args.epochs} epochs  {args.seeds} seeds  "
          f"{len(tr)} train windows, {len(va_index)} val songs\n")
    print(f"{'arch':<9} {'params':>9} {'acc':>7} {'q(1-5)':>8} {'h':>7} {'r':>7}")
    print("-" * 52)
    res = {}
    for kind in args.kinds.split(","):
        per_seed = []
        for s in range(args.seeds):
            out, npar = run(kind, W, songs, args.epochs, s, tr, va_index)
            per_seed.append(out)
        res[kind] = per_seed
        M = lambda k: np.mean([[v[k] for v in o.values()] for o in per_seed])
        print(f"{kind:<9} {npar:9,} {M('acc'):7.3f} {M('q'):8.3f} "
              f"{M('h'):7.3f} {M('r'):7.3f}", flush=True)
    Path(ROOT / "data/arch_compare.json").write_text(json.dumps(res, indent=1))
    print("\nchance: acc 0.125, h 0.500, r 0.250")


if __name__ == "__main__":
    main()
