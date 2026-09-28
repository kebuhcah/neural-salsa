"""When in Amor y Control does the 2.5-6 kHz band pull toward the inversion?

Localises NOTES 13a in time, so the claim can be checked by ear with the
explorer's mute toggles. Needs the checkpoints amor_compare.py saves.

    python amor_when.py
"""
import sys
from pathlib import Path
WT = Path(__file__).resolve().parent
sys.path.insert(0, str(WT))
import numpy as np, torch, torch.nn.functional as F
from ablation import BANDS, band_mask
from arch_compare import Model
from train_phase import load_all, make_index, batches, DEV, FPB

W, BLK = 24, 16
songs = load_all()
perm = np.random.default_rng(0).permutation(len(songs))
si = next(int(i) for i in perm[:16] if songs[i]["title"] == "Amor y Control")
idx = make_index(songs, [si], W)
times = np.load(next(f for f in sorted((WT / "data/features").glob("*.npz"))
                     if str(np.load(f, allow_pickle=True)["title"]) == "Amor y Control"),
                allow_pickle=True)["times"]


def ev(model, mask=None):
    L, Y = [], []
    m = None if mask is None else torch.from_numpy(mask).to(DEV)
    with torch.no_grad():
        for X, y in batches(songs, idx, W, 128, np.random.default_rng(1), 0.0, 1.0, shuffle=False):
            if m is not None: X = X * m
            L.append(F.log_softmax(model(X)[0], 1).cpu().numpy()); Y.append(y.cpu().numpy())
    lp, t = np.concatenate(L), np.concatenate(Y)
    j = np.arange(len(t))
    return lp[j, t] - lp[j, (t + 4) % 8]


E0, EH, EM = [], [], []
for s in range(4):
    m = Model("gru", W * FPB).to(DEV)
    m.load_state_dict(torch.load(WT / f"data/amor_w24_s{s}.pt", map_location=DEV)); m.eval()
    E0.append(ev(m)); EH.append(ev(m, band_mask(*BANDS["high-mid 2.5-6k"], keep=False)))
    EM.append(ev(m, band_mask(*BANDS["mid 800-2.5k"], keep=False)))
e0, dh, dm = np.mean(E0, 0), np.mean(EH, 0) - np.mean(E0, 0), np.mean(EM, 0) - np.mean(E0, 0)
beats = idx[:len(e0), 1]
mmss = lambda t: f"{int(t // 60)}:{int(t % 60):02d}"
print(f"{'time':>11} {'beats':>9} {'e full':>7} {'+high-mid mute':>15} {'+mid mute':>10}")
rows = []
for k in np.unique(beats // BLK):
    m = beats // BLK == k
    b0, b1 = beats[m][0], beats[m][-1]
    rows.append((times[b0], times[b1], b0, b1, e0[m].mean(), dh[m].mean(), dm[m].mean()))
for t0, t1, b0, b1, e, h, d in rows:
    flag = "  <-- inverts, high-mid mute fixes" if e < 0 and e + h > 0 else (
           "  <-- inverts" if e < 0 else "")
    print(f"{mmss(t0)}-{mmss(t1)} {b0:4d}-{b1:<4d} {e:+7.2f} {h:+15.2f} {d:+10.2f}{flag}")
print(f"\ncorr over windows between e(full) and high-mid effect: {np.corrcoef(e0, dh)[0,1]:+.2f}")
