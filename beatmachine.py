#!/usr/bin/env python3
"""Render Salsa Beat Machine patterns with a known 1, as model probes. (NOTES 15)

The Salsa Beat Machine (salsabeatmachine.org, source urish/beat-machine)
plays canonical salsa patterns per instrument from one-shot samples. This
reimplements its playback (engine/beat-engine.ts) offline, so we can render
any combination of instruments and patterns -- each alone, all but one, in
any key, with an instrument rotated by 4 beats (e.g. 3-2 instead of 2-3
clave) -- and know the count of every beat exactly.

Assets are NOT in this repo (no licence; used locally with the owner's OK):
data/beatmachine/{salsa.xml, main.json, main.mp3}, from the site's
public/assets/machines/salsa.xml and assets/audio/main.{json,mp3}.

Playback, as in the site:
  * patterns are on an eighth-note grid; program index 0 is count 1
    (the instructor's "1,5" program says "one" at 0 and "five" at 8)
  * note sample = "<instrument>-<pitch + pitchOffset (+ key if keyed)>"
  * piano plays both hands: also pitch + leftHandPitchOffset
  * gain = instrument volume x note velocity
  * main.json maps a sample to [_, start, length] in 44.1 kHz samples;
    attacks land within 1 ms of the indexed starts (checked)

    from beatmachine import Machine
    m = Machine()
    audio, times, counts = m.render({"clave": 0, "bass": 0, "piano": 1}, eights=48)
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "data/beatmachine"
SR_SRC, SR = 44100, 22050
BM = "{http://www.salsabeatmachine.org/xns/bm}"
INS = "{http://www.salsabeatmachine.org/xns/instruments}"


class Machine:
    def __init__(self, assets=ASSETS):
        root = ET.parse(assets / "salsa.xml").getroot()
        self.instruments = {}
        for inst in root.find(BM + "instrumentList"):
            name = inst.tag.replace(INS, "").lower()
            f = lambda k, d: inst.findtext(INS + k, d)
            progs = inst.find(INS + "programs")
            self.instruments[name] = {
                "enabled": f("enabled", "false") == "true",
                "active": int(f("activeProgram", "0")),
                "keyed": f("keyedInstrument", "false") == "true",
                "offset": int(f("pitchOffset", "0")),
                "both_hands": f("playBothHands", "false") == "true",
                "left_offset": int(f("leftHandPitchOffset", "0")),
                "volume": float(f("volume", "1.0")),
                "programs": [{
                    "title": p.get("title"), "length": int(p.get("length")),
                    "notes": [(int(n.get("index")), int(n.get("pitch")),
                               float(n.get("velocity") or 1.0)) for n in p.findall(BM + "Note")],
                } for p in (progs.findall(BM + "Program") if progs is not None else [])],
            }
        sprite = json.load(open(assets / "main.json"))
        self.sprite = sprite.get("spritemap", sprite)
        bank, sr = sf.read(assets / "main.mp3", dtype="float32")
        assert sr == SR_SRC
        self.bank = bank if bank.ndim == 1 else bank.mean(1)

    def default_setup(self):
        """The site's default line-up, without the instructor (it says the count)."""
        return {k: v["active"] for k, v in self.instruments.items() if v["enabled"] and k != "instructor"}

    def sample(self, name):
        _, start, length = self.sprite[name]
        return self.bank[start:start + length]

    def render(self, setup, eights=48, bpm=180, key=0, rotate=None, lead=0.5, jitter_ms=0.0, seed=0):
        """setup: {instrument: program index}. rotate: {instrument: beats} (e.g. 4 -> 3-2 clave).

        Returns audio at 22.05 kHz, beat times (one per beat, plus one past
        the end for beat-synchronous features), and 0-indexed counts (0 = 1).
        jitter_ms: optional random timing jitter per note (humanising).
        """
        rng = np.random.default_rng(seed)
        rotate = rotate or {}
        eighth = 30.0 / bpm
        n_beats = 8 * eights
        out = np.zeros(int((lead + n_beats * 2 * eighth + 2.0) * SR_SRC), dtype=np.float32)
        for name, prog in setup.items():
            inst = self.instruments[name]
            p = inst["programs"][prog]
            shift = 2 * rotate.get(name, 0)                    # beats -> eighths
            for j in range(n_beats * 2):
                for idx, pitch, vel in p["notes"]:
                    if idx != (j - shift) % p["length"]:
                        continue
                    pitches = [pitch + (key if inst["keyed"] else 0)]
                    names = [f"{name}-{pitches[0] + inst['offset']}"]
                    if inst["both_hands"]:
                        names.append(f"{name}-{pitches[0] + inst['left_offset']}")
                    for s in names:
                        clip = self.sample(s) * inst["volume"] * vel
                        t = lead + j * eighth + (rng.normal(0, jitter_ms / 1000) if jitter_ms else 0)
                        i0 = max(0, int(round(t * SR_SRC)))
                        out[i0:i0 + len(clip)] += clip[:len(out) - i0]
        peak = np.abs(out).max()
        if peak > 0:
            out *= 0.9 / peak
        audio = out[::2].copy()                                  # 44.1 -> 22.05 kHz
        times = lead + np.arange(n_beats + 1) * 2 * eighth
        counts = (np.arange(n_beats) % 8).astype(np.int8)
        return audio, times, counts


# --- Model inputs, exactly as for real songs ---------------------------------
# Copies of build_features / build_chroma's computations (those are untracked:
# they import the grid decoder). Verified identical on a real song.
FPB = 16
MEL = dict(n_fft=2048, hop=256, n_mels=128, f_min=30, f_max=10000)
CHR = dict(n_fft=8192, hop=512, f_max=2000.0)


def _beat_sync(feat, beat_times, fps, fpb=FPB):
    frames = beat_times * fps
    idx = np.empty((len(beat_times) - 1) * fpb, dtype=np.int32)
    for i in range(len(beat_times) - 1):
        pos = np.linspace(frames[i], frames[i + 1], fpb, endpoint=False)
        idx[i * fpb:(i + 1) * fpb] = np.clip(np.round(pos), 0, len(feat) - 1)
    return feat[idx]


def logmel(audio):
    """[T, 128] log-mel at 22.05 kHz, as build_features.process."""
    import torch, torchaudio
    tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=MEL["n_fft"], hop_length=MEL["hop"],
                                              n_mels=MEL["n_mels"], f_min=MEL["f_min"],
                                              f_max=MEL["f_max"], power=1.0)
    spec = tf(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)))
    return torch.log1p(spec * 1000).T.numpy().astype(np.float32)


def chroma(audio):
    """[T, 12] chroma, as build_chroma.chroma_of: 8192-point STFT, harmonic mask."""
    import torch
    from scipy.ndimage import median_filter
    w = torch.hann_window(CHR["n_fft"])
    S = torch.stft(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), CHR["n_fft"],
                   CHR["hop"], window=w, return_complex=True).abs().numpy()
    f = np.fft.rfftfreq(CHR["n_fft"], 1 / SR)
    keep = (f > 25) & (f < CHR["f_max"])
    S, f = S[keep], f[keep]
    H = median_filter(S, size=(1, 17)); P = median_filter(S, size=(17, 1))
    S = S * (H ** 2 / (H ** 2 + P ** 2 + 1e-9))
    pc = np.rint(69 + 12 * np.log2(f / 440.0)).astype(int) % 12
    out = np.zeros((12, S.shape[1]), dtype=np.float32)
    np.add.at(out, pc, S)
    out /= out.sum(0, keepdims=True) + 1e-9
    return out.T


def raw_features(audio, times, with_chroma=True):
    """Beat-synchronous float16 features before per-song normalisation."""
    mel = _beat_sync(logmel(audio), times, SR / MEL["hop"]).astype(np.float16)
    if not with_chroma:
        return mel, None
    return mel, _beat_sync(chroma(audio), times, SR / CHR["hop"]).astype(np.float16)


def features(audio, times, with_chroma=True):
    """Model input as train_phase.load_input builds it: per-song z-scored log-mel
    (+ z-scored chroma, concatenated per frame), float16."""
    mel, ch = raw_features(audio, times, with_chroma)
    z = lambda a: (a.astype(np.float32) - a.astype(np.float32).mean()) / (a.astype(np.float32).std() + 1e-6)
    m = z(mel).astype(np.float16)
    if ch is None:
        return m
    return np.concatenate([m, z(ch).astype(np.float16)], 1)
