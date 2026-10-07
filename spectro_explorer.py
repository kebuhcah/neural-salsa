#!/usr/bin/env python3
"""Build a browser tool for hearing and seeing what lives in each frequency band.

The band-ablation results rest on a mapping from mel bins to instruments that
was asserted from prior knowledge, never checked against this audio -- and at
least one label was wrong (trumpet fundamentals sit in the band labelled
"conga, piano low"). This makes the mapping checkable by ear: view the
spectrogram, band-limit playback live, and listen to what is actually there.

Beat markers from the grid are overlaid, so you can also see which
instruments line up with the 1.

Bands can also be muted individually, which the low/high cut cannot express:
muting high-mid plays the song without 2.5-6 kHz, the band that argues for
the 1<->5 inversion on Amor y Control (NOTES section 13a).

    python3 spectro_explorer.py --song "Lluvia Con Nieve"
    cd data/explorer && python3 -m http.server 8765
    # then open http://localhost:8765/<song>.html

Beat times, counts, title and artist come from data/features/*.npz -- the same
contract the modelling code reads -- so this file has no dependency on the
grid decoder and is tracked. It still needs the local audio (data/audio/) to
play and draw anything, and its output in data/explorer/ stays untracked.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "data/audio"
FEAT = ROOT / "data/features"
OUT = ROOT / "data/explorer"
SR = 22050
BAND_EDGES = [(0, 250), (250, 800), (800, 2500), (2500, 6000), (6000, 11025)]
BAND_NAMES = ["bass", "low-mid", "mid", "high-mid", "high"]


def render_spectrogram(y, png, width=2400, height=700):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_fft, hop = 2048, 512
    win = np.hanning(n_fft)
    n = 1 + (len(y) - n_fft) // hop
    S = np.empty((n_fft // 2 + 1, n), dtype=np.float32)
    for i in range(n):
        S[:, i] = np.abs(np.fft.rfft(y[i * hop:i * hop + n_fft] * win))
    S = 20 * np.log10(S + 1e-6)
    # Percentile floor, not max-80: salsa is dense and a fixed 80dB window
    # saturates into a solid wall. This keeps transients legible.
    lo, hi = np.percentile(S, 55), np.percentile(S, 99.9)
    S = np.clip(S, lo, hi)

    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
              extent=[0, len(y) / SR, 0, SR / 2])
    ax.set_yscale("log")
    ax.set_ylim(40, SR / 2)
    for lo, hi in BAND_EDGES[1:]:
        ax.axhline(lo, color="cyan", lw=1.0, alpha=0.75)
    ticks = [50, 100, 250, 500, 1000, 2000, 4000, 8000]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t}" for t in ticks])
    # mm:ss every 15 s (minor every 5 s), so a time quoted in the notes can be
    # found by eye; plain seconds made "2:15" a mental conversion.
    dur = len(y) / SR
    xt = np.arange(0, dur, 15)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{int(t // 60)}:{int(t % 60):02d}" for t in xt])
    ax.set_xticks(np.arange(0, dur, 5), minor=True)
    ax.set_xlim(0, dur)
    ax.set_ylabel("Hz"); ax.set_xlabel("time (m:ss)")
    # The figure background is dark, and default black tick text vanished.
    ax.tick_params(which="both", colors="#bbb")
    ax.xaxis.label.set_color("#bbb"); ax.yaxis.label.set_color("#bbb")
    for (lo, hi), nm in zip(BAND_EDGES, BAND_NAMES):
        ax.text(0.3, np.sqrt(max(lo, 45) * min(hi, SR / 2)), nm,
                color="cyan", fontsize=10, va="center", fontweight="bold")
    fig.tight_layout(pad=0.4)
    fig.savefig(png, facecolor="#111")
    plt.close(fig)
    # axes box in figure-fraction terms, so the overlay can line up exactly
    p = ax.get_position()
    return {"x0": p.x0, "x1": p.x1, "y0": 1 - p.y1, "y1": 1 - p.y0}


def build(song_query):
    OUT.mkdir(parents=True, exist_ok=True)
    cand = []
    for f in sorted(FEAT.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        title = str(z["title"]) if "title" in z.files else ""
        if song_query.lower() in title.lower():
            cand.append(z)
    if not cand:
        sys.exit(f"no song matching {song_query!r} in {FEAT}")
    if len(cand) > 1:
        print("multiple matches, using the first:")
        for z in cand:
            print(f"   {z['artist']} - {z['title']}")
    z = cand[0]
    title, artist, sid = str(z["title"]), str(z["artist"]), str(z["id"])
    wav = AUDIO / f"{sid}.wav"
    if not wav.exists():
        sys.exit(f"no audio for {title}")

    y, sr = sf.read(wav, dtype="float32")
    slug = "".join(c if c.isalnum() else "-" for c in title).strip("-").lower()
    shutil.copy(wav, OUT / f"{slug}.wav")
    box = render_spectrogram(y, OUT / f"{slug}.png")

    # Counts are stored 0-indexed (0 == count "1"); the page shows 1-8.
    beats = [{"t": round(float(t), 4), "c": int(c) + 1}
             for t, c in zip(z["times"], z["counts"])]
    data = {"title": title, "artist": artist, "dur": len(y) / sr,
            "beats": beats, "box": box,
            "bands": [{"name": n, "lo": lo, "hi": hi}
                      for n, (lo, hi) in zip(BAND_NAMES, BAND_EDGES)]}
    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__SLUG__", slug)
    (OUT / f"{slug}.html").write_text(html)
    print(f"wrote data/explorer/{slug}.html  ({len(beats)} beats)")
    return slug


TEMPLATE = r"""<!doctype html><meta charset=utf-8>
<title>spectro explorer</title>
<style>
 body{background:#111;color:#ddd;font:13px/1.4 system-ui,sans-serif;margin:16px}
 #wrap{position:relative;display:inline-block;cursor:crosshair}
 #spec{display:block;max-width:100%}
 #ov{position:absolute;inset:0;pointer-events:none}
 .row{margin:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 button{background:#222;color:#ddd;border:1px solid #444;padding:5px 11px;
        border-radius:4px;cursor:pointer}
 button:hover{background:#333} button.on{background:#0a6;border-color:#0d8;color:#fff}
 button:disabled{opacity:.4;cursor:default}
 button.muted{background:#722;border-color:#c44;color:#fff;text-decoration:line-through}
 input[type=range]{width:220px}
 #goto{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;padding:4px 6px}
 #cnt{font-size:34px;font-weight:600;min-width:2ch;display:inline-block}
 .lbl{color:#888} #status{color:#fa0}
</style>
<h2 id=ttl></h2>
<div id=wrap><img id=spec src="__SLUG__.png"><canvas id=ov></canvas></div>
<div class=row>
  <button id=play disabled>loading...</button>
  <span class=lbl>count</span><span id=cnt>-</span>
  <span class=lbl id=tpos>0:00.0</span>
  <span class=lbl>go to</span><input id=goto size=6 placeholder="m:ss">
  <span class=lbl id=hov></span>
  <label><input type=checkbox id=click> click on the 1</label>
  <span id=status></span>
</div>
<div class=row id=bands><span class=lbl>band:</span></div>
<div class=row id=mutes><span class=lbl>mute:</span></div>
<div class=row>
  <span class=lbl>low cut</span><input type=range id=lo min=0 max=1 step=0.001 value=0>
  <span id=lov>20 Hz</span>
  <span class=lbl>high cut</span><input type=range id=hi min=0 max=1 step=0.001 value=1>
  <span id=hiv>11025 Hz</span>
</div>
<p class=lbl>Click the spectrogram to seek, or type a time (2:15, or 135) in
"go to" and press Enter. Hovering shows the time under the cursor. Cyan lines are the ablation band
edges. Ticks are beats; tall green ticks are the 1, amber are the 5.
"band" and the cuts select one contiguous range; "mute" removes any
combination of bands (shaded on the spectrogram) and stacks with them.</p>
<script>
const D=__DATA__;
document.getElementById('ttl').textContent=D.artist+' \u2014 '+D.title;
const NY=11025, LOF=20;
const F2S=f=>Math.log(f/LOF)/Math.log(NY/LOF), S2F=s=>LOF*Math.pow(NY/LOF,s);
const btn=document.getElementById('play'), st=document.getElementById('status');

// Decode the whole file up front and drive playback from an AudioBufferSource.
// An <audio> element cannot seek unless the server answers Range requests, and
// python -m http.server does not -- it returns 200 with the entire file, so
// setting currentTime silently snaps back to zero.
let ctx,buf,src,hp,lp,playing=false,offset=0,startedAt=0;
let gains=[]; const muted=D.bands.map(()=>false);
async function load(){
  ctx=new (window.AudioContext||window.webkitAudioContext)();
  hp=ctx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=LOF; hp.Q.value=0.7;
  lp=ctx.createBiquadFilter(); lp.type='lowpass';  lp.frequency.value=NY;  lp.Q.value=0.7;
  hp.connect(lp);
  // Per-band mutes. The cut filters can only pass one contiguous range, so
  // "everything except high-mid" is impossible with them. Instead split the
  // signal at the band edges with 8th-order Linkwitz-Riley crossovers (a
  // 4th-order Butterworth squared: four biquads) and give each band a gain.
  // 24 dB/oct was too shallow -- high-mid is only 1.3 octaves wide and its
  // neighbours leaked it back in at -15 dB. Each lower band also passes
  // through the all-pass of every crossover above it, so the unmuted sum is
  // flat (simulated: +-0.00 dB) rather than dipping ~4 dB near the edges.
  // Muting high-mid removes ~22 dB mid-band; wider bands get 33-64 dB.
  const BWQ=[0.5412,1.3066];                    // 4th-order Butterworth Qs
  const biq=(type,f,q)=>{const b=ctx.createBiquadFilter(); b.type=type;
    b.frequency.value=f; b.Q.value=q; return b;};
  const series=(from,nodes)=>nodes.reduce((a,n)=>(a.connect(n),n),from);
  const lr8=(type,f,from)=>series(from,
    [...BWQ,...BWQ].map(q=>biq(type,f,q)));
  const edges=D.bands.slice(0,-1).map(b=>b.hi);
  let rest=lp;
  gains=D.bands.map((b,i)=>{
    const gn=ctx.createGain(); gn.gain.value=muted[i]?0:1;
    if(i<edges.length){
      const later=edges.slice(i+1).flatMap(e=>BWQ.map(q=>biq('allpass',e,q)));
      series(lr8('lowpass',edges[i],rest),later).connect(gn);
      rest=lr8('highpass',edges[i],rest);
    } else rest.connect(gn);
    gn.connect(ctx.destination); return gn; });
  try{
    const r=await fetch('__SLUG__.wav'); if(!r.ok) throw new Error('HTTP '+r.status);
    buf=await ctx.decodeAudioData(await r.arrayBuffer());
    btn.disabled=false; btn.textContent='play';
  }catch(e){ st.textContent='could not load audio: '+e.message+
    ' (serve over http, not file://)'; }
}
const dur=()=>buf?buf.duration:D.dur;
const now=()=>Math.min(playing?offset+(ctx.currentTime-startedAt):offset, dur());
function start(){ if(!buf)return;
  src=ctx.createBufferSource(); src.buffer=buf; src.connect(hp);
  src.start(0, Math.max(0,Math.min(offset,buf.duration-0.01)));
  startedAt=ctx.currentTime; playing=true; btn.textContent='pause'; }
function halt(){ if(src){ try{src.stop()}catch(e){} src.disconnect(); src=null; } }
function pause(){ if(playing){ offset=now(); halt(); playing=false; btn.textContent='play'; } }
function seek(t){ const was=playing; if(was){halt();playing=false;}
  offset=Math.max(0,Math.min(t,dur()-0.01)); if(was) start(); }
btn.onclick=()=>{ if(!buf)return; ctx.resume();
  if(playing) pause(); else { if(offset>=dur()-0.02) offset=0; start(); } };

D.bands.forEach(b=>{ const el=document.createElement('button');
  el.textContent=b.name+' '+b.lo+'-'+b.hi;
  el.onclick=()=>{setBand(Math.max(b.lo,LOF),b.hi); mark(el);};
  document.getElementById('bands').appendChild(el); });
const allb=document.createElement('button'); allb.textContent='full'; allb.className='on';
allb.onclick=()=>{setBand(LOF,NY); mark(allb)};
document.getElementById('bands').appendChild(allb);
D.bands.forEach((b,i)=>{ const el=document.createElement('button');
  el.textContent=b.name+' '+b.lo+'-'+b.hi;
  el.onclick=()=>{ muted[i]=!muted[i]; el.classList.toggle('muted',muted[i]);
    // short ramp rather than a step, so toggling mid-song does not click
    if(gains[i]) gains[i].gain.setTargetAtTime(muted[i]?0:1,ctx.currentTime,0.01); };
  document.getElementById('mutes').appendChild(el); });
function mark(x){document.querySelectorAll('#bands button').forEach(b=>b.classList.remove('on'));x.classList.add('on');}
function setBand(lo,hi){ if(!hp)return; hp.frequency.value=lo; lp.frequency.value=hi;
  document.getElementById('lo').value=F2S(lo); document.getElementById('hi').value=F2S(hi);
  document.getElementById('lov').textContent=Math.round(lo)+' Hz';
  document.getElementById('hiv').textContent=Math.round(hi)+' Hz'; }
for(const id of ['lo','hi']) document.getElementById(id).oninput=e=>{
  if(!hp)return; const f=S2F(+e.target.value);
  if(id==='lo'){hp.frequency.value=f; document.getElementById('lov').textContent=Math.round(f)+' Hz';}
  else {lp.frequency.value=f; document.getElementById('hiv').textContent=Math.round(f)+' Hz';}
  document.querySelectorAll('#bands button').forEach(b=>b.classList.remove('on')); };

const img=document.getElementById('spec'), ov=document.getElementById('ov'), g=ov.getContext('2d');
function fit(){ ov.width=img.clientWidth; ov.height=img.clientHeight; }
img.onload=fit; window.onresize=fit; if(img.complete)fit();
const px=t=>{const W=ov.width; return D.box.x0*W+(D.box.x1-D.box.x0)*W*(t/D.dur);};
function draw(){
  const W=ov.width,H=ov.height,y0=D.box.y0*H,y1=D.box.y1*H;
  g.clearRect(0,0,W,H);
  // Shade muted bands. The PNG's y axis is log-frequency from 40 Hz to Nyquist.
  const fy=f=>y1-(y1-y0)*Math.log(Math.max(f,40)/40)/Math.log(NY/40);
  const xa=D.box.x0*W, xb=D.box.x1*W;
  g.fillStyle='rgba(0,0,0,0.7)';
  D.bands.forEach((b,i)=>{ if(muted[i]){ const ya=fy(b.hi), yb=fy(b.lo);
    g.fillRect(xa,ya,xb-xa,yb-ya); } });
  const step=D.beats.length>1200?2:1;
  for(let i=0;i<D.beats.length;i+=step){ const b=D.beats[i];
    if(b.c!==1&&b.c!==5&&step>1)continue;
    const x=px(b.t);
    g.strokeStyle=b.c===1?'#0f8':(b.c===5?'#fa0':'#666');
    g.lineWidth=b.c===1?2:1;
    g.beginPath(); g.moveTo(x,y1); g.lineTo(x,y1-(b.c===1?18:(b.c===5?11:6))); g.stroke(); }
  const x=px(now());
  g.strokeStyle='#fff'; g.lineWidth=1.5;
  g.beginPath(); g.moveTo(x,y0); g.lineTo(x,y1); g.stroke();
  if(hoverT!==null){ const hx=px(hoverT);
    g.strokeStyle='rgba(255,255,255,0.35)'; g.lineWidth=1; g.setLineDash([4,4]);
    g.beginPath(); g.moveTo(hx,y0); g.lineTo(hx,y1); g.stroke(); g.setLineDash([]);
    g.fillStyle='#fff'; g.font='12px system-ui';
    g.fillText(fmt(hoverT), Math.min(hx+5,W-50), y0+14); }
}
// Round first, so 59.96 s shows as 1:00.0 rather than 0:60.0.
const fmt=t=>{ t=Math.round(t*10)/10;
  return Math.floor(t/60)+':'+(t%60).toFixed(1).padStart(4,'0'); };
// Fraction along the time axis under the pointer, or null outside the plot.
function frac(e){
  const r=img.getBoundingClientRect(), W=img.clientWidth;
  const x0=D.box.x0*W, x1=D.box.x1*W;
  const f=((e.clientX-r.left)-x0)/(x1-x0);
  return f>=0&&f<=1?f:null;
}
const wrap=document.getElementById('wrap'), hov=document.getElementById('hov');
let hoverT=null;
wrap.addEventListener('click',e=>{ const f=frac(e); if(f!==null) seek(f*D.dur); });
wrap.addEventListener('mousemove',e=>{ const f=frac(e);
  hoverT=f===null?null:f*D.dur; hov.textContent=hoverT===null?'':'cursor '+fmt(hoverT); });
wrap.addEventListener('mouseleave',()=>{ hoverT=null; hov.textContent=''; });
// Accepts "m:ss", "m:ss.s" or plain seconds.
document.getElementById('goto').addEventListener('keydown',e=>{
  if(e.key!=='Enter')return;
  const v=e.target.value.trim(), p=v.split(':');
  const t=p.length===2?(+p[0])*60+(+p[1]):+v;
  if(isFinite(t)&&v!==''){ seek(t); e.target.style.borderColor='#444'; }
  else e.target.style.borderColor='#c44';
});
let last=-1;
function tick(){
  const t=now();
  let i=0,lo=0,hi=D.beats.length-1;
  while(lo<=hi){const m=(lo+hi)>>1; if(D.beats[m].t<=t){i=m;lo=m+1}else hi=m-1;}
  const b=D.beats[i], c=document.getElementById('cnt');
  c.textContent=b?b.c:'-';
  c.style.color=b&&b.c===1?'#0f8':(b&&b.c===5?'#fa0':'#ddd');
  document.getElementById('tpos').textContent=fmt(t);
  if(document.getElementById('click').checked&&b&&i!==last&&b.c===1&&ctx&&playing){
    const o=ctx.createOscillator(), gg=ctx.createGain();
    o.frequency.value=1200; gg.gain.setValueAtTime(0.25,ctx.currentTime);
    gg.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+0.06);
    o.connect(gg); gg.connect(ctx.destination); o.start(); o.stop(ctx.currentTime+0.07);
  }
  last=i; draw(); requestAnimationFrame(tick);
}
load(); tick();
</script>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    build(ap.parse_args().song)
