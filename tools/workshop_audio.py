#!/usr/bin/env python
"""워크샵 발표용 전/후 음원 — VoxCPM2 원본(before) vs NUX2sN s3000(after) (2026-11-09)

지시문: _handoff/20261109_1600_prompt_workshop_audio_full.md(q3tts 세션 작성 · 세 모델 공통 REF · 문장 · seed)
엔진 = 청음 노트북과 같음(voxko.listen.Listener · notebooks/lora/A7NUX2sN_xref_LoRA_01_r32_lr5e5_listening.ipynb):
  입력 "(v1 지시문)문장"(ko_tts_norm 정규화) + reference = 서비스 REF(Controllable cloning · prompt 없음) · cfg 2.0 · timesteps 10 · max_len 600 · 준비 합성 뒤 · bf16
  원본도 같은 "(지시문)문장" 입력(전/후 입력이 같아야 공정).
출력: ~/workspaces/tmp_listen/vox_koft/workshop/{wav/<id>_<before|after>_s<seed>.wav(24kHz 모노), manifest.json, index.html}
  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/voxcpm/bin/python tools/workshop_audio.py
11/09 16:4x 지시문(REF = aihub_0156 · D 재측정과 같은 16k 원본 · 폴더 workshop_0156):
  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/voxcpm/bin/python    tools/workshop_audio.py synth --set 0156
  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/q3tts_ser/bin/python tools/workshop_audio.py check_ser --set 0156   # CER · e2v
  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/q3tts/bin/python     tools/workshop_audio.py check_sv  --set 0156   # 같은 사람 점수(REF 와 · WavLM · ERes2Net)
  ~/miniforge3/envs/voxcpm/bin/python tools/workshop_audio.py page --set 0156      # 깨짐 의심 · 교체 seed 를 페이지에
"""
from __future__ import annotations

import html
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voxko import listen  # noqa: E402

H = Path.home() / "workspaces"
OUT = H / "tmp_listen/vox_koft/workshop"
RUN, STEP = "A7NUX2sN_xref_LoRA_01_r32_lr5e5", 3000
REF = H / "reference_audios/qwen3tts/set2_trim03_rms20/female_ment_natural_24k_trim03_rms20.wav"
REF_TEXT = json.load(open(REF.with_suffix(".json")))["normalized_text"]   # 참고로만 적음(Controllable cloning 은 전사문을 안 씀)
SEEDS = [1986, 1987, 1988]
SR_OUT = 24000
ITEMS = [("N1", "Neutrality", "안녕하세요, 하상일 리더님. 혹시 이게 레퍼런스 오디오와 목소리가 동일한가요?"),
         ("N2", "Neutrality", "안녕하세요, 리더님. 혹시 이게 레퍼런스 오디오와 목소리가 동일한가요?"),
         ("HA", "Happy-A", "드디어 합격했어! 진짜 믿기지가 않아."),
         ("HB", "Happy-B", "드디어 합격했어! 진짜 믿기지가 않아."),
         ("SA", "Sad-A", "코인에 올인했다가 사기당해서 1억을 날렸어. 아직 대출금도 많이 남았는데 ..."),
         ("SB", "Sad-B", "코인에 올인했다가 사기당해서 1억을 날렸어 아직 대출금도 많이 남았는데 ..."),
         ("AA", "Angry-A", "너 자꾸 나 빡치게 할래?"),
         ("AB", "Angry-B", "너 자꾸 나 빡치게 할래?")]
SETS = {"service": dict(out=OUT, ref=REF, ref_text=REF_TEXT, note=""),
        "0156": dict(out=H / "tmp_listen/vox_koft/workshop_0156",
                     ref=H / "preprocessed/vox_koft/A7N/val_unseen8/wavs16k/71349_F-NX-Q-156-0315.wav",   # D 재측정이 합성에 쓴 파일 그대로(16k)
                     ref_text="시아버지는 나이가 많았지만 성격도 좋고 재산도 많았습니다.",
                     note="REF = 처음 보는 남성 화자(학습에 없음) · 같은 문장을 Base 와 학습 모델로")}
KO = {"Neutrality": "중립", "Happy-A": "기쁨A", "Happy-B": "기쁨B", "Sad-A": "슬픔A", "Sad-B": "슬픔B", "Angry-A": "분노A", "Angry-B": "분노B"}


def to24(w, sr):
    import librosa
    return librosa.resample(np.asarray(w, dtype=np.float32), orig_sr=sr, target_sr=SR_OUT)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", nargs="?", default="synth", choices=["synth", "check_ser", "check_sv", "page"])
    ap.add_argument("--set", default="service", choices=list(SETS))
    a = ap.parse_args()
    global OUT, REF, REF_TEXT
    S = SETS[a.set]
    OUT, REF, REF_TEXT = S["out"], Path(S["ref"]), S["ref_text"]
    if a.step == "synth":
        synth(S)
    elif a.step == "page":
        page(json.load(open(OUT / "manifest.json")), S["note"])
    else:
        globals()[a.step]()


def synth(S):
    import soundfile as sf
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    shutil.copy(REF, OUT / "ref.wav")
    t0 = time.time()
    L = listen.Listener(RUN, seed=SEEDS[0], cfg_value=2.0, timesteps=10)
    ref = {"ref": str(REF)}
    man = dict(created_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)), ref=str(REF), ref_text=REF_TEXT,
               ref_text_used=False, mode="Controllable cloning(reference 만 · prompt 없음)", knobs=dict(cfg_value=L.cfg, inference_timesteps=L.ts, max_len=L.max_len,
               normalize="ko_tts_norm(profile=qwen3tts) · VoxCPM normalize=False", dtype="bf16", warmup="준비 합성 1회(모델 로드 뒤)"),
               emotion_map=str(L.emap), models={}, items=[], out_sr=SR_OUT)
    for side, step in (("before", None), ("after", STEP)):
        e = L.engine(step)
        man["models"][side] = dict(model="VoxCPM2 원본" if step is None else f"{RUN} step_{step} export_lora",
                                   path=e.path, step=step, kind=e.kind, sr_native=e.sr)
        for i, emo, text in ITEMS:
            for sd in SEEDS:
                w, sr, full = L.synth(step, text, emotion=emo, ref=ref, seed=sd)
                y = to24(w, sr)
                p = OUT / "wav" / f"{i}_{side}_s{sd}.wav"
                sf.write(p, y, SR_OUT, subtype="PCM_16")
                man["items"].append(dict(id=i, cell=emo, side=side, seed=sd, text=text, input=full, file=p.name, dur=round(len(y) / SR_OUT, 3),
                                         peak=round(float(np.abs(y).max()), 3)))
                print(side, i, sd, f"{len(y) / SR_OUT:.2f}s", full, flush=True)
        # 재현: 같은 엔진으로 첫 문항(seed 1986)을 한 번 더 → 48kHz 원음끼리 비교
        i, emo, text = ITEMS[2]
        w1, _, _ = L.synth(step, text, emotion=emo, ref=ref, seed=SEEDS[0])
        w2, _, _ = L.synth(step, text, emotion=emo, ref=ref, seed=SEEDS[0])
        y0 = sf.read(OUT / "wav" / f"{i}_{side}_s{SEEDS[0]}.wav")[0]
        y1 = to24(w1, e.sr)
        man["models"][side]["repro"] = dict(id=i, seed=SEEDS[0], twice_same=bool(len(w1) == len(w2) and float(np.abs(np.asarray(w1) - np.asarray(w2)).max()) < 1e-3),
                                            same_as_file=bool(len(y1) == len(y0) and float(np.abs(y1 - y0).max()) < 2e-3))
        print(side, "재현", man["models"][side]["repro"], flush=True)
    man["sec"] = round(time.time() - t0)
    json.dump(man, open(OUT / "manifest.json", "w"), ensure_ascii=False, indent=1)
    page(man, S["note"])


def _cer(r, h):
    d = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(h)] / max(1, len(r))


def check_ser():
    """CER(whisper · 정규화 뒤 입력과 비교) · e2v 1위 · 클리핑(|y| ≥ 0.999 비율) → manifest."""
    import re
    import soundfile as sf
    sys.path.insert(0, str(H / "q3tts_koft"))
    from ser.models import e2v_infer, load_16k, load_e2v, load_whisper
    m = json.load(open(OUT / "manifest.json"))
    wh, e2v = load_whisper(), load_e2v()
    cl = lambda t: re.sub(r"[^가-힣a-zA-Z0-9]", "", t)  # noqa: E731
    for x in m["items"]:
        p = str(OUT / "wav" / x["file"])
        w = load_16k(p)
        hyp = wh.transcribe(w, language="ko", temperature=0.0, condition_on_previous_text=False)["text"]
        labs, sc = e2v_infer(e2v, w)
        y = sf.read(p)[0]
        x.update(cer=round(_cer(cl(x["input"].split(")", 1)[1]), cl(hyp)), 3), asr=hyp.strip(), e2v=labs[int(np.argmax(sc))],
                 clip=float((np.abs(y) >= 0.999).mean()))
    json.dump(m, open(OUT / "manifest.json", "w"), ensure_ascii=False, indent=1)


def check_sv():
    """같은 사람 점수 = REF 와의 코사인(WavLM-large · ERes2Net 평균 · d_eval 과 같은 잣대) → manifest."""
    import librosa
    sys.path.insert(0, str(H / "q3tts_koft"))
    from q3ko.eval.sv import load_embedder
    m = json.load(open(OUT / "manifest.json"))
    F = {"w": load_embedder("wavlm_large", device="cuda:0"), "e": load_embedder("eres2net", device="cuda:0")}
    emb = lambda p, k: (lambda v: v / np.linalg.norm(v))(np.asarray(F[k](librosa.load(str(p), sr=16000)[0], 16000), dtype=np.float64))  # noqa: E731
    r = {k: emb(m["ref"], k) for k in F}
    for x in m["items"]:
        x["ref_sim"] = round(float(np.mean([emb(OUT / "wav" / x["file"], k) @ r[k] for k in F])), 4)
    json.dump(m, open(OUT / "manifest.json", "w"), ensure_ascii=False, indent=1)


SIM_DROP = 0.08                                                                   # 같은 줄 · 같은 모델 seed 3개 중앙값보다 이만큼 낮으면 "목소리 의심"


def flags(m):
    """id · side · seed → 깨짐 의심 이유 목록(CER > 10% · 클리핑 > 1e-4 · 같은 사람 점수 낮음)."""
    out = {}
    for x in m["items"]:
        sims = [y["ref_sim"] for y in m["items"] if y["id"] == x["id"] and y["side"] == x["side"] and "ref_sim" in y]
        f = []
        if x.get("cer", 0) > 0.10:
            f.append(f"CER {100 * x['cer']:.0f}%(“{x.get('asr', '')}”)")
        if x.get("clip", 0) > 1e-4:
            f.append(f"클리핑 {100 * x['clip']:.2f}%")
        if "ref_sim" in x and sims and x["ref_sim"] < float(np.median(sims)) - SIM_DROP:
            f.append(f"같은 사람 점수 낮음 {x['ref_sim']:.2f}")
        out[(x["id"], x["side"], x["seed"])] = f
    return out


def page(man, note=""):
    E = html.escape
    FL = flags(man)
    has_chk = any("cer" in x for x in man["items"])
    by = {(x["id"], x["side"], x["seed"]): x for x in man["items"]}
    A = lambda f: f'<audio controls preload="none" src="wav/{f}"></audio>'  # noqa: E731
    h = ["<!doctype html><meta charset=utf-8><title>워크샵 전/후 — VoxCPM2</title><style>body{font-family:sans-serif;max-width:1200px;margin:16px auto;padding:0 10px}"
         "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}th{background:#f3f3f3}.n{color:#666;font-size:.8em}"
         "button{margin:1px}button.on{background:#1565c0;color:#fff}textarea{width:96%}audio{width:240px}</style>",
         "<h2>워크샵 전/후 — VoxCPM2 원본(Base) vs 학습 모델(NUX2sN s3000)</h2>",
         (f"<p style='background:#fff8e1;padding:8px;border:1px solid #f0d58a'><b>{E(note)}</b></p>" if note else ""),
         f"<p>REF(서비스 목소리): <audio controls src='ref.wav'></audio><br><span class=n>{E(man['ref_text'])}</span></p>",
         "<p class=n>입력은 전/후 같음: \"(v1 지시문)문장\" + REF · cfg 2.0 · timesteps 10 · seed 1986(예비 1987 · 1988 은 wav/ 폴더) · 24kHz. "
         "<button onclick='dl()'>CSV 내려받기</button></p>",
         "<table><tr><th>id · 칸 · 문장</th><th>Base(학습 전)</th><th>학습 모델(NUX2sN s3000)</th><th>메모</th></tr>"]
    for i, emo, text in ITEMS:
        cells = []
        for side in ("before", "after"):
            x = by[(i, side, 1986)]
            chk = ""
            if has_chk:
                f = FL[(i, side, 1986)]
                sim = f" · 같은 사람 {x['ref_sim']:.2f}" if "ref_sim" in x else ""
                chk = f"<div class=n>CER {100 * x['cer']:.0f}%{sim}</div>"
                if f:
                    alt = [sd for sd in SEEDS[1:] if not FL[(i, side, sd)]]
                    chk += (f"<div style='color:#b71c1c;font-size:.85em'>⚠️ 자동 검사: {E(' · '.join(f))} → 교체 후보 "
                            + (", ".join(f"<a href='wav/{i}_{side}_s{sd}.wav'>s{sd}</a>" for sd in alt) or "없음") + "</div>")
            cells.append(f"<td>{A(x['file'])}<div class=n>{x['dur']:.1f}초</div>{chk}<div class=b data-id='{i}|{side}'>"
                         "<button data-v='쓸 만함'>쓸 만함</button><button data-v='깨짐'>깨짐</button></div></td>")
        h.append(f"<tr><td><b>{i}</b> · {KO[emo]}<div>{E(text)}</div></td>{''.join(cells)}<td><textarea rows=3 data-id='{i}|memo'></textarea></td></tr>")
    h.append("</table>")
    h.append("""<script>const K='workshop_vox_judg';let S=JSON.parse(localStorage.getItem(K)||'{}');
document.querySelectorAll('audio').forEach(a=>a.volume=0.35);
function save(){localStorage.setItem(K,JSON.stringify(S))}
function draw(){document.querySelectorAll('.b').forEach(d=>d.querySelectorAll('button').forEach(b=>b.classList.toggle('on',S[d.dataset.id]===b.dataset.v)));
 document.querySelectorAll('textarea[data-id]').forEach(t=>t.value=S[t.dataset.id]||'')}
document.querySelectorAll('.b button').forEach(b=>b.onclick=()=>{const id=b.parentElement.dataset.id;if(S[id]===b.dataset.v)delete S[id];else S[id]=b.dataset.v;save();draw()});
document.querySelectorAll('textarea[data-id]').forEach(t=>t.oninput=()=>{if(t.value.trim())S[t.dataset.id]=t.value;else delete S[t.dataset.id];save()});
function q(s){s=(s==null?'':String(s));return /[",\\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
function dl(){const L=['id,모델,판정,메모'];const ids=[...new Set([...document.querySelectorAll('.b')].map(d=>d.dataset.id.split('|')[0]))];
 for(const i of ids){for(const s of ['before','after'])if(S[i+'|'+s])L.push([i,s,S[i+'|'+s],''].map(q).join(','));if(S[i+'|memo'])L.push([i,'','',S[i+'|memo']].map(q).join(','))}
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+L.join('\\n')],{type:'text/csv'}));a.download='vox_koft_workshop.csv';a.click()}
draw();</script>""")
    (OUT / "index.html").write_text("\n".join(h), encoding="utf-8")


if __name__ == "__main__":
    main()
