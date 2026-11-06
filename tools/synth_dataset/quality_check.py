"""SREF 합성 데이터 '깨짐' 검사(2026-11-05) — env q3tts · GPU 1장
  · SQUIM(torchaudio, 정답 없이 STOI · PESQ · SI-SDR 추정) — 하위 몇 %를 '의심'으로 표시
  · 긴 무음(문장 중간 0.8초 이상) · 클리핑(|x| ≥ 0.999 표본 비율)
CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/q3tts/bin/python quality_check.py --out <OUT>
→ <OUT>/quality.json · <OUT>/quality_summary.json (거르기에는 아직 안 씀 — 청음으로 기준을 정한 뒤 반영)"""
import argparse, json
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--low_pct", type=float, default=5.0)
a = ap.parse_args(); od = Path(a.out)
from torchaudio.pipelines import SQUIM_OBJECTIVE
m = SQUIM_OBJECTIVE.get_model().cuda().eval(); res = {}
ids = [json.loads(l)["id"] for l in open(od / "sentences.jsonl")]
for i in ids:
    p = od / "wav24" / f"{i}.wav"
    if not p.exists():
        continue
    y, sr = sf.read(p, dtype="float32")
    x = torchaudio.functional.resample(torch.from_numpy(y), sr, 16000)
    with torch.no_grad():
        stoi, pesq, sisdr = m(x[None].cuda())
    hop = int(0.01 * sr); n = len(y) // hop
    fr = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-12
    loud = 20 * np.log10(fr / (np.abs(y).max() + 1e-12)) > -40
    idx = np.where(loud)[0]; gap = 0.0
    if len(idx) > 1:
        gap = float(np.diff(idx).max() - 1) * 0.01          # 말소리 사이 가장 긴 무음(초)
    res[i] = dict(stoi=float(stoi), pesq=float(pesq), sisdr=float(sisdr), max_gap=round(gap, 2), clip=float((np.abs(y) >= 0.999).mean()))
json.dump(res, open(od / "quality.json", "w"), indent=1)
v = {k: np.array([r[k] for r in res.values()]) for k in ("stoi", "pesq", "sisdr")}
thr = {k: float(np.percentile(v[k], a.low_pct)) for k in v}
sus = {i: [k for k in thr if r[k] < thr[k]] + (["긴 무음"] if r["max_gap"] >= 0.8 else []) + (["클리핑"] if r["clip"] > 1e-4 else []) for i, r in res.items()}
sus = {i: w for i, w in sus.items() if w}
summ = dict(n=len(res), low_pct=a.low_pct, thr=thr, mean={k: float(v[k].mean()) for k in v}, suspect=len(sus),
            by_reason={k: sum(k in w for w in sus.values()) for k in ["stoi", "pesq", "sisdr", "긴 무음", "클리핑"]})
json.dump(dict(summ, suspect_ids=sus), open(od / "quality_summary.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(summ, ensure_ascii=False))
