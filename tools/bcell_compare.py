#!/usr/bin/env python
"""B칸 판(A7NUX2sNp) 출력 비교 — 분노B · 기쁨B 높이 · 갈라짐 · 세기 (2026-11-09 · 지시문 q3tts 세션 11/09 09:30 §2-2)

같은 D 재측정 문항(16명 × 칸마다 2문장 = 32개)에서 이름표끼리 비교. 새 합성 없음(d_eval 음원).
  - 높이: REF 대비 반음 중앙값 · +8 넘는 비율 — tools/iv3_compare.py 와 같은 규칙(REF 60~800Hz · 출력 [REF/2, REF +20반음] · 유성 0.3초 미만 뺌)
  - 갈라짐: d_eval quality 의 clip(|y| ≥ 0.999 비율) > 1e-4 인 출력 비율 · 최대 진폭 ≥ 0.99 비율 · SQUIM PESQ 평균
  - 세기: e2v 칸 감정 맞힘(angry · happy) · 그 확률 · arousal
  - 목소리: 목표 유사도(WavLM · ERes2Net)

  ~/miniforge3/envs/voxcpm/bin/python tools/bcell_compare.py --tags NUX2sNlr5e5_s3000 NUX2sNplr5e5_s3000 …   → bcell_pick/out_table.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import angryB_pitch as AB  # noqa: E402

H = Path.home() / "workspaces"
D = H / "tmp_listen/vox_koft/d_eval"
OUT = H / "tmp_listen/vox_koft/bcell_pick"
CELLS = ["Angry-B", "Happy-B", "Angry-A", "Happy-A"]
E2V = {"Angry": "angry", "Happy": "happy"}
F0_UP, MIN_VOICED, CLIP_FRAC = 20, 0.3, 1e-4


def peak(p):
    import soundfile as sf
    return float(np.abs(sf.read(str(p), dtype="float32")[0]).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    P = json.load(open(D / "plan.json"))
    f = OUT / "out_f0.json"
    F = json.load(open(f)) if f.exists() else {}
    with Pool(32) as pool:
        if "REF" not in F:
            F["REF"] = {s: v for s, (_, v, _) in zip(P["spk"], pool.map(AB.f0_med, [(v["ref"], 60.0, 800.0) for v in P["spk"].values()]))}
        jobs, keys = [], []
        for t in a.tags:
            for x in P["items"]:
                k = f"{t}|{x['id']}"
                if x["emo"] in CELLS and k not in F:
                    fr = F["REF"][x["tgt"]]
                    jobs.append((str(D / "wav" / t / f"{x['id']}.wav"), fr / 2, fr * 2 ** (F0_UP / 12)))
                    keys.append(k)
        for k, (p, v, vs) in zip(keys, pool.map(AB.f0_med, jobs)):
            F[k] = [v, vs, peak(p)]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(F, open(f, "w"))
    out = {}
    for t in a.tags:
        sv, ser, q = (json.load(open(D / f"{n}_{t}.json")) for n in ("sv", "ser", "quality"))
        for c in CELLS:
            xs = [x for x in P["items"] if x["emo"] == c]
            st = [12 * math.log2(F[f"{t}|{x['id']}"][0] / F["REF"][x["tgt"]]) for x in xs
                  if F[f"{t}|{x['id']}"][1] >= MIN_VOICED and np.isfinite(F[f"{t}|{x['id']}"][0])]
            e = E2V[c.split("-")[0]]
            out[f"{t}|{c}"] = dict(n=len(xs), n_f0=len(st), st_med=float(np.median(st)), st_mean=float(np.mean(st)),
                                   over8=float(100 * np.mean(np.array(st) > 8)),
                                   clip=float(100 * np.mean([q[x["id"]]["clip"] > CLIP_FRAC for x in xs])),
                                   peak99=float(100 * np.mean([F[f"{t}|{x['id']}"][2] >= 0.99 for x in xs])),
                                   pesq=float(np.nanmean([q[x["id"]]["pesq"] for x in xs])),
                                   hit=float(100 * np.mean([ser[x["id"]]["top"] == e for x in xs])), p_emo=float(np.mean([ser[x["id"]][e] for x in xs])),
                                   arousal=float(np.mean([ser[x["id"]]["arousal"] for x in xs])),
                                   t_sim=float(np.mean([sv[x["id"]]["t_sim"] for x in xs])), e_t_sim=float(np.mean([sv[x["id"]]["e_t_sim"] for x in xs])),
                                   cer=float(100 * np.mean([ser[x["id"]]["cer"] for x in xs])))
    json.dump(out, open(OUT / "out_table.json", "w"), ensure_ascii=False, indent=1)
    for k, v in out.items():
        print(f"{k:28s} F0 {v['st_med']:+.1f}(+8 {v['over8']:.0f}%) · 클리핑 {v['clip']:.0f}% · peak≥.99 {v['peak99']:.0f}% · PESQ {v['pesq']:.2f} · "
              f"맞힘 {v['hit']:.0f}% · p {v['p_emo']:.2f} · aro {v['arousal']:.2f} · 유사도 {v['t_sim']:.3f}/{v['e_t_sim']:.3f} · CER {v['cer']:.1f}")


if __name__ == "__main__":
    main()
