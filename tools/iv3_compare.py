#!/usr/bin/env python
"""분노B: 재학습(지시문 v3) vs 서비스 문구만 바꾸기(C2) — 같은 D 재측정 문항에서 견주기 (2026-11-09 · 지시문 q3tts 세션 11/08 23:25 §2)

물음: "재학습이 필요한가, 서비스 문구만 바꾸면 되나"
  - sN      = A7NUX2sN s3000 + v1 분노B 문구(지금) — d_eval 음원
  - sN+C2   = A7NUX2sN s3000 + 분노A 문구(11/08 시험 A 의 C2) — **새로 합성**(D 재측정 분노B 32문항 = 16명 × 2문장)
  - iv3     = A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3 + v3 분노B 문구(학습과 같음) — d_eval 음원(tag NUX2sNiv3lr5e5_s*)
  - iv3+v1  = 같은 ckpt + v1 문구(참고) — d_eval 음원(tag NUX2sNiv3v1_s*)
잴 것(분노B 32문항): F0 REF 대비 반음(시험 A 와 같은 pyin 범위 규칙) · 목표 유사도(WavLM · ERes2Net) · 목표 1위 · e2v angry 맞힘 · CER
합성 조건은 d_eval vox() 와 같음(Engine · REF = plan spk ref · seed 1986 · cfg 2.0 · timesteps 10 · max_len 600 · 준비 합성).
재현: 같은 엔진으로 v1 문구 문항 1개를 다시 만들어 d_eval wav 와 같아야(길이 같음 · max diff < 1e-3) 새 합성을 함.

  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/voxcpm/bin/python tools/iv3_compare.py synth
  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/q3tts/bin/python tools/iv3_compare.py measure --what sv
  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/q3tts_ser/bin/python tools/iv3_compare.py measure --what ser
  ~/miniforge3/envs/voxcpm/bin/python tools/iv3_compare.py f0 --tags NUX2sNlr5e5_s3000 NUX2sNiv3lr5e5_s<best> NUX2sNiv3v1_s<best> …
  ~/miniforge3/envs/voxcpm/bin/python tools/iv3_compare.py table --tags …   → iv3_cmp/table.json
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
import listen_pages as LP  # noqa: E402

OUT = LP.TL / "iv3_cmp"
SRC = "NUX2sNlr5e5_s3000"
C2TAG = "NUX2sNc2_s3000"
C2 = "Coldly irritated, restrained anger, firm and clipped"
CELLS = ["Angry-B", "Angry-A"]
F0_UP, MIN_VOICED = 20, 0.3


def plan():
    P = LP.jl(LP.D / "plan.json")
    P["items"] = [x for x in P["items"] if x["emo"] == "Angry-B"]
    return P


def cmd_synth(a):
    import soundfile as sf
    from voxko.engine.synth import Engine
    P = plan()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(P, open(OUT / "plan_c2.json", "w"), ensure_ascii=False, indent=1)
    model = LP.jl(LP.D / f"vox_{SRC}.json")["model"]
    eng = Engine(model, emotion_map=str(LP.EMAP))                                 # 준비 합성 포함
    x = P["items"][0]
    w = eng.synth(eng.compose(x["text"], emotion=x["emo"], normalize=False), ref=P["spk"][x["tgt"]]["ref"], seed=LP.SEED, max_len=LP.MAXLEN)
    w0 = sf.read(LP.D / "wav" / SRC / f"{x['id']}.wav")[0]
    ok = LP.same(w, w0)
    print("재현", x["id"], ok, len(w), len(w0), flush=True)
    if not ok:
        raise SystemExit("❌ 재현 실패 — 새 합성 안 함")
    W = OUT / "wav" / C2TAG
    W.mkdir(parents=True, exist_ok=True)
    for x in P["items"]:
        p = W / f"{x['id']}.wav"
        if not p.exists():
            sf.write(p, eng.synth(eng.compose(x["text"], instruct=C2, normalize=False), ref=P["spk"][x["tgt"]]["ref"],
                                  seed=LP.SEED, max_len=LP.MAXLEN), eng.sr)
    json.dump(dict(model=model, instruct=C2, n=len(P["items"]), repro_id=P["items"][0]["id"], repro=ok,
                   text_example=eng.compose(P["items"][0]["text"], instruct=C2, normalize=False)),
              open(OUT / "synth_meta.json", "w"), ensure_ascii=False, indent=1)
    print("✅ 합성", len(P["items"]))


def cmd_measure(a):
    LP.de_measure(OUT, OUT / "plan_c2.json", a.what, [C2TAG])


def f0_med(job):
    import librosa
    path, fmin, fmax = job
    y, sr = librosa.load(str(path), sr=16000)
    f0, vf, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=160)
    v = f0[vf & np.isfinite(f0)]
    return str(path), (float(np.median(v)) if len(v) else float("nan")), float(len(v) * 0.01)


def wav_of(tag, i):
    return (OUT if tag == C2TAG else LP.D) / "wav" / tag / f"{i}.wav"


def cmd_f0(a):
    P = LP.jl(LP.D / "plan.json")
    f = OUT / "f0.json"
    res = json.load(open(f)) if f.exists() else {}
    with Pool(32) as pool:
        if "REF" not in res:
            res["REF"] = {s: v for s, v, _ in [(s,) + f0_med((v["ref"], 60.0, 800.0))[1:] for s, v in P["spk"].items()]}
        jobs, keys = [], []
        for t in a.tags:
            for x in P["items"]:
                if x["emo"] in CELLS and f"{t}|{x['id']}" not in res and wav_of(t, x["id"]).exists():
                    fr = res["REF"][x["tgt"]]
                    jobs.append((str(wav_of(t, x["id"])), fr / 2, fr * 2 ** (F0_UP / 12)))
                    keys.append(f"{t}|{x['id']}")
        for k, (_, v, vs) in zip(keys, pool.map(f0_med, jobs)):
            res[k] = [v, vs]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(f, "w"), indent=1)
    print("F0", len(keys), "개 새로")


def cmd_table(a):
    P = LP.jl(LP.D / "plan.json")
    F = LP.jl(OUT / "f0.json")
    out = {}
    for t in a.tags:
        base = OUT if t == C2TAG else LP.D
        sv, ser = LP.jl(base / f"sv_{t}.json"), LP.jl(base / f"ser_{t}.json")
        for c in CELLS if t != C2TAG else ["Angry-B"]:
            xs = [x for x in P["items"] if x["emo"] == c and x["id"] in sv and x["id"] in ser]
            st = [12 * math.log2(F[f"{t}|{x['id']}"][0] / F["REF"][x["tgt"]]) for x in xs
                  if f"{t}|{x['id']}" in F and F[f"{t}|{x['id']}"][1] >= MIN_VOICED and np.isfinite(F[f"{t}|{x['id']}"][0])]
            grp = {g: [12 * math.log2(F[f"{t}|{x['id']}"][0] / F["REF"][x["tgt"]]) for x in xs
                       if P["spk"][x["tgt"]]["src"] == g and f"{t}|{x['id']}" in F and F[f"{t}|{x['id']}"][1] >= MIN_VOICED
                       and np.isfinite(F[f"{t}|{x['id']}"][0])] for g in ("542", "unseen8")}
            out[f"{t}|{c}"] = dict(n=len(xs), n_f0=len(st), st_med=float(np.median(st)), st_mean=float(np.mean(st)),
                                   over8=float(100 * np.mean(np.array(st) > 8)),
                                   st_542=float(np.median(grp["542"])), st_unseen8=float(np.median(grp["unseen8"])),
                                   t_sim=float(np.mean([sv[x["id"]]["t_sim"] for x in xs])), e_t_sim=float(np.mean([sv[x["id"]]["e_t_sim"] for x in xs])),
                                   top1=float(100 * np.mean([(sv[x["id"]]["top1"] + sv[x["id"]]["e_top1"]) / 2 for x in xs])),
                                   angry=float(100 * np.mean([ser[x["id"]]["top"] == "angry" for x in xs])),
                                   p_angry=float(np.mean([ser[x["id"]]["angry"] for x in xs])),
                                   arousal=float(np.mean([ser[x["id"]]["arousal"] for x in xs])),
                                   cer=float(100 * np.mean([ser[x["id"]]["cer"] for x in xs])))
    json.dump(out, open(OUT / "table.json", "w"), ensure_ascii=False, indent=1)
    for k, v in out.items():
        print(f"{k:34s} n {v['n']:2d}/{v['n_f0']:2d} · F0 REF 대비 중앙 {v['st_med']:+.1f}(542 {v['st_542']:+.1f} · un8 {v['st_unseen8']:+.1f}) · "
              f"+8 {v['over8']:.0f}% · 유사도 {v['t_sim']:.3f}/{v['e_t_sim']:.3f} · 1위 {v['top1']:.0f}% · angry {v['angry']:.0f}% · "
              f"aro {v['arousal']:.2f} · CER {v['cer']:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["synth", "measure", "f0", "table"])
    ap.add_argument("--what", choices=["sv", "ser"])
    ap.add_argument("--tags", nargs="*", default=[])
    a = ap.parse_args()
    {"synth": cmd_synth, "measure": cmd_measure, "f0": cmd_f0, "table": cmd_table}[a.cmd](a)
