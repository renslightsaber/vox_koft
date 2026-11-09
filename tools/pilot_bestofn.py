#!/usr/bin/env python
"""파일럿 데이터셋 — NUX2sN s3000 하나로 "같은 사람이 7칸 감정을 말하는 데이터"를 여러 번 만들어 고르기(best-of-N)로 만들 수 있는가 (2026-11-09)

지시문: _handoff/20261109_1400_prompt_pilot_bestofn_full.md §2 · §3(q3tts 세션 작성 · 사용자 결정 11/09 오후) · 학습 없음
바탕: tools/bestofn.py · bestofn_report.py(11/06 · V3 = V1 + arousal ≥ 다섯 개 중앙값)

세트
  화자 20명(여 10 · 남 10) = 542 처음 보는 화자(val_unseen542 · vox 학습에 안 씀 · 무감정만 있는 사람)
    · d_eval 의 542 8명(JCE · KKH · KHG 포함) + 나머지에서 seed 1986 으로 6 · 6명
    · REF = d_eval 과 같은 규칙(refs.jsonl 의 긴 무감정 ref · 11초 이하 · val 녹음 2개 이상) · REF 음질 하위(SQUIM PESQ 하위 20%)는 뺌
  문장 = 칸마다 10개 · 모든 화자 공통 — 아래 SENT(직접 읽고 칸 감정에 맞는 것만 고름)
    · 감정 6칸: A7N val(이 녹음은 학습에 안 씀 · d_eval 과 같은 출처) · 2~6초 · 그 글이 다른 감정 계열로는 쓰이지 않은 것
    · 중립: 학습 텍스트에 없는 A7N 계열 중립(d_eval 중립 2개 포함) 6개 + 542 val 문장 4개(학습 텍스트에 없음)
  후보 = 항목(화자 · 칸 · 문장)마다 seed 1986~1989(5개면 3시간 넘어 4개 · 지시문) · 입력은 d_eval 과 같음("(v1 지시문)문장" + REF · cfg 2.0 · timesteps 10 · max_len 600 · 준비 합성 뒤)
  기쁨A 후보(같은 화자 · 문장 · seed 로 기쁨A 만): hA_b = NUX2sN s3000 · 더 밝은 추론 문구(학습 표는 그대로) · hA_c = A7NX3 s4500 · hA_d = A7NX3 s2250 · hA_e = X2Rs6 s2500

단계(결과 ~/workspaces/tmp_listen/vox_koft/pilot/)
  CUDA_VISIBLE_DEVICES=2 ~/miniforge3/envs/voxcpm/bin/python tools/pilot_bestofn.py refq        # 후보 REF 음질(SQUIM)
  ~/miniforge3/envs/voxcpm/bin/python tools/pilot_bestofn.py prep                               # 화자 · 문장 · 문항 → plan_main.json · plan_hA.json
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/voxcpm/bin/python tools/pilot_bestofn.py vox --tag main --shard 0/4   (hA_b · hA_c · hA_d · hA_e)
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/voxcpm/bin/python tools/pilot_bestofn.py repro       # d_eval 같은 문항 seed 1986 재현
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts/bin/python     tools/pilot_bestofn.py emb --tag main   # WavLM · ERes2Net
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts_ser/bin/python tools/pilot_bestofn.py ser --tag main --shard 0/2   # e2v · arousal · CER
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/q3tts_ser/bin/python tools/pilot_bestofn.py real_aro  # 진짜 녹음(A7N val) 칸별 arousal 범위
  CUDA_VISIBLE_DEVICES=N ~/miniforge3/envs/voxcpm/bin/python    tools/pilot_bestofn.py quality --tag main --shard 0/2   # d_eval 이상 + SQUIM + 갈라짐 후보
  ~/miniforge3/envs/voxcpm/bin/python tools/pilot_bestofn.py merge        # 조각 합치기
  ~/miniforge3/envs/voxcpm/bin/python tools/pilot_bestofn.py select       # 고르기 → picked_main.jsonl · picked_hA.jsonl · report.json
  ~/miniforge3/envs/voxcpm/bin/python tools/pilot_bestofn.py page         # 청음 페이지(20줄) → pilot/index.html
숫자는 모두 참고용 — 판정은 청음.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
H = Path.home() / "workspaces"
PL = H / "tmp_listen/vox_koft/pilot"
os.environ.setdefault("VOX_D_EVAL_OUT", str(H / "tmp_listen/vox_koft/d_eval"))
from voxko.eval import d_eval as DE  # noqa: E402

D = H / "tmp_listen/vox_koft/d_eval"
CK = H / "outputs/vox_koft/ckpt"
MODELS = {"main": CK / "A7NUX2sN_xref_LoRA_01_r32_lr5e5/step_3000/export_lora",
          "hA_b": CK / "A7NUX2sN_xref_LoRA_01_r32_lr5e5/step_3000/export_lora",
          "hA_c": CK / "A7NX3_xref_LoRA_01_r32/step_4500/export_lora",
          "hA_d": CK / "A7NX3_xref_LoRA_01_r32/step_2250/export_lora",
          "hA_e": CK / "A7NX2Rs6_xref_LoRA_01_r32_lr5e5/step_2500/export_lora"}
HA_LAB = {"main": "(가) NUX2sN s3000 · v1 문구", "hA_b": "(나) NUX2sN s3000 · 밝은 문구", "hA_c": "(다) A7NX3 s4500",
          "hA_d": "(라) A7NX3 s2250", "hA_e": "(마) X2Rs6 s2500"}
D_EVAL_TAG = {"main": "NUX2sNlr5e5_s3000", "hA_c": "A7NX3_s4500", "hA_d": "A7NX3_s2250", "hA_e": "X2Rs6lr5e5_s2500"}
HA_BRIGHT = "Happy and pleased, bright and lively, a warm smile in the voice"     # (나) 추론 때만 · 학습 표(v1)는 그대로
SEEDS = [1986, 1987, 1988, 1989]                                                    # 5개면 합성 3시간 넘음(분당 ~50개) → 지시문대로 4개
CELLS = DE.EMO7                                                                    # 중립 + 감정 6칸
MAX_LEN = 600
REF_PESQ_Q = 20                                                                    # REF 음질 하위 20% 는 뺌
# 감정 6칸 문장 — A7N val(2~6초 · 그 글의 감정 계열이 하나뿐) 정렬 목록에서 직접 읽고 고른 번호(목록은 sent_pool())
SENT_IDX = {"Happy-A": [2, 4, 7, 9, 12, 15, 16, 20, 22, 25], "Happy-B": [0, 2, 3, 5, 8, 9, 10, 12, 13, 15],
            "Sad-A": [1, 6, 7, 8, 9, 10, 11, 13, 18, 20], "Sad-B": [2, 6, 7, 9, 12, 17, 18, 19, 20, 21],
            "Angry-A": [0, 2, 7, 8, 9, 11, 15, 16, 17, 19], "Angry-B": [3, 4, 5, 6, 8, 9, 10, 11, 13, 19]}
NEU_SKIP = {"오늘 점심엔 갈비나 뜯고 골프나 치자!"}                                    # 들뜬 내용이라 중립에서 뺌


def rd(p):
    return DE.rd(p)


def jd(o, p):
    json.dump(o, open(p, "w"), ensure_ascii=False, indent=1)


def shard(xs, s):
    k, n = (int(v) for v in s.split("/"))
    return xs[k::n]


def plan_of(tag):
    return json.load(open(PL / ("plan_main.json" if tag == "main" else "plan_hA.json")))


# ───────────────────────────── refq (env voxcpm · GPU) — 후보 화자 REF 음질
def candidates():
    V = DE.V542
    val_by = defaultdict(list)
    for r in rd(V / "val.jsonl"):
        val_by[r["speaker"]].append(r)
    return {r["speaker"]: dict(r, val=val_by[r["speaker"]]) for r in rd(V / "refs.jsonl")
            if len(val_by[r["speaker"]]) >= 2 and float(r["duration"]) <= 11.0}


def refq(a):
    import soundfile as sf
    import torch
    import torchaudio
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    m = SQUIM_OBJECTIVE.get_model().to("cuda").eval()
    out = {}
    for s, r in sorted(candidates().items()):
        y, sr = sf.read(r["audio"], dtype="float32")
        with torch.no_grad():
            z = torchaudio.functional.resample(torch.from_numpy(y), sr, 16000)[None].cuda()
            stoi, pesq, sisdr = (float(v) for v in m(z))
        out[s] = dict(gender=r["gender"], dur=float(r["duration"]), stoi=stoi, pesq=pesq, sisdr=sisdr)
    PL.mkdir(parents=True, exist_ok=True)
    jd(out, PL / "ref_quality.json")
    p = np.array([v["pesq"] for v in out.values()])
    print(f"후보 {len(out)}명 · PESQ 분위(10/20/50) {np.percentile(p, [10, 20, 50]).round(2)}")


# ───────────────────────────── prep
def sent_pool():
    """감정 6칸 문장 후보 — A7N val · 2~6초 · 그 글이 (val + 학습) 에서 한 감정 계열로만 쓰임 · 글 정렬."""
    val = rd(DE.VD / "val.jsonl")
    tr = rd(H / "preprocessed/vox_koft/A7NUX2sN/train.jsonl")
    bt = defaultdict(set)
    for r in val + tr:
        bt[r["text"]].add(DE.base(r["emotion"]))
    out = {}
    for e in DE.EMO6:
        u = {}
        for r in sorted(val, key=lambda r: r["utt"]):
            if r["emotion"] == e and 2.0 <= float(r["duration"]) <= 6.0 and len(bt[r["text"]]) == 1:
                u.setdefault(r["text"], r["utt"])
        out[e] = sorted(u.items())
    return out, {r["text"] for r in tr}


def prep(a):
    Q = json.load(open(PL / "ref_quality.json"))
    C = candidates()
    thr = float(np.percentile([v["pesq"] for v in Q.values()], REF_PESQ_Q))
    bad = sorted(s for s, v in Q.items() if v["pesq"] < thr)
    DP = json.load(open(D / "plan.json"))
    d8 = [s for s, v in DP["spk"].items() if v["src"] == "542"]
    spk = {}
    rng = random.Random(f"{DE.SEED}|pilot")
    for g in ("female", "male"):
        keep = [s for s in d8 if C[s]["gender"] == g and s not in bad]
        rest = sorted(s for s in C if C[s]["gender"] == g and s not in bad and s not in d8)
        rng.shuffle(rest)
        for s in keep + rest[: 10 - len(keep)]:
            spk[s] = dict(src="542", gender=g, ref=C[s]["audio"], ref_text=C[s]["text"], real=sorted(r["audio"] for r in C[s]["val"]),
                          d_eval=s in d8, ref_pesq=Q[s]["pesq"])
    pool, trt = sent_pool()
    sents = []
    for e in DE.EMO6:
        for i in SENT_IDX[e]:
            t, u = pool[e][i]
            sents.append(dict(emo=e, text=t, src_utt=u, text_src="A7N val"))
    npool, _ = DE._unseen_neutral_pool()
    neu = [r for r in npool if 2.0 <= float(r["duration"]) <= 6.0 and r["text"] not in NEU_SKIP]
    sents += [dict(emo="Neutrality", text=r["text"], src_utt=r["utt"], text_src="A7N(학습 텍스트에 없음)") for r in neu]
    used = {x["text"] for x in sents}
    v = sorted([r for r in rd(DE.V542 / "val.jsonl") if 2.0 <= float(r["duration"]) <= 6.0 and r["text"] not in trt and r["text"] not in used
                and r["speaker"] not in spk], key=lambda r: r["utt"])
    for r in random.Random(f"{DE.SEED}|pilot_neu").sample(v, 10 - len(neu)):
        sents.append(dict(emo="Neutrality", text=r["text"], src_utt=r["utt"], text_src="542 val(학습 텍스트에 없음 · 고른 20명 아닌 화자)"))
    for e in CELLS:
        k = [x for x in sents if x["emo"] == e]
        assert len(k) == 10, (e, len(k))
        for j, x in enumerate(k):
            x["k"] = j
    base = [dict(base_id=f"{s}_{x['emo']}_{x['k']:02d}", tgt=s, emo=x["emo"], text=x["text"], k=x["k"]) for s in spk for x in sents]
    items = [dict(b, id=f"{b['base_id']}_s{sd}", seed=sd) for b in base for sd in SEEDS]
    ha = [x for x in items if x["emo"] == "Happy-A"]
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    common = dict(spk=spk, distract={}, sents=sents, seeds=SEEDS, ref_pesq_thr=thr, ref_bad=bad, created_utc=now)
    jd(dict(common, items=items, model=str(MODELS["main"])), PL / "plan_main.json")
    jd(dict(common, items=ha, models={k: str(v) for k, v in MODELS.items() if k != "main"}, bright=HA_BRIGHT), PL / "plan_hA.json")
    print(f"화자 {len(spk)}명(d_eval {sum(v['d_eval'] for v in spk.values())}) · REF 음질로 뺀 후보 {len(bad)}명(PESQ < {thr:.2f}) · "
          f"d_eval 542 중 빠진 화자 {[s for s in d8 if s in bad]} · 문장 {len(sents)} · main {len(items)} · 기쁨A 후보 {len(ha)} × 4")


# ───────────────────────────── vox (env voxcpm)
def vox(a):
    import soundfile as sf
    from voxko.engine.synth import Engine
    P = plan_of(a.tag)
    W = PL / "wav" / a.tag
    W.mkdir(parents=True, exist_ok=True)
    todo = [x for x in shard(P["items"], a.shard) if not (W / f"{x['id']}.wav").exists()]
    if not todo:
        print(a.tag, a.shard, "할 것 없음")
        return
    eng = Engine(str(MODELS[a.tag]), emotion_map=str(DE.EMAP))                     # 준비 합성 포함
    t0, tot = time.time(), 0.0
    for n, x in enumerate(todo, 1):
        text = eng.compose(x["text"], emotion=x["emo"], instruct=HA_BRIGHT if a.tag == "hA_b" else None, normalize=False)
        w = eng.synth(text, ref=P["spk"][x["tgt"]]["ref"], seed=x["seed"], max_len=MAX_LEN)
        sf.write(W / f"{x['id']}.wav", w, eng.sr)
        tot += len(w) / eng.sr
        if n % 100 == 0:
            print(f"{a.tag} {a.shard} {n}/{len(todo)} · {time.time() - t0:.0f}s · rtf {(time.time() - t0) / tot:.2f}", flush=True)
    meta = dict(tag=a.tag, shard=a.shard, model=str(MODELS[a.tag]), n=len(todo), rtf=(time.time() - t0) / tot, sec=time.time() - t0,
                instruct=HA_BRIGHT if a.tag == "hA_b" else "v1", time_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    jd(meta, PL / f"vox_{a.tag}_{a.shard.replace('/', 'of')}.json")
    print(json.dumps(meta, ensure_ascii=False))


# ───────────────────────────── repro (env voxcpm) — d_eval 같은 문항(같은 화자 · 같은 글 · seed 1986)
def repro(a):
    import soundfile as sf
    from voxko.engine.synth import Engine
    DP = json.load(open(D / "plan.json"))
    out = {}
    for tag, dt in D_EVAL_TAG.items():
        P = plan_of(tag)
        idx = {(x["tgt"], x["text"]): x for x in P["items"] if x["seed"] == DE.SEED}
        pairs = [(x, idx[(x["tgt"], x["text"])]) for x in DP["items"] if (x["tgt"], x["text"]) in idx]
        res = []
        for dx, px in pairs:
            w1, w2 = sf.read(D / "wav" / dt / f"{dx['id']}.wav")[0], sf.read(PL / "wav" / tag / f"{px['id']}.wav")[0]
            res.append(len(w1) == len(w2) and float(np.abs(w1 - w2).max()) < 1e-3)
        out[tag] = dict(d_eval_tag=dt, n=len(res), same=int(sum(res)))
        print(tag, out[tag])
    # 노트북 엔진(새로 불러온 Engine)으로 main 2개 다시 만들기
    P = plan_of("main")
    eng = Engine(str(MODELS["main"]), emotion_map=str(DE.EMAP))
    chk = [x for x in P["items"] if x["seed"] in (1986, 1989) and x["k"] == 3][:: 70][:4]
    rr = []
    for x in chk:
        w = eng.synth(eng.compose(x["text"], emotion=x["emo"], normalize=False), ref=P["spk"][x["tgt"]]["ref"], seed=x["seed"], max_len=MAX_LEN)
        w0 = sf.read(PL / "wav/main" / f"{x['id']}.wav")[0]
        rr.append(dict(id=x["id"], same=bool(len(w) == len(w0) and float(np.abs(w - w0).max()) < 1e-3)))
    out["engine"] = rr
    jd(out, PL / "repro.json")
    print(rr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    fn = globals().get(a.step)
    if fn is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import pilot_bestofn_score as S  # 채점 · 고르기 · 페이지(길어서 따로)
        fn = getattr(S, a.step)
    fn(a)


if __name__ == "__main__":
    main()
