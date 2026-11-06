"""SREF v3 만들기(2026-11-06) — SREF v2 3,025개 그대로 + 새 합성분(v3/new, 서빙 조각 모양 문장) 중 합격 · 깨짐 의심 없음
거르기(사용자 결정 11/06: v2 와 같이 '불합격 · 의심 모두 제외'):
  · sref_synth.py filter 합격(CER · 목소리 ≥ 0.60 · 끝 잘림 · 말 속도) — 불합격은 이미 train.jsonl 에 없음
  · SQUIM STOI · PESQ · SI-SDR 가 **v1 하위 5% 문턱**(v2 와 같은 값) 미만이면 제외
  · 문장 중간 무음 ≥ 0.8초 · 클리핑(> 1e-4) 제외
검증: v2 검증 60 그대로 + 새 묶음(kind)마다 10개씩
~/miniforge3/envs/q3tts/bin/python make_sref_v3.py"""
import json, random, time
from collections import Counter
from pathlib import Path
H = Path.home() / "workspaces"; S = H / "preprocessed/q3tts_koft/SREF"; V1, V2, V3 = S / "v1", S / "v2", S / "v3"; NEW = V3 / "new"
THR = json.load(open(V1 / "quality_summary.json"))["thr"]; GAP_MAX, N_VAL_KIND = 0.8, 10
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# 1) v2 그대로
v2_tr, v2_va = jl(V2 / "sref_train.jsonl"), jl(V2 / "sref_val.jsonl")
for r in v2_tr + v2_va:
    r.setdefault("src_version", "v2")

# 2) 새 합성분 거르기
Q = json.load(open(NEW / "quality.json")); new, drop = [], Counter()
n_rej = sum(1 for _ in open(NEW / "rejected.jsonl", encoding="utf-8"))
for r in jl(NEW / "train.jsonl"):
    q = Q[r["utt"]]
    why = [k for k in ("stoi", "pesq", "sisdr") if q[k] < THR[k]] + (["긴 무음"] if q["max_gap"] >= GAP_MAX else []) + (["클리핑"] if q["clip"] > 1e-4 else [])
    if why:
        drop["/".join(why)] += 1; continue
    r["squim"] = {k: round(q[k], 3) for k in ("pesq", "stoi", "sisdr")}; r["max_gap"] = q["max_gap"]; r["src_version"] = "v3new"
    new.append(r)

# 3) 검증 = v2 검증 + 새 kind 마다 10
rng = random.Random(1986); va_new = []
for k in sorted({r["kind"] for r in new}):
    rows = sorted([r for r in new if r["kind"] == k], key=lambda r: r["utt"]); rng.shuffle(rows); va_new += rows[:N_VAL_KIND]
vid = {r["utt"] for r in va_new}
tr = v2_tr + [r for r in new if r["utt"] not in vid]; va = v2_va + va_new
V3.mkdir(exist_ok=True)
for n, rr in (("sref_train.jsonl", tr), ("sref_val.jsonl", va)):
    with open(V3 / n, "w", encoding="utf-8") as f:
        for r in rr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
allr = tr + va
info = dict(name="SREF v3", created_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)),
            base="SREF v2 3,025 그대로 + 새 합성(v3/new): 서빙 조각 모양 문장(영어만 조각 · 영어 의문문 · 한국어 조각 · 코드스위치 · 서비스형 단문 · 긴 발화) + A7N · 542 새 문장",
            why="사용자 결정 11/06: 서빙은 발화를 문장 → 한글/영어 묶음으로 조각내 따로 생성 → 조각 모양 · 영어 의문문 보강 · 수량 ↑",
            filter=dict(squim_min=THR, max_gap_s=GAP_MAX, clip="> 1e-4", synth_filter_rejected=n_rej, dropped_new=dict(drop),
                        rule="sref_synth filter 불합격 + 깨짐 의심(SQUIM v1 하위 5% · 긴 무음 · 클리핑) 모두 제외"),
            counts=dict(train=len(tr), val=len(va), total=len(allr), new_passed=len(new),
                        by_kind=dict(Counter(r["kind"] for r in allr)), by_src=dict(Counter(r["src_version"] for r in allr)),
                        val_by_kind=dict(Counter(r["kind"] for r in va)), hours=round(sum(r["duration"] for r in allr) / 3600, 2)),
            v2_info="../v2/dataset_info.json", new_info="new/dataset_info.json")
json.dump(info, open(V3 / "dataset_info.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(info, ensure_ascii=False, indent=1))
