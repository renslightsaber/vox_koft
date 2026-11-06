"""SREF v2 만들기(2026-11-05) — v1 합격분에서 '깨짐 의심'을 거르고, 보충 합성분(topup)을 더해 3,000개 이상
거르기(사용자 결정 11/05 16:40): v1 청음 페이지의 **'깨짐 의심' 전부 제외** — SQUIM STOI · PESQ · SI-SDR 가 v1 하위 5%(같은 문턱값을 topup 에도)
· 문장 중간 무음 ≥ 0.8초 · 클리핑 · 그리고 사용자 청음 '기타' 제외
~/miniforge3/envs/q3tts/bin/python make_sref_v2.py"""
import csv, json, random, time
from collections import Counter
from pathlib import Path
H = Path.home() / "workspaces"; S = H / "preprocessed/q3tts_koft/SREF"; V1, V2 = S / "v1", S / "v2"
THR = json.load(open(V1 / "quality_summary.json"))["thr"]; GAP_MAX, N_VAL = 0.8, 60
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
J = {i: j for i, j in list(csv.reader(open(H / "tmp_listen/datasets/sref_v1/sref_v1_judgments.csv", encoding="utf-8-sig")))[1:]}
rows, drop = [], Counter()
for src in (V1, V2 / "topup"):
    Q = json.load(open(src / "quality.json"))
    for r in jl(src / "train.jsonl"):
        q = Q[r["utt"]]
        if J.get(r["utt"]) == "기타": drop["청음 기타"] += 1; continue
        why = [k for k in ("stoi", "pesq", "sisdr") if q[k] < THR[k]] + (["긴 무음"] if q["max_gap"] >= GAP_MAX else []) + (["클리핑"] if q["clip"] > 1e-4 else [])
        if why: drop["/".join(why)] += 1; continue
        r["squim"] = {k: round(q[k], 3) for k in ("pesq", "stoi", "sisdr")}; r["max_gap"] = q["max_gap"]; r["src_version"] = src.name
        rows.append(r)
v1val = {r["utt"] for r in jl(V1 / "sref_val.jsonl")}
val = [r for r in rows if r["utt"] in v1val]
rest = [r for r in rows if r["utt"] not in v1val]
random.Random(1986).shuffle(rest)
val += rest[: max(0, N_VAL - len(val))]; vid = {r["utt"] for r in val}
tr = [r for r in rows if r["utt"] not in vid]
for n, rr in (("sref_train.jsonl", tr), ("sref_val.jsonl", val)):
    with open(V2 / n, "w", encoding="utf-8") as f:
        for r in rr: f.write(json.dumps(r, ensure_ascii=False) + "\n")
info = dict(name="SREF v2", created_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)),
            base="v1 합격분 + topup(A7N 260 · 542 260 새 문장, 같은 설정 · 프로파일 qwen3tts_sref)",
            filter=dict(squim_min=THR, max_gap_s=GAP_MAX, clip="> 1e-4", user_judgment_drop="기타", dropped=dict(drop),
                        why="사용자 결정: v1 의 깨짐 의심(SQUIM 하위 5% · 긴 무음 · 클리핑)은 전부 제외(청음 21개 중 기타 12)"),
            counts=dict(train=len(tr), val=len(val), total=len(rows), by_kind=dict(Counter(r["kind"] for r in rows)),
                        hours=round(sum(r["duration"] for r in rows) / 3600, 2)),
            v1_info="../v1/dataset_info.json", topup_info="topup/dataset_info.json")
json.dump(info, open(V2 / "dataset_info.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(info, ensure_ascii=False))
