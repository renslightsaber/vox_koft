"""A7NX · A7NX3 메타데이터 만들기(2026-11-06 · q3tts_koft 세션 작성 · 사용자 아이디어) — "다른 칸 ref → 정답 칸"(같은 사람)
왜: 지금 A7N 학습은 "무감정 ref → 감정 정답" 한 방향뿐이고, 무감정이 없는 화자는 ref 없이(vox) 학습한다.
    → 센 감정에서 목소리가 학습 화자 쪽으로 끌림(분석: docs/reports/20261106_cell_voice_drift_analysis.md).
    같은 사람의 **다른 칸 녹음**을 ref 로 주고 정답 칸 지시문으로 학습하면 "목소리는 ref, 감정은 지시문"을 배운다(가설).
규칙(사용자 결정 11/06 16:2x)
  · 정답 = A7N 학습 행 중 칸(중립 포함)이 2개 이상인 화자(96명) · 칸 1개뿐인 화자 15명은 뺌
  · ref = 같은 화자 · **정답과 다른 칸** · 길이 2~10초 · 자기 자신 아님. 무감정 후보 = 학습 Neutrality + ref 풀(466 무감정)
  · ref 칸 고르기 비율 = 무감정 0.4 · A 칸 0.4 · B 칸 0.2(없는 묶음은 빼고 다시 나눔) → 그 묶음 안에서 고르게
  · 항상 ref(ref 없는 행 0) · 지시문은 학습 때 정답 칸 것(기존과 같음)
  · A7NX  = 정답마다 ref 1개 고정(1차 · 3,000 step ≈ 0.9 epoch 라 "매번 무작위"와 사실상 같음)
  · A7NX3 = 정답마다 서로 다른 ref 3개(2차 · 3배로 펼침) · 후보가 3개 미만이면 있는 만큼만
  · seed 1986 · 결정적(같은 입력 → 같은 결과)
출력(두 프로젝트 같은 내용 · 경로만 다름)
  · vox_koft: ~/workspaces/preprocessed/vox_koft/{A7NX,A7NX3}/train.jsonl(16kHz · ref_utt · ref_audio · ref_emotion · ref_kind) + val_xref.jsonl(감정 ref 평가용) + dataset_info.json
  · q3tts_koft: ~/workspaces/preprocessed/q3tts_koft/{A7NX,A7NX3}/train.jsonl(24kHz 원본 경로) + dataset_info.json — 학습 코드는 아직 없음(나중에 q3tts 에서 쓸 때)
~/miniforge3/envs/q3tts/bin/python tools/make_a7nx.py
"""
import collections, json, random, time
from pathlib import Path
H = Path.home() / "workspaces"; V = H / "preprocessed/vox_koft"; Q = H / "preprocessed/q3tts_koft"
SEED, MIN_D, MAX_D = 1986, 2.0, 10.0
W = {"N": 0.4, "A": 0.4, "B": 0.2}
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
kind = lambda e: "N" if e == "Neutrality" else e[-1]


def main():
    tr = jl(V / "A7N/train.jsonl"); pool = jl(V / "A7N/ref_pool.jsonl"); val = jl(V / "A7N/val.jsonl")
    q_pool = {r["utt"]: r for r in jl(Q / "q3_extra/ref_neutral466/refs.jsonl")}
    # 후보: 학습 행(자기 칸) + ref 풀(무감정)
    cand = collections.defaultdict(list); cells = collections.defaultdict(set)
    for r in tr:
        cells[r["speaker"]].add(r["emotion"])
        if MIN_D <= r["duration"] <= MAX_D:
            cand[r["speaker"]].append(dict(utt=r["utt"], audio=r["audio"], audio24=r["src_audio"], emotion=r["emotion"], duration=r["duration"]))
    for p in pool:
        if p["utt"] in {c["utt"] for c in cand[p["speaker"]]}:
            continue
        cells[p["speaker"]].add("Neutrality")
        if MIN_D <= p["duration"] <= MAX_D:
            a24 = q_pool.get(p["utt"], {}).get("audio")
            cand[p["speaker"]].append(dict(utt=p["utt"], audio=p["audio"], audio24=a24, emotion="Neutrality", duration=p["duration"]))
    keep_spk = {s for s, c in cells.items() if len(c) >= 2}

    def pick(r, k, salt):
        cs = [c for c in cand[r["speaker"]] if c["emotion"] != r["emotion"] and c["utt"] != r["utt"]]
        by = collections.defaultdict(list)
        for c in cs:
            by[kind(c["emotion"])].append(c)
        rng = random.Random(f"{SEED}|{salt}|{r['utt']}"); out, used = [], set()
        for _ in range(k):
            avail = {g: [c for c in v if c["utt"] not in used] for g, v in by.items()}
            avail = {g: v for g, v in avail.items() if v}
            if not avail:
                break
            gs = sorted(avail); w = [W[g] for g in gs]
            g = rng.choices(gs, weights=w)[0]; c = avail[g][rng.randrange(len(avail[g]))]
            used.add(c["utt"]); out.append(c)
        return out

    def build(name, k, rows, salt):
        out = []
        for r in rows:
            if r["speaker"] not in keep_spk:
                continue
            for j, c in enumerate(pick(r, k, salt)):
                out.append(dict(r, utt_x=f"{r['utt']}__x{j}", ref_utt=c["utt"], ref_audio=c["audio"], ref_audio24=c["audio24"],
                                ref_emotion=c["emotion"], ref_kind=kind(c["emotion"]), ref_duration=c["duration"]))
        return out

    info_all = {}
    for name, k in (("A7NX", 1), ("A7NX3", 3)):
        rows = build(name, k, tr, "train")
        vx = build(name, 1, val, "val")                       # 감정 ref 평가용 검증(기존 val 은 그대로 둠)
        tgt = len({r["utt"] for r in rows})
        info = dict(name=name, created_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)),
                    rule=dict(ref="같은 화자 · 정답과 다른 칸 · 2~10초 · 자기 아님 · 항상 ref", weights=W, refs_per_target=k, seed=SEED,
                              drop="칸(중립 포함) 1개뿐인 화자", neutral_source="학습 Neutrality + q3_extra/ref_neutral466"),
                    why="사용자 아이디어(11/06): 다른 감정 ref 로 나머지 칸을 만들게 학습 → 목소리는 ref · 감정은 지시문(떼어 놓기)",
                    counts=dict(rows=len(rows), targets=tgt, speakers=len({r["speaker"] for r in rows}),
                                dropped_speakers=sorted(set(cells) - keep_spk), dropped_rows=sum(1 for r in tr if r["speaker"] not in keep_spk),
                                hours_targets=round(sum(r["duration"] for r in rows) / 3600, 2),
                                target_emotion=dict(collections.Counter(r["emotion"] for r in rows)),
                                ref_kind=dict(collections.Counter(r["ref_kind"] for r in rows)),
                                ref_emotion=dict(collections.Counter(r["ref_emotion"] for r in rows)),
                                same_cell_ref=sum(r["ref_emotion"] == r["emotion"] for r in rows), self_ref=sum(r["ref_utt"] == r["utt"] for r in rows),
                                targets_with_fewer_refs=sum(1 for u, n in collections.Counter(r["utt"] for r in rows).items() if n < k),
                                val_xref=len(vx)),
                    source=dict(train=str(V / "A7N/train.jsonl"), pool=str(V / "A7N/ref_pool.jsonl")),
                    code=str(Path(__file__).resolve()))
        for root, a_key, ra_key in ((V, "audio", "ref_audio"), (Q, "src_audio", "ref_audio24")):
            d = root / name; d.mkdir(parents=True, exist_ok=True)
            with open(d / "train.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    o = dict(r)
                    if root == Q:
                        o["audio"], o["ref_audio"] = r["src_audio"], r["ref_audio24"]
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            with open(d / "val_xref.jsonl", "w", encoding="utf-8") as f:
                for r in vx:
                    o = dict(r)
                    if root == Q:
                        o["audio"], o["ref_audio"] = r.get("src_audio", r["audio"]), r["ref_audio24"]
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
            json.dump(dict(info, project=("vox_koft · 16kHz" if root == V else "q3tts_koft · 24kHz 원본 경로")), open(d / "dataset_info.json", "w"), ensure_ascii=False, indent=1)
        info_all[name] = info["counts"]
    print(json.dumps(info_all, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
