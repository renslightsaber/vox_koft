"""A7NX2Rs6 메타데이터 만들기(2026-11-07 15시 · q3tts_koft 세션 작성 · 사용자 결정 "542 줄인 판 추가")
A7NX2Rs6 = A7NX2Rs 에서 **542 만 화자당 12 → 6문장**(화자 992명 그대로 · A7N 행은 A7NX2Rs 와 똑같음).
왜: X2R02 의 감정 ↓ 는 542 화자(무감정 낭독만 녹음한 사람)에 몰려 있음(D 재측정 542 A칸 e2v LoRA01 42% → X2R02 s4500 19%,
    unseen8 은 오히려 ↑) → 가설 = 542 학습 행("낭독 목소리 ref → 늘 무감정 정답")이 만든 지름길.
    A7NX2Rs(542 5.8%)와 **542 양만 다르게** 해서 이 가설을 확인(542 약 3% · 무감정 비율도 함께 13.2 → 약 10.6%).
    542 정답은 같은 seed · 같은 순서에서 앞 6개만 → A7NX2Rs 542 행의 부분집합.
--- 아래는 A7NX2Rs 원래 설명 ---
A7NX2Rs 메타데이터 만들기(2026-11-07 10시 · q3tts_koft 세션 작성 · 사용자 결정 "A7NX2Rs")
A7NX2Rs = A7NX2R 에서 **무감정 정답(A7N 무감정 · 542)의 ref 만 2개 → 1개**. 감정 정답은 A7NX2R 과 똑같음(같은 seed · 같은 ref).
왜: A7NX2R(lr 1e-4 · 5e-5)은 목소리 최고지만 감정 ↓ — 무감정 비율 14.5% → 23.2% 로 감정 연습량이 약 10% 줄어든 탓으로 추정
    → 542 화자 992명 · 문장은 그대로 두고 반복만 줄여 무감정 약 13% 로(근거: q3tts_koft docs/explanations/20261107_opinion_A7NX2R_A7NX3.md)
--- 아래는 A7NX2R 원래 설명 ---
A7NX2R 메타데이터 만들기(2026-11-07 · q3tts_koft 세션 작성 · 사용자 결정 "A7NX3 수정안")
왜: A7NX3(정답마다 다른 칸 ref 3개 · 화자 96명)는 처음 보는 화자 loss(val_unseen)가 s1500 부터 오름 = 같은 정답을 여러 번 보며
    학습 화자 96명에 맞춰짐(외움). 고치는 두 갈래:
      ① 정답 반복 줄이기 → ref 를 3개 → **2개**(학습은 정답당 약 1.2번만 보게 step 을 정함)
      ② 화자 수 늘리기 → **542 무감정 1,000명 × 12개**를 섞음(같은 화자의 다른 문장 ref · Neutrality 지시문)
    + 사용자 요청: 음질이 별로인 녹음은 정답 · ref 모두 뺌.
규칙
  · A7N 쪽 = tools/make_a7nx.py 와 같은 규칙(같은 화자 · 정답과 다른 칸 · 2~10초 · 자기 아님 · 무감정 0.4 / A 0.4 / B 0.2)
    + 음질: 정답 · ref 모두 bad_snr · bad_hf(먹먹함) 걸린 녹음 제외. SQUIM · DNSMOS 는 센 감정을 나쁘게 채점하므로 감정 녹음에는 안 씀
      (근거: q3tts_koft docs/reports/20261106_audio_quality_survey.md)
  · 542 쪽 = 음질 bad_any(SNR · 먹먹함 · SQUIM · DNSMOS) 없는 녹음 중 화자마다 12개(seed 1986) · 2~15초
    ref = 같은 화자 · 다른 문장 · 2~10초 · bad_any 없음 · 정답마다 서로 다른 2개 · 지시문 Neutrality(행의 emotion)
    (542/train.jsonl 은 처음 보는 542 화자 142명 · val_unseen8 이 이미 빠진 상태 — voxko/data/prepare.py · 여기서도 다시 확인)
  · val_xref = A7NX 의 val_xref.jsonl 그대로(192 · 이전 run 과 같은 잣대로 비교)
출력(두 프로젝트 · 경로만 다름): ~/workspaces/preprocessed/{vox_koft,q3tts_koft}/A7NX2Rs/{train.jsonl, val_xref.jsonl, dataset_info.json, _READY}
~/miniforge3/envs/q3tts/bin/python tools/make_a7nx2rs6.py
"""
import collections, csv, json, random, shutil, time
from pathlib import Path
H = Path.home() / "workspaces"; V = H / "preprocessed/vox_koft"; Q = H / "preprocessed/q3tts_koft"
META = H / "outputs/q3tts_koft/eval/audio_quality/meta"
SEED, MIN_D, MAX_D, K, N542, MAX542 = 1986, 2.0, 10.0, 2, 6, 15.0
W = {"N": 0.4, "A": 0.4, "B": 0.2}
NAME = "A7NX2Rs6"
K_NEU = 1                                   # 무감정 정답(A7N 무감정 · 542)의 ref 수 — 감정 정답은 K(2)
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
kind = lambda e: "N" if e == "Neutrality" else e[-1]
T = lambda v: v in ("True", "true", "1")


def bad_set(files, cols):
    out = set()
    for f in files:
        for r in csv.DictReader(open(META / f, encoding="utf-8")):
            if any(T(r[c]) for c in cols):
                out.add(r["utt"])
    return out


def main():
    bad_a7n = bad_set(["a7n_train.csv", "ref_neutral466.csv"], ["bad_snr", "bad_hf"])
    bad_542 = bad_set(["s542_train.csv"], ["bad_any"])
    tr = jl(V / "A7N/train.jsonl"); pool = jl(V / "A7N/ref_pool.jsonl")
    q_pool = {r["utt"]: r for r in jl(Q / "q3_extra/ref_neutral466/refs.jsonl")}
    # ---- A7N 쪽(make_a7nx.py 와 같은 후보 규칙 + 음질) ----
    cand = collections.defaultdict(list); cells = collections.defaultdict(set)
    for r in tr:
        cells[r["speaker"]].add(r["emotion"])
        if MIN_D <= r["duration"] <= MAX_D and r["utt"] not in bad_a7n:
            cand[r["speaker"]].append(dict(utt=r["utt"], audio=r["audio"], audio24=r["src_audio"], emotion=r["emotion"], duration=r["duration"]))
    for p in pool:
        if p["utt"] in {c["utt"] for c in cand[p["speaker"]]}:
            continue
        cells[p["speaker"]].add("Neutrality")
        if MIN_D <= p["duration"] <= MAX_D and p["utt"] not in bad_a7n:
            cand[p["speaker"]].append(dict(utt=p["utt"], audio=p["audio"], audio24=q_pool.get(p["utt"], {}).get("audio"),
                                           emotion="Neutrality", duration=p["duration"]))
    keep_spk = {s for s, c in cells.items() if len(c) >= 2}

    def pick(r, k, salt, cs):
        by = collections.defaultdict(list)
        for c in cs:
            by[kind(c["emotion"])].append(c)
        rng = random.Random(f"{SEED}|{salt}|{r['utt']}"); out, used = [], set()
        for _ in range(k):
            avail = {g: [c for c in v if c["utt"] not in used] for g, v in by.items()}
            avail = {g: v for g, v in avail.items() if v}
            if not avail:
                break
            gs = sorted(avail); g = rng.choices(gs, weights=[W[x] for x in gs])[0]
            c = avail[g][rng.randrange(len(avail[g]))]; used.add(c["utt"]); out.append(c)
        return out

    rows, n_bad_tgt = [], 0
    for r in tr:
        if r["speaker"] not in keep_spk:
            continue
        if r["utt"] in bad_a7n:
            n_bad_tgt += 1; continue
        cs = [c for c in cand[r["speaker"]] if c["emotion"] != r["emotion"] and c["utt"] != r["utt"]]
        for j, c in enumerate(pick(r, K_NEU if r["emotion"] == "Neutrality" else K, "train2", cs)):
            rows.append(dict(r, src="A7N", utt_x=f"{r['utt']}__x{j}", ref_utt=c["utt"], ref_audio=c["audio"], ref_audio24=c["audio24"],
                             ref_emotion=c["emotion"], ref_kind=kind(c["emotion"]), ref_duration=c["duration"]))
    # ---- 542 쪽 ----
    s542 = jl(V / "542/train.jsonl"); ref542 = jl(V / "542/ref_pool.jsonl")
    unseen = {json.loads(l)["speaker"] for f in (V / "542/val_unseen542").glob("*.jsonl") for l in open(f) if l.strip()} \
        if (V / "542/val_unseen542").exists() else set()
    unseen |= {json.loads(l)["speaker"] for l in open(V / "A7N/val_unseen8/val.jsonl") if l.strip()}
    by_spk = collections.defaultdict(list); rp = collections.defaultdict(list)
    for r in s542:
        if r["utt"] not in bad_542 and 2.0 <= r["duration"] <= MAX542:
            by_spk[r["speaker"]].append(r)
    for p in ref542:
        if p["utt"] not in bad_542 and MIN_D <= p["duration"] <= MAX_D:
            rp[p["speaker"]].append(p)
    n542_tgt, short_spk, leak = 0, 0, 0
    for spk in sorted(by_spk):
        if spk in unseen:
            leak += 1; continue
        rng = random.Random(f"{SEED}|542|{spk}")
        cs = sorted(by_spk[spk], key=lambda r: r["utt"]); rng.shuffle(cs)
        took = 0
        for r in cs:
            if took >= N542:
                break
            refs = [p for p in rp[spk] if p["utt"] != r["utt"]]
            if len(refs) < K:                   # 정답 고르기는 A7NX2R 과 같게(ref 2개 후보가 있는 문장)
                continue
            rr = random.Random(f"{SEED}|542ref|{r['utt']}"); sel = rr.sample(sorted(refs, key=lambda p: p["utt"]), K)[:K_NEU]
            for j, c in enumerate(sel):
                rows.append(dict(r, src="542", person=r["speaker"], utt_x=f"{r['utt']}__x{j}", ref_utt=c["utt"], ref_audio=c["audio"],
                                 ref_audio24=str(Q / "542/wavs" / (c["utt"] + ".wav")), ref_emotion="Neutrality", ref_kind="N",
                                 ref_duration=c["duration"]))
            took += 1; n542_tgt += 1
        short_spk += took < N542
    # ---- 검증 ----
    chk = dict(same_cell_ref=sum(r["ref_emotion"] == r["emotion"] and r["src"] == "A7N" for r in rows),
               self_ref=sum(r["ref_utt"] == r["utt"] for r in rows),
               bad_target=sum(r["utt"] in (bad_a7n if r["src"] == "A7N" else bad_542) for r in rows),
               bad_ref=sum(r["ref_utt"] in (bad_a7n if r["src"] == "A7N" else bad_542) for r in rows),
               unseen_leak=sum(r["speaker"] in unseen for r in rows),
               dup_pair=len(rows) - len({(r["utt"], r["ref_utt"]) for r in rows}))
    ref_spk = {}
    for r in tr: ref_spk[r["utt"]] = r["speaker"]
    for p in pool: ref_spk[p["utt"]] = p["speaker"]
    for p in ref542: ref_spk[p["utt"]] = p["speaker"]
    chk["other_speaker_ref"] = sum(ref_spk.get(r["ref_utt"]) != r["speaker"] for r in rows)
    tg = lambda src: len({r["utt"] for r in rows if r["src"] == src})
    emo = collections.Counter(r["emotion"] for r in rows)
    emo_of = {r["utt"]: r["emotion"] for r in rows}
    info = dict(name=NAME, created_kst=time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + 9 * 3600)),
                why="A7NX2Rs 의 542 줄인 판(사용자 11/07 15시): 542 화자당 12 → 6문장(992명 그대로) · X2R02 감정 ↓ 가 542 화자에 몰림 → '542 = 늘 무감정' 지름길 가설을 542 양만 바꿔 확인",
                rule=dict(a7n="make_a7nx.py 규칙 · 정답 · ref 음질 bad_snr|bad_hf 제외", refs_per_target=dict(emotion=K, neutral=K_NEU), weights=W,
                          s542=f"bad_any 없는 녹음 · 화자마다 {N542}개 · 2~{MAX542}초 · ref = 같은 화자 다른 문장 {K_NEU}개 · Neutrality 지시문",
                          val_xref="A7NX/val_xref.jsonl 그대로(192)", seed=SEED),
                counts=dict(rows=len(rows), rows_a7n=sum(r["src"] == "A7N" for r in rows), rows_542=sum(r["src"] == "542" for r in rows),
                            targets_a7n=tg("A7N"), targets_542=tg("542"), dropped_bad_targets_a7n=n_bad_tgt,
                            speakers_a7n=len({r["speaker"] for r in rows if r["src"] == "A7N"}),
                            speakers_542=len({r["speaker"] for r in rows if r["src"] == "542"}), speakers_542_short=short_spk,
                            hours_rows=round(sum(r["duration"] for r in rows) / 3600, 2),
                            share_542=round(sum(r["src"] == "542" for r in rows) / len(rows), 4),
                            share_neutral=round(emo["Neutrality"] / len(rows), 4), target_emotion=dict(emo),
                            ref_kind=dict(collections.Counter(r["ref_kind"] for r in rows)),
                            targets_with_fewer_refs=sum(1 for u, n in collections.Counter(r["utt"] for r in rows).items() if n < (K_NEU if emo_of[u] == "Neutrality" else K))),
                checks=chk, code=str(Path(__file__).resolve()))
    assert all(v == 0 for v in chk.values()), chk
    for root in (V, Q):
        d = root / NAME; d.mkdir(parents=True, exist_ok=True)
        with open(d / "train.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                o = dict(r)
                if root == Q:
                    o["audio"], o["ref_audio"] = r["src_audio"], r["ref_audio24"]
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        shutil.copy(root / "A7NX/val_xref.jsonl", d / "val_xref.jsonl")
        json.dump(dict(info, project=("vox_koft · 16kHz" if root == V else "q3tts_koft · 24kHz 원본 경로")),
                  open(d / "dataset_info.json", "w"), ensure_ascii=False, indent=1)
        (d / "_READY").write_text(info["created_kst"] + "\n")
    print(json.dumps(info["counts"], ensure_ascii=False, indent=1)); print("checks", chk)


if __name__ == "__main__":
    main()
