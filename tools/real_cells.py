#!/usr/bin/env python
"""D 재측정 청음 페이지용 — unseen8 화자의 **진짜 사람 감정 녹음** 칸마다 1개 고르기 (2026-11-07 · 지시문 q3tts 세션 15:30 §5).

목적: 생성 출력을 "그 사람이 실제로 그 감정을 낼 때"와 귀로 비교(말투 · 문장은 다름 — 비교용).
대상: d_eval plan 의 unseen8 8명 × 7칸. 진짜 녹음이 있는 칸만 1개.
고르는 순서
  1) ~/workspaces/preprocessed/q3tts_koft/q3_extra/val_unseen8/val.jsonl (24kHz · 칸 이름 그대로 · 이 세트는 강도 2 도 B)
  2) 없으면 ~/workspaces/ourdata/aihub_71349/meta/metadata.csv — 화자 = reciter_id · 강도 1 → A · 3 → B · 무감정 = 강도 0
     · 투표 평균 ≥ 4.5 · 2~10초 · 원본 /data1/rohoon/aihub_71349/<audio_path>
  3) 그래도 없으면 "없음"
조건: 평가 문장(plan sents)과 같은 글 · 목표 reference 녹음 · 감정 ref 판(d_eval_eref)의 ref 녹음은 뺌 · seed 1986(칸마다 random.Random(f"1986|화자|칸")) · utt 순 정렬 뒤 고름
출력: ~/workspaces/tmp_listen/vox_koft/d_eval/real_cells.json + real/<utt>.wav(24kHz 로 복사) — d_eval page 가 읽어 "진짜 사람 녹음" 열을 그림
    ~/miniforge3/envs/voxcpm/bin/python tools/real_cells.py
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

H = Path.home() / "workspaces"
D = H / "tmp_listen/vox_koft/d_eval"
Q = H / "preprocessed/q3tts_koft/q3_extra/val_unseen8/val.jsonl"
META = H / "ourdata/aihub_71349/meta/metadata.csv"
RAW = Path("/data1/rohoon/aihub_71349")
EMO7 = ["Neutrality", "Happy-A", "Happy-B", "Sad-A", "Sad-B", "Angry-A", "Angry-B"]
KO2EN = {"기쁨": "Happy", "슬픔": "Sad", "분노": "Angry"}
clean = lambda t: re.sub(r"[^가-힣a-zA-Z0-9]", "", t or "")  # noqa: E731


def cell_of_meta(m) -> str | None:
    if m["emotion"] == "무감정" and m["intensity"] == "0":
        return "Neutrality"
    if m["emotion"] in KO2EN and m["intensity"] in ("1", "3"):
        return f"{KO2EN[m['emotion']]}-{'A' if m['intensity'] == '1' else 'B'}"
    return None


def main():
    import librosa
    import soundfile as sf
    P = json.load(open(D / "plan.json"))
    spk = [s for s, v in P["spk"].items() if v["src"] == "unseen8"]
    bad_txt = {clean(x["text"]) for x in P["sents"]} | {clean(x["text"]) for x in P["items"]}
    ref_utt = {s: Path(P["spk"][s]["ref"]).stem for s in spk}
    ef = H / "tmp_listen/vox_koft/d_eval_eref/plan.json"                 # 감정 ref 판에서 ref 로 쓴 녹음도 뺌
    eref = {x["ref_utt"] for x in json.load(open(ef))["items"]} if ef.exists() else set()
    q = [json.loads(l) for l in open(Q)]
    meta = None
    out, R = {}, D / "real"
    R.mkdir(exist_ok=True)
    for s in spk:
        for e in EMO7:
            ok = lambda u, t: u != ref_utt[s] and u not in eref and clean(t) not in bad_txt  # noqa: E731
            c1 = sorted([x for x in q if x["speaker"] == s and x["emotion"] == e and ok(x["utt"], x["text"])], key=lambda x: x["utt"])
            rng = random.Random(f"1986|{s}|{e}")
            if c1:
                x = rng.choice(c1)
                pick = dict(utt=x["utt"], text=x["text"], src="q3_extra/val_unseen8", audio=x["audio"], dur=x["duration"], n_cand=len(c1))
            else:
                if meta is None:
                    meta = list(csv.DictReader(open(META, encoding="utf-8")))
                rid = str(int(s.split("_")[1]))
                c2 = sorted([m for m in meta if m["reciter_id"] == rid and cell_of_meta(m) == e and float(m["votes_avg"] or 0) >= 4.5
                             and 2.0 <= float(m["duration"] or 0) <= 10.0 and ok(f"71349_{m['file_id']}", m["text_tr"])], key=lambda m: m["file_id"])
                if not c2:
                    out[f"{s}|{e}"] = None
                    continue
                m = rng.choice(c2)
                pick = dict(utt=f"71349_{m['file_id']}", text=m["text_tr"], src="metadata.csv", audio=str(RAW / m["audio_path"]),
                            dur=float(m["duration"]), n_cand=len(c2), intensity=m["intensity"], votes=float(m["votes_avg"]))
            d = R / f"{pick['utt']}.wav"
            if not d.exists():
                y, _ = librosa.load(pick["audio"], sr=24000)
                sf.write(d, y, 24000)
            pick["file"] = f"real/{d.name}"
            out[f"{s}|{e}"] = pick
    json.dump(dict(note="진짜 사람 녹음(비교용 · 말투 · 문장 다름) — tools/real_cells.py", cells=out), open(D / "real_cells.json", "w"),
              ensure_ascii=False, indent=1)
    for s in spk:
        print(s, " · ".join(f"{e}:{'없음' if out[f'{s}|{e}'] is None else out[f'{s}|{e}']['src'][:4]}" for e in EMO7))
    print("✅", D / "real_cells.json", "· 채운 칸", sum(v is not None for v in out.values()), "/", len(out))


if __name__ == "__main__":
    main()
