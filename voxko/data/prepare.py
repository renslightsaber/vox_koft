"""q3tts_koft 전처리 데이터 → VoxCPM2 학습 데이터 (16kHz wav + JSONL + 무감정 ref 풀).

    python -m voxko.data.prepare a7n  [--limit 100] [--workers 32]          # A7N 감정 7칸 (+ 처음 보는 화자 8명 val)
    python -m voxko.data.prepare r542 [--limit 100] [--workers 32]          # AIHub 542 무감정 (섞기용 · 선택)

산출물 (기본 ~/workspaces/preprocessed/vox_koft/<이름>/, --limit 이면 <이름>_smoke<N>/):
    wavs16k/<utt>.wav      16kHz · mono · PCM_16 (soxr HQ 로 24k → 16k)
    train.jsonl / val.jsonl  utt · audio · text · emotion · speaker · gender · duration · src_audio
    ref_pool.jsonl         화자별 **무감정** ref 후보(학습 때 고름 — 이 파일엔 길이 거르기 없음)
    val_unseen8/{val.jsonl, ref_pool.jsonl}   (a7n 만) 처음 보는 화자 8명 — 평가용 · 학습 제외
    dataset_info.json      건수 · 화자 · 감정 분포 · 원천 · 시각

규칙 (2026-11-05 사용자 결정)
  · 지시문은 여기서 text 에 붙이지 않는다(학습 때 configs/emotion/<표>.yaml 로 붙임 → 표를 바꿔도 재변환 불필요).
  · ref 풀 = 같은 화자 Neutrality 문장(A7N 안) + q3_extra/ref_neutral466(466 화자 무감정 ref 전용).
    둘 다 없는 화자(_ani 배역 · 무감정 녹음이 없는 화자)는 풀이 비어 있다 → 학습 때 ref 없이.
  · val = 화자마다 고정 표본 2문장(seed 1986). val 문장은 train · ref 풀에서 빠진다.
  · 🔴 처음 보는 화자 8명(q3_extra/val_unseen8)은 train 에 있으면 중단.
  · 원천(~/workspaces/preprocessed/q3tts_koft)은 읽기만 한다. 이미 있는 16k wav 는 건너뛴다(이어하기).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

SRC = Path(os.path.expanduser("~/workspaces/preprocessed/q3tts_koft"))
A7N_JSONL = SRC / "A7/preprocessed_outputs/train_cleanend.codes.jsonl"
REF466_JSONL = SRC / "q3_extra/ref_neutral466/refs.jsonl"
UNSEEN8_DIR = SRC / "q3_extra/val_unseen8"
R542_JSONL = SRC / "542/train_cleanend.codes.jsonl"
V542_REFS = SRC / "542/val_unseen542/refs.jsonl"     # 542 처음 보는 화자 142명
OUT_ROOT = Path(os.path.expanduser(os.environ.get("VOX_DATA_ROOT", "~/workspaces/preprocessed/vox_koft")))
SR = 16000
SEED = 1986
KEEP = ("utt", "text", "raw_text", "emotion", "speaker", "gender", "age", "src_corpus", "person")


def read_jsonl(p) -> list:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def _convert(job):
    """(src, dst) → 16kHz mono PCM_16. 반환 (dst, 길이초, 오류)."""
    import numpy as np
    import soundfile as sf
    import soxr
    src, dst = job
    try:
        if os.path.exists(dst):
            info = sf.info(dst)
            if info.samplerate == SR and info.frames > 0:
                return dst, info.frames / SR, None
        wav, sr = sf.read(src, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = soxr.resample(wav, sr, SR, quality="HQ")
        wav = np.clip(wav, -1.0, 1.0)
        tmp = dst + ".tmp.wav"
        sf.write(tmp, wav, SR, subtype="PCM_16")
        os.replace(tmp, dst)
        return dst, len(wav) / SR, None
    except Exception as e:  # noqa: BLE001
        return dst, 0.0, f"{type(e).__name__}: {e}"


def convert_all(rows: list, wav_dir: Path, workers: int, what: str) -> list:
    """rows 의 audio 를 16k 로 바꾸고 audio·src_audio·duration 을 채운 새 행 목록(실패 행 제외)."""
    wav_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r["audio"], str(wav_dir / f"{r['utt']}.wav")) for r in rows]
    out, fails, t0 = [], [], time.time()
    with Pool(workers) as pool:
        for i, (r, (dst, dur, err)) in enumerate(zip(rows, pool.imap(_convert, jobs, chunksize=64))):
            if err:
                fails.append((r["utt"], err))
                continue
            out.append({**{k: r[k] for k in KEEP if k in r}, "audio": dst, "src_audio": r["audio"],
                        "duration": round(dur, 3)})
            if (i + 1) % 10000 == 0:
                print(f"  [{what}] {i + 1}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
    print(f"  [{what}] 16k 변환 {len(out)}/{len(rows)} · 실패 {len(fails)} · {time.time() - t0:.0f}s", flush=True)
    for u, e in fails[:10]:
        print(f"    ✗ {u}: {e}")
    return out


def pick_val(rows: list, per_speaker: int, seed: int, min_rows: int = 10) -> set:
    """화자마다 per_speaker 문장을 고정 seed 로 고른다(감정이 겹치지 않게 먼저 다른 감정에서).
    문장이 min_rows 개 미만인 화자는 val 을 뽑지 않는다(전부 val 로 가서 train 에서 사라지지 않게 — aihub_0024_ani 2문장)."""
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["speaker"]].append(r)
    val = set()
    for spk in sorted(by):
        if len(by[spk]) < min_rows:
            continue
        rs = sorted(by[spk], key=lambda r: r["utt"])
        rng.shuffle(rs)
        seen = set()
        for r in rs:                                   # 서로 다른 감정 우선
            if len(seen) < per_speaker and r["emotion"] not in seen:
                seen.add(r["emotion"])
                val.add(r["utt"])
        for r in rs:
            if sum(1 for x in rs if x["utt"] in val) >= per_speaker:
                break
            val.add(r["utt"])
    return val


def summary(rows: list) -> dict:
    return {"rows": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 2),
            "speakers": len({r["speaker"] for r in rows}),
            "emotion": dict(sorted(collections.Counter(r.get("emotion") for r in rows).items()))}


def unseen8_speakers() -> list:
    return list(json.loads((UNSEEN8_DIR / "dataset_info.json").read_text())["speakers"])


def build_a7n(args) -> Path:
    out = OUT_ROOT / ("A7N" if not args.limit else f"A7N_smoke{args.limit}")
    src = read_jsonl(A7N_JSONL)
    unseen = set(unseen8_speakers())
    leak = sorted({r["speaker"] for r in src if r["speaker"] in unseen} | {r.get("person") for r in src if r.get("person") in unseen})
    if leak:
        sys.exit(f"❌ 처음 보는 화자 8명이 A7N 에 있습니다: {leak} — 학습에서 빼야 합니다")
    bad = [r["utt"] for r in src if "(" in r["text"] or ")" in r["text"]]
    if bad:
        sys.exit(f"❌ 본문에 괄호가 있는 문장 {len(bad)}개(지시문과 헷갈림): {bad[:5]}")
    if args.limit:                                     # smoke: 화자 6명(고정 seed)에서 limit 문장 — 화자당 val 2 를 빼도 train 이 남게
        rng = random.Random(SEED)
        spk6 = set(rng.sample(sorted({r["speaker"] for r in src}), 6))
        src = [r for r in src if r["speaker"] in spk6]
        src = rng.sample(src, min(args.limit, len(src)))
    ref466 = read_jsonl(REF466_JSONL)
    if args.limit:
        spk = {r["speaker"] for r in src}
        ref466 = [r for r in ref466 if r["speaker"] in spk][: args.limit]
    print(f"A7N {len(src)}문장 · ref_neutral466 {len(ref466)}문장 → {out}")

    rows = convert_all(src, out / "wavs16k", args.workers, "A7N")
    refs = convert_all(ref466, out / "wavs16k", args.workers, "ref466")
    val_utts = pick_val(rows, args.val_per_speaker, SEED)
    train = [r for r in rows if r["utt"] not in val_utts]
    val = [r for r in rows if r["utt"] in val_utts]
    pool = [{"utt": r["utt"], "audio": r["audio"], "speaker": r["speaker"], "duration": r["duration"],
             "text": r["text"], "source": "A7N"} for r in train if r["emotion"] == "Neutrality"]
    pool += [{"utt": r["utt"], "audio": r["audio"], "speaker": r["speaker"], "duration": r["duration"],
              "text": r["text"], "source": "ref_neutral466"} for r in refs]
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "ref_pool.jsonl", pool)

    # 처음 보는 화자 8명 (평가 전용) — 문장 76 · ref 8(무감정)
    uv = read_jsonl(UNSEEN8_DIR / "val.jsonl")
    ur = read_jsonl(UNSEEN8_DIR / "refs.jsonl")
    if args.limit:
        uv = uv[:16]
    uvd = out / "val_unseen8"
    uv16 = convert_all(uv, uvd / "wavs16k", args.workers, "unseen8 val")
    ur16 = convert_all(ur, uvd / "wavs16k", args.workers, "unseen8 ref")
    write_jsonl(uvd / "val.jsonl", uv16)
    write_jsonl(uvd / "ref_pool.jsonl", [{"utt": r["utt"], "audio": r["audio"], "speaker": r["speaker"],
                                          "duration": r["duration"], "text": r["text"], "source": "val_unseen8"} for r in ur16])

    pool_spk = collections.Counter(p["speaker"] for p in pool)
    no_pool = sorted({r["speaker"] for r in train} - set(pool_spk))
    info = {"created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "seed": SEED, "sample_rate": SR,
            "limit": args.limit, "val_per_speaker": args.val_per_speaker,
            "source": {"a7n": str(A7N_JSONL), "ref_neutral466": str(REF466_JSONL), "val_unseen8": str(UNSEEN8_DIR)},
            "train": summary(train), "val": summary(val),
            "ref_pool": {"rows": len(pool), "speakers": len(pool_spk),
                         "by_source": dict(collections.Counter(p["source"] for p in pool))},
            "speakers_without_ref_pool": no_pool,
            "val_unseen8": {"val": summary(uv16) if uv16 else {}, "refs": len(ur16)},
            "rule": "지시문은 학습 때 붙임 · ref = 같은 화자 무감정(없으면 ref 없이) · val 화자당 고정 표본"}
    (out / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("train", "val", "ref_pool")}, ensure_ascii=False))
    print(f"ref 풀이 없는 화자 {len(no_pool)}명(ref 없이 학습): {', '.join(no_pool[:30])}")
    return out


def build_542(args) -> Path:
    out = OUT_ROOT / ("542" if not args.limit else f"542_smoke{args.limit}")
    src = read_jsonl(R542_JSONL)
    unseen = set(unseen8_speakers()) | {r["speaker"] for r in read_jsonl(V542_REFS)}   # + 542 처음 보는 화자 142명(D안 평가 목표)
    leak = sorted({r["speaker"] for r in src if r["speaker"] in unseen})
    if leak:
        sys.exit(f"❌ 처음 보는 화자(val_unseen8 · val_unseen542)가 542 학습 원천에 있습니다: {leak[:10]}")
    if args.limit:                                     # smoke: 화자 10명(고정 seed)에서 limit 문장 — ref 풀(같은 화자 다른 문장)이 생기게
        rng = random.Random(SEED)
        spk10 = set(rng.sample(sorted({r["speaker"] for r in src}), 10))
        src = rng.sample([r for r in src if r["speaker"] in spk10], min(args.limit, sum(r["speaker"] in spk10 for r in src)))
    print(f"542 {len(src)}문장 → {out}")
    rows = convert_all(src, out / "wavs16k", args.workers, "542")
    for r in rows:
        r["emotion"] = "Neutrality"
    write_jsonl(out / "train.jsonl", rows)
    pool = [{"utt": r["utt"], "audio": r["audio"], "speaker": r["speaker"], "duration": r["duration"],
             "text": r["text"], "source": "542"} for r in rows]
    write_jsonl(out / "ref_pool.jsonl", pool)
    info = {"created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "seed": SEED, "sample_rate": SR,
            "limit": args.limit, "source": str(R542_JSONL), "train": summary(rows),
            "rule": "무감정 전용 · ref 풀 = 같은 화자 다른 문장"}
    (out / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(info["train"], ensure_ascii=False))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["a7n", "r542"])
    ap.add_argument("--limit", type=int, default=0, help="smoke: 이 수만큼만(고정 seed 표본) → <이름>_smoke<N>/")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--val_per_speaker", type=int, default=2)
    args = ap.parse_args(argv)
    (build_a7n if args.which == "a7n" else build_542)(args)


if __name__ == "__main__":
    main()
