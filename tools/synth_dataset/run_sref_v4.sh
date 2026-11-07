#!/usr/bin/env bash
# SREF v4 만들기(2026-11-07 · q3tts_koft 세션) — 후보 풀(VoxCPM2 seed 3개 + SREF2 s600 seed 2개) → 기존 필터 전부 → 고르기 → v4 → audio_codes
#   GPU 0,1 만 사용(GPU 2,3 은 vox_koft) · 문장: SREF/sentences/v4/*.jsonl(make_sref_v4_sentences.py)
set -euo pipefail
H=~/workspaces; T=$H/vox_koft/tools/synth_dataset; S=$H/preprocessed/q3tts_koft/SREF; NEW=$S/v4/new
PQ=~/miniforge3/envs/q3tts/bin/python; PV=~/miniforge3/envs/voxcpm/bin/python; PS=~/miniforge3/envs/q3tts_ser/bin/python
mkdir -p "$NEW"; cd "$T"
kst(){ TZ=Asia/Seoul date '+%m-%d %H:%M'; }
echo "[$(kst)] 1) pick(정규화 qwen3tts_sref)"
$PQ sref_synth.py pick --out "$NEW" --profile qwen3tts_sref --n_cs 0 --n_run117 0 --n_long 0 --n_a7n 0 --n_542 0 --id_tag V4 \
  --all_files v4/ko_quote.jsonl,v4/en_num.jsonl,v4/retry.jsonl
# retry_of 칸 되살리기(pick 이 버림)
$PQ - <<'E'
import json; from pathlib import Path
H=Path.home()/"workspaces"; S=H/"preprocessed/q3tts_koft/SREF"; NEW=S/"v4/new"
ro={json.loads(l)["id"]:json.loads(l).get("retry_of") for l in open(S/"sentences/v4/retry.jsonl")}
R=[json.loads(l) for l in open(NEW/"sentences.jsonl")]
for r in R:
    if r["id"] in ro: r["retry_of"]=ro[r["id"]]
with open(NEW/"sentences.jsonl","w") as f:
    for r in R: f.write(json.dumps(r,ensure_ascii=False)+"\n")
for c in ("vox_s1","vox_s2","vox_s3"):
    (NEW/c).mkdir(exist_ok=True); (NEW/c/"sentences.jsonl").write_text((NEW/"sentences.jsonl").read_text())
sub=[r for r in R if r["kind"] in ("en_only","codeswitch","en_q_embed")]
for c in ("sref2_s0","sref2_s1"):
    (NEW/c).mkdir(exist_ok=True)
    with open(NEW/c/"sentences.jsonl","w") as f:
        for r in sub: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("후보 문장: VoxCPM2", len(R), "· SREF2", len(sub))
E
echo "[$(kst)] 2) 합성 — GPU0: vox_s1(×2) · vox_s3 0/2 · GPU1: vox_s2(×2) · vox_s3 1/2 · SREF2 s0 → s1"
lane(){ g=$1; shift; for job in "$@"; do c=${job%%:*}; sh=${job#*:}; off=${c#vox_s}
  CUDA_VISIBLE_DEVICES=$g $PV sref_synth.py synth --out "$NEW/$c" --shard $sh --seed_offset $off > "$NEW/$c/synth_${sh/\//of}.log" 2>&1; done; }
lane 0 vox_s1:0/2 vox_s3:0/2 &
lane 0 vox_s1:1/2 &
lane 1 vox_s2:0/2 vox_s3:1/2 &
lane 1 vox_s2:1/2 &
( CUDA_VISIBLE_DEVICES=1 $PQ sref_v4.py sref2 --cand sref2_s0 --seed 1986 > "$NEW/sref2_s0/synth.log" 2>&1 && \
  CUDA_VISIBLE_DEVICES=1 $PQ sref_v4.py sref2 --cand sref2_s1 --seed 1987 > "$NEW/sref2_s1/synth.log" 2>&1 ) &
wait
grep -h "합성" "$NEW"/*/synth*.log || true
echo "[$(kst)] 3) 채점(후보마다 score · quality = GPU0 · asr · asr_en = GPU1)"
( for c in vox_s1 vox_s2 vox_s3 sref2_s0 sref2_s1; do
    CUDA_VISIBLE_DEVICES=0 $PQ sref_synth.py score --out "$NEW/$c" > "$NEW/$c/score.log" 2>&1
    CUDA_VISIBLE_DEVICES=0 $PQ quality_check.py --out "$NEW/$c" > "$NEW/$c/quality.log" 2>&1; done ) &
( for c in vox_s1 vox_s2 vox_s3 sref2_s0 sref2_s1; do
    CUDA_VISIBLE_DEVICES=1 $PS sref_synth.py asr --out "$NEW/$c" > "$NEW/$c/asr.log" 2>&1
    CUDA_VISIBLE_DEVICES=1 $PS sref_v4.py asr_en --cand "$c" > "$NEW/$c/asr_en.log" 2>&1; done ) &
wait
echo "[$(kst)] 3b) UTMOSv2(영어만 조각 · env q3tts_mos 가 있으면)"
if [ -x ~/miniforge3/envs/q3tts_mos/bin/python ] && [ -f "$H/q3tts_koft/tools/diag/utmos_check.py" ]; then
  CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/q3tts_mos/bin/python $H/q3tts_koft/tools/diag/utmos_check.py all > "$NEW/utmos.log" 2>&1 || echo "UTMOS 실패 → 없이 진행"
fi
echo "[$(kst)] 4) 고르기 · v4 만들기"
$PQ sref_v4.py select > "$NEW/select.log" 2>&1; tail -40 "$NEW/select.log"
$PQ sref_v4.py build > "$S/v4/make.log" 2>&1; cat "$S/v4/make.log"
echo "[$(kst)] 5) audio_codes(GPU 0,1)"
cd $H/q3tts_koft
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i $PQ -m q3ko.data.extract_codes --jsonl "$S/v4/sref_train.jsonl" "$S/v4/sref_val.jsonl" --shard $i/2 > "$S/v4/codes_$i.log" 2>&1 &
done
wait
$PQ -m q3ko.data.extract_codes --jsonl "$S/v4/sref_train.jsonl" "$S/v4/sref_val.jsonl" --finalize_only > "$S/v4/codes_final.log" 2>&1
wc -l "$S"/v4/*.codes.jsonl
echo "[$(kst)] 끝"
