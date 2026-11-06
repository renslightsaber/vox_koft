#!/usr/bin/env bash
# SREF v3 새 합성분 만들기 → 거르기 → v3 합치기 → audio_codes (2026-11-06 · q3tts_koft 세션)
#   GPU 0,1 만 사용(GPU 2,3 은 vox_koft) · 한 GPU 에 합성 2개씩(4 shard)
#   문장: SREF/sentences/v3/*.jsonl (Claude 작성) + A7N · 542 새 문장 150개씩(v1 · topup 에서 쓴 문장 제외)
set -euo pipefail
H=~/workspaces; T=$H/vox_koft/tools/synth_dataset; S=$H/preprocessed/q3tts_koft/SREF; OUT=$S/v3/new
PQ=~/miniforge3/envs/q3tts/bin/python; PV=~/miniforge3/envs/voxcpm/bin/python; PS=~/miniforge3/envs/q3tts_ser/bin/python
mkdir -p "$OUT"; cd "$T"
kst(){ TZ=Asia/Seoul date '+%m-%d %H:%M'; }
echo "[$(kst)] 1) pick"
$PQ sref_synth.py pick --out "$OUT" --profile qwen3tts_sref --n_cs 0 --n_run117 0 --n_long 0 --n_a7n 150 --n_542 150 --id_tag V3 \
  --all_files v3/en_only.jsonl,v3/en_q_embed.jsonl,v3/ko_fragment.jsonl,v3/codeswitch_3.jsonl,v3/service_short.jsonl,v3/long_multi.jsonl \
  --exclude_used "$S/v1/sentences.jsonl,$S/v2/topup/sentences.jsonl"
echo "[$(kst)] 2) synth (GPU 0,1 × 2)"
for i in 0 1 2 3; do
  g=$(( i / 2 ))
  CUDA_VISIBLE_DEVICES=$g $PV sref_synth.py synth --out "$OUT" --shard $i/4 > "$OUT/synth_$i.log" 2>&1 &
done
wait
grep -h "합성" "$OUT"/synth_*.log
echo "[$(kst)] 3) score · quality(GPU 0) · asr(GPU 1)"
( CUDA_VISIBLE_DEVICES=0 $PQ sref_synth.py score --out "$OUT" > "$OUT/score.log" 2>&1 && \
  CUDA_VISIBLE_DEVICES=0 $PQ quality_check.py --out "$OUT" > "$OUT/quality.log" 2>&1 ) &
CUDA_VISIBLE_DEVICES=1 $PS sref_synth.py asr --out "$OUT" > "$OUT/asr.log" 2>&1 &
wait
echo "[$(kst)] 4) filter · v3 합치기"
$PQ sref_synth.py filter --out "$OUT" > "$OUT/filter.log" 2>&1; tail -1 "$OUT/filter.log" | cut -c1-400
$PQ make_sref_v3.py > "$S/v3/make.log" 2>&1; tail -30 "$S/v3/make.log"
echo "[$(kst)] 5) audio_codes(GPU 0,1)"
cd $H/q3tts_koft
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i $PQ -m q3ko.data.extract_codes --jsonl "$S/v3/sref_train.jsonl" "$S/v3/sref_val.jsonl" --shard $i/2 > "$S/v3/codes_$i.log" 2>&1 &
done
wait
$PQ -m q3ko.data.extract_codes --jsonl "$S/v3/sref_train.jsonl" "$S/v3/sref_val.jsonl" --finalize_only > "$S/v3/codes_final.log" 2>&1
wc -l "$S"/v3/*.codes.jsonl
echo "[$(kst)] 끝"
