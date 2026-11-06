#!/usr/bin/env bash
# scripts/run.sh — vox_koft 공통 런처 (hardware ↔ accelerate 짝 맞추기 + 로그 저장 + 실행 기록)
# ─────────────────────────────────────────────────────────────────────────────
# 사용:
#   ./scripts/run.sh train sft  --exp configs/experiments/sft/<ID>.yaml  --hw <하드웨어> [--accel 2gpu_a|2gpu_b] [--gpu N] [학습 인자...]
#   ./scripts/run.sh train lora --exp configs/experiments/lora/<ID>.yaml --hw …   (LoRA · 실험 yaml 에 lora.enable: true)
#
#   --hw rtxpro6000_4gpu                    → configs/accelerate/rtxpro6000_4gpu.yaml (GPU 0~3 · 포트 29600)
#   --hw rtxpro6000_2gpu --accel 2gpu_a|b   → configs/accelerate/rtxpro6000_2gpu_{a,b}.yaml (GPU 0,1 / 2,3 · 포트 29610/29620)
#   --hw rtxpro6000_1gpu --gpu N            → configs/accelerate/rtxpro6000_1gpu.yaml + CUDA_VISIBLE_DEVICES=N
#   나머지 인자(--smoke · --resume · --allow_schedule_change · --force_overwrite · --set k=v · --max_steps N)는
#   학습 진입점(python -m voxko.<sft|lora>.train)에 그대로 전달.
#
# 예:
#   ./scripts/run.sh train sft  --exp configs/experiments/sft/A7N_ref_FT_01.yaml --hw rtxpro6000_1gpu --gpu 0 --smoke
#   ./scripts/run.sh train lora --exp configs/experiments/lora/A7N_ref_LoRA_01_r32.yaml --hw rtxpro6000_2gpu --accel 2gpu_b
#   ./scripts/run.sh train sft  --exp configs/experiments/sft/A7N_ref_FT_01.yaml --hw rtxpro6000_4gpu --resume
#
# 이 셸은 "실행" 만 한다 — 설정 검증 · 스냅샷 · 재진입 가드는 Python(voxko) 이 맡는다.
#   · conda env voxcpm 확인 · hw ↔ accelerate 짝 · 로그를 {save_root}/{run}/logs/<run>_<ts>.log 로 tee
#     (smoke 는 {save_root}/_smoke_logs/) · VOX_COLOR=1 로 로그 파일도 컬러(less -R / tail -f)
#   · VOX_LAUNCH_CMD · VOX_ACCEL_FILE 을 넘겨 cmd.txt 에 "실제로 어떻게 실행했는지" 를 남긴다
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_NAME="${VOX_ENV:-voxcpm}"

die()  { echo -e "\033[1;31m❌ $*\033[0m" >&2; exit 2; }
info() { echo -e "\033[1;36m$*\033[0m"; }

[[ $# -ge 2 ]] || { sed -n '2,24p' "$0"; exit 2; }
CMD="$1"; TRACK="$2"; shift 2
case "$CMD/$TRACK" in
  train/sft)  ENTRY="voxko.sft.train" ;;
  train/lora) ENTRY="voxko.lora.train" ;;
  *) die "지원하지 않는 명령: $CMD $TRACK  (현재: train sft | train lora)" ;;
esac

EXP=""; HW=""; ACCEL=""; GPU=""; SMOKE=0; PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)   EXP="$2"; shift 2 ;;
    --hw)    HW="$2"; shift 2 ;;
    --accel) ACCEL="$2"; shift 2 ;;
    --gpu)   GPU="$2"; shift 2 ;;
    --smoke) SMOKE=1; PASS+=("$1"); shift ;;
    *)       PASS+=("$1"); shift ;;
  esac
done
[[ -n "$EXP" ]] || die "--exp 가 필요합니다"
[[ -f "$EXP" ]] || die "실험 파일이 없습니다: $EXP"
[[ -n "$HW" ]]  || die "--hw 가 필요합니다 (rtxpro6000_4gpu | rtxpro6000_2gpu | rtxpro6000_1gpu)"
case "$TRACK/$EXP" in
  sft/*experiments/lora/*)  die "lora 실험 파일은 'train lora' 로 실행하세요: $EXP" ;;
  lora/*experiments/sft/*)  die "sft 실험 파일은 'train sft' 로 실행하세요: $EXP" ;;
esac

# ── conda env ───────────────────────────────────────────────────────────────
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
  [[ -f "$CONDA_SH" ]] || CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
  # shellcheck disable=SC1090
  source "$CONDA_SH" && conda activate "$ENV_NAME" || die "conda env $ENV_NAME 를 활성화하지 못했습니다"
fi

# ── hw ↔ accelerate 짝 ──────────────────────────────────────────────────────
case "$HW" in
  *_4gpu) ACC_FILE="configs/accelerate/${HW}.yaml" ;;
  *_2gpu) [[ "$ACCEL" == "2gpu_a" || "$ACCEL" == "2gpu_b" ]] || die "2GPU 는 --accel 2gpu_a(GPU 0,1) 또는 2gpu_b(GPU 2,3) 를 지정하세요"
          ACC_FILE="configs/accelerate/${HW%_2gpu}_${ACCEL}.yaml" ;;
  *_1gpu) [[ -n "$GPU" ]] || die "1GPU 는 --gpu N 으로 사용할 GPU 를 지정하세요"
          ACC_FILE="configs/accelerate/${HW}.yaml"; export CUDA_VISIBLE_DEVICES="$GPU" ;;
  *) die "알 수 없는 --hw: $HW" ;;
esac
[[ -f "$ACC_FILE" ]] || die "accelerate 파일이 없습니다: $ACC_FILE"
case "$HW" in                                     # 학습 코드가 실제 GPU(UUID)와 대조 — 다르면 시작 전에 멈춤
  *_4gpu) export VOX_EXPECT_GPUS="0,1,2,3" ;;
  *_2gpu) [[ "$ACCEL" == "2gpu_a" ]] && export VOX_EXPECT_GPUS="0,1" || export VOX_EXPECT_GPUS="2,3" ;;
  *_1gpu) export VOX_EXPECT_GPUS="$GPU" ;;
esac
[[ -f "configs/hardware/${HW}.yaml" ]] || die "하드웨어 파일이 없습니다: configs/hardware/${HW}.yaml"

# ── 로그 위치 (설정의 save_root 를 Python 으로 읽음) ────────────────────────
RUN="$(basename "$EXP" .yaml)"
SETS=(); for ((k=0; k<${#PASS[@]}; k++)); do [[ "${PASS[$k]}" == "--set" ]] && SETS+=("${PASS[$((k+1))]}"); done
# --set paths.save_root=… 같은 덮어쓰기도 로그 위치에 반영(2026-11-06 고침: 전에는 무시돼 검증 로그가 실제 run 폴더로 감)
read -r SAVE_ROOT TB_ROOT < <(python - "$HW" "$EXP" "${SETS[@]}" <<'PY' 2>/dev/null
import sys
from voxko.common.config import load_config as L
c = L(sys.argv[1], sys.argv[2], sets=sys.argv[3:], check_files=False)["_derived"]; print(c["save_root"], c["tb_root"])
PY
)
[[ -n "${SAVE_ROOT:-}" ]] || die "설정을 읽지 못했습니다 — python -m $ENTRY --hw $HW --exp $EXP 로 오류를 확인하세요"
if [[ $SMOKE -eq 1 ]]; then LOG_DIR="$SAVE_ROOT/_smoke_logs"; else LOG_DIR="$SAVE_ROOT/$RUN/logs"; fi
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN}_$(date +%Y%m%d_%H%M%S).log"

export VOX_COLOR=1
# PyTorch 메모리 조각화 줄이기(2026-11-05 사용자 결정): 전체 FT 2GPU 최장 배치 실측 89.9 → 85.1GB · loss 같음. 따로 주면 그 값
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VOX_ACCEL_FILE="$ROOT/$ACC_FILE"
export VOX_LAUNCH_CMD="scripts/run.sh $CMD $TRACK --exp $EXP --hw $HW${ACCEL:+ --accel $ACCEL}${GPU:+ --gpu $GPU} ${PASS[*]:-}"

info "▶ $VOX_LAUNCH_CMD"
info "  accelerate : $ACC_FILE${CUDA_VISIBLE_DEVICES:+   (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)}"
info "  log        : $LOG"
# TensorBoard 그룹(2026-11-06 규칙 · q3tts_koft 와 같음 · docs/guides/tensorboard_groups.md): 학습 방식/데이터/run 트리. 보기 = ./scripts/tb.sh
TB_GROUP="$(python tools/tb_group.py --where "$RUN" 2>/dev/null || echo '?')"
info "  tensorboard: ./scripts/tb.sh  →  그룹 runs_grouped/$TB_GROUP   (맥북: ssh -N -L ${TB_PORT:-6008}:localhost:${TB_PORT:-6008} 112)"
[[ "$TB_GROUP" == _misc* ]] && echo -e "\033[1;33m⚠️  이 run 이름은 TensorBoard 그룹 규칙에 없어 _misc 로 갑니다 — tools/tb_group.py 의 method_of() 에 학습 방식을 추가하세요\033[0m"
if [[ $SMOKE -eq 0 ]]; then   # 기록 폴더가 생긴 뒤(약 2분 30초) 그룹 트리에 바로 연결 — tb.sh 를 띄워 두지 않았어도 다음에 바로 보이게
  ( sleep 150; python tools/tb_group.py ) >/dev/null 2>&1 < /dev/null &   # 출력 통로를 떼어 냄(안 그러면 run.sh 출력을 파이프로 받는 쪽이 150초 동안 안 끝남)
fi
echo

accelerate launch --config_file "$ACC_FILE" -m "$ENTRY" --exp "$EXP" --hw "$HW" "${PASS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo
if [[ $RC -eq 0 ]]; then info "✅ 완료 (rc=0) — log: $LOG"; else echo -e "\033[1;31m❌ 실패 (rc=$RC) — log: $LOG\033[0m"; fi
exit "$RC"
