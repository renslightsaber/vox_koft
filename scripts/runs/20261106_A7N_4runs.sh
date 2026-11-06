#!/usr/bin/env bash
# 4개 본 학습을 GPU 2,3(2gpu_b · 포트 29620)에서 차례로 — 2026-11-06 사용자 결정
#   1) A7N_ref_FT_01  2) A7N_ref_LoRA_01_r32  3) A7N_ref_FT_02_iv2  4) A7N_ref_LoRA_02_r32_iv2
# 동작
#   · 앞 run 이 실패해도 다음 run 으로 넘어간다(';' 와 같음) — 밤새 하나가 죽어도 나머지는 돈다. 실패한 run 은 나중에 --resume.
#   · Ctrl-C(또는 kill)를 누르면 **지금 run 을 멈추고 뒤 run 도 시작하지 않는다**('&&' 처럼 전체 중단).
#   · 시작 전에 GPU 2,3 에 다른 프로세스가 있으면 멈춘다(--force 로 무시). 각 run 시작 직전에도 다시 확인.
#   · 끝나면 run 별 결과(rc) · 걸린 시간 · 로그 위치를 표로 보여 준다.
# 사용(byobu):  cd ~/workspaces/vox_koft && ./scripts/runs/20261106_A7N_4runs.sh [--force] [--from N]
#   --from N : N 번째 run 부터(예: 1·2 가 끝난 뒤 다시 시작할 때). 이미 끝난 run 을 다시 돌리면 run.sh 가 거부한다.
set -u
cd "$(dirname "$0")/../.."
RUNS=("sft configs/experiments/sft/A7N_ref_FT_01.yaml"
      "lora configs/experiments/lora/A7N_ref_LoRA_01_r32.yaml"
      "sft configs/experiments/sft/A7N_ref_FT_02_iv2.yaml"
      "lora configs/experiments/lora/A7N_ref_LoRA_02_r32_iv2.yaml")
FORCE=0; FROM=1
while [[ $# -gt 0 ]]; do case "$1" in --force) FORCE=1; shift ;; --from) FROM="$2"; shift 2 ;; *) echo "모르는 옵션: $1"; exit 2 ;; esac; done
RUNNER="${VOX_CHAIN_RUNNER:-./scripts/run.sh}"          # 시험용으로 바꿀 수 있음(기본 = 실제 학습)
STOP=0
trap 'STOP=1; echo -e "\n\033[1;31m⛔ 중단 신호 — 지금 run 이 끝나면 뒤 run 은 시작하지 않습니다\033[0m"' INT TERM

gpu_busy() {   # GPU 2,3 에서 도는 다른 프로세스 수
  local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==2||$1==3{print $2}')
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -cF "$u" || true
}
declare -a RES
for i in "${!RUNS[@]}"; do
  n=$((i + 1)); [[ $n -lt $FROM ]] && continue
  [[ $STOP == 1 ]] && { RES+=("$n | 건너뜀(중단) | - | -"); continue; }
  read -r TR EXP <<< "${RUNS[$i]}"; RUN=$(basename "$EXP" .yaml)
  B=$(gpu_busy); W=0
  while [[ "$B" != "0" && $FORCE == 0 && $W -lt 60 ]]; do sleep 5; W=$((W + 5)); B=$(gpu_busy); done   # 앞 run 정리가 늦을 수 있어 60초까지 기다림
  if [[ "$B" != "0" && $FORCE == 0 ]]; then
    echo -e "\033[1;31m❌ GPU 2,3 에 다른 프로세스 ${B}개 — 멈춥니다(nvidia-smi 확인 · 무시하려면 --force)\033[0m"; RES+=("$n | $RUN | 시작 안 함(GPU 사용 중) | -"); STOP=1; continue
  fi
  T0=$(date +%s); echo -e "\033[1;36m━━ [$n/4] $RUN 시작 $(TZ=Asia/Seoul date '+%m/%d %H:%M KST') ━━\033[0m"
  # 포그라운드로 실행(백그라운드 '&' 는 비대화형 셸에서 SIGINT 를 무시하게 돼 Ctrl-C 가 학습을 못 멈춤)
  # Ctrl-C 는 학습 프로세스에 직접 가고, 이 스크립트의 trap 은 그 run 이 끝난 뒤 STOP=1 을 세운다
  $RUNNER train "$TR" --exp "$EXP" --hw rtxpro6000_2gpu --accel 2gpu_b; RC=$?
  DT=$(( $(date +%s) - T0 ))
  RES+=("$n | $RUN | rc=$RC | $((DT / 3600))h$(((DT % 3600) / 60))m")
done
echo -e "\n\033[1;36m━━ 요약 ($(TZ=Asia/Seoul date '+%m/%d %H:%M KST')) ━━\033[0m"
printf '%s\n' "${RES[@]}"
echo "로그: ~/workspaces/outputs/vox_koft/ckpt/<run>/logs/ · 실패한 run 은 같은 명령에 --resume"
