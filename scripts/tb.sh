#!/usr/bin/env bash
# TensorBoard 를 '학습 방식/데이터/run' 그룹으로 띄운다 — 안내: docs/guides/tensorboard_groups.md (q3tts_koft scripts/tb.sh 를 옮김)
#   ./scripts/tb.sh                # 그룹 트리(runs_grouped) · 포트 6008 · localhost 만 · 5분마다 트리 갱신
#   ./scripts/tb.sh --port 6011    # 다른 포트
#   ./scripts/tb.sh --flat         # 원래 runs 폴더 그대로(그룹 없음, 갱신 루프 없음)
#   ./scripts/tb.sh --every 300    # 트리 갱신 주기(초, 기본 300)
# Ctrl-C(또는 종료)하면 트리 갱신 루프도 같이 끝난다. 원래 runs/ 는 건드리지 않는다.
# 포트: 6006 · 6007 은 다른 프로젝트(cv3_koft · q3tts_koft) — vox_koft 는 6008.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_BIN="${VOX_ENV_BIN:-$HOME/miniforge3/envs/voxcpm/bin}"
OUT="$HOME/workspaces/outputs/vox_koft"
PORT=6008; FLAT=0; EVERY=300
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --flat) FLAT=1; shift ;;
    --every) EVERY="$2"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "모르는 옵션: $1" >&2; exit 1 ;;
  esac
done

if [[ $FLAT == 1 ]]; then
  LOGDIR="$OUT/runs"; mkdir -p "$LOGDIR"
else
  LOGDIR="$OUT/runs_grouped"
  "$ENV_BIN/python" "$ROOT/tools/tb_group.py"
  ( while sleep "$EVERY"; do "$ENV_BIN/python" "$ROOT/tools/tb_group.py" >/dev/null || true; done ) &
  LOOP_PID=$!
  trap 'kill $LOOP_PID 2>/dev/null || true' EXIT INT TERM
  echo "🔁 트리 갱신 루프 pid $LOOP_PID (${EVERY}초마다)"
fi

echo "📈 TensorBoard: $LOGDIR → http://localhost:$PORT  (맥북: ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L $PORT:localhost:$PORT 112)"
"$ENV_BIN/tensorboard" --logdir "$LOGDIR" --port "$PORT" \
  --reload_interval 30 --reload_multifile true
