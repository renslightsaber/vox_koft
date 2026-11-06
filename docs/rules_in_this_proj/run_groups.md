# run 계열 · 폴더 · GPU 규칙

> 📅 2026-11-05 · 처음 보는 사람용 · q3tts_koft 규칙을 옮김
> 모음 첫 장: [README](README.md) · 이름: [run_naming.md](run_naming.md)

---

## 0. 한눈에
- run 하나는 **설정 · 기록 · ckpt · TensorBoard · 청음 노트북** 다섯 곳에 흔적을 남깁니다. 이름(run_name)이 모두 같습니다.
- 계열은 둘 — **sft(전체 FT)** · **lora**.

## 1. 다섯 곳

| 무엇 | 위치 | 비고 |
|---|---|---|
| 설정 | `configs/experiments/<sft\|lora>/<run>.yaml` | 병합: `config.yaml → hardware/<gpu>.yaml → experiments → --set` |
| 기록(저장소 안) | `exp_reports/<sft\|lora>/<run>/` | `config_backup/`(학습 전) · `run_backup/`(학습 뒤 — 도구는 첫 본 학습 뒤 만듦) |
| ckpt(저장소 밖) | `~/workspaces/outputs/vox_koft/ckpt/<run>/step_N/` | `export_ft/` 또는 `export_lora/`(추론 · 청음) · `_COMPLETE` · 재개용 파일(지우면 `_RESUME_REMOVED`) · `configs/`(스냅샷) · `logs/` |
| TensorBoard | `~/workspaces/outputs/vox_koft/runs/<run>/` · 그룹 보기 `runs_grouped/<방식>/<데이터>/<run>`(폴더 symlink) | 포트 **6008** · `./scripts/tb.sh` — [안내](../guides/tensorboard_groups.md) |
| 청음 노트북 | `notebooks/<sft\|lora>/<run>_listening.ipynb` | [notebooks.md](notebooks.md) |

## 2. GPU · 포트
- GPU 배정은 **그때그때 사용자가 정합니다.** 실행 명령에 `--gpu N`(1장) · `--accel 2gpu_a|2gpu_b`(2장) 로 적고, 시작 뒤 로그의 `GPU 확인: rank0 → nvidia-smi N번` 으로 맞는지 봅니다.
- 🔴 서버 GPU 4장을 **q3tts_koft 와 함께** 씁니다 → 시작 전 `nvidia-smi` 로 빈 GPU · 남은 메모리를 확인합니다.
  - 실측(2026-11-06 · 2GPU · 가장 긴 배치 · expandable_segments 켬): 전체 FT bs4 실제 점유 최대 **83GB** · LoRA bs8 **56GB**(97.9GB 중) — [사전 점검](../reports/20261106_four_runs_preflight.md) §4. 함께 도는 작업이 있으면 FT 는 들어가지 않을 수 있음.
  - 2026-11-06 첫 본 학습 4개: **GPU 2,3 만**(`--accel 2gpu_b`) · GPU 0,1 은 q3tts_koft 가 씀. 연속 실행 스크립트 `scripts/runs/<날짜>_<내용>.sh`(실패해도 다음 · Ctrl-C 면 전체 중단 · 시작 전 GPU 확인).
- 학습과 평가를 같이 돌릴 때는 **학습 GPU 와 평가 GPU 를 나눕니다**.
- 포트
  - TensorBoard **6008**(6006 · 6007 은 다른 프로젝트)
  - accelerate: 4GPU **29600** · 2GPU a/b **29610 / 29620**(q3tts_koft 의 29500 · 29510 · 29520 과 겹치지 않음)
  - 청음 페이지 HTTP **8765**(`~/workspaces/tmp_listen` — q3tts_koft 와 같은 서버, 폴더 `vox_koft/`)
