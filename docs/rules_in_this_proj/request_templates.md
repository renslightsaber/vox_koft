# 자주 하는 요청 — 짧게 말해도 되는 "약속" 모음

> 📅 2026-11-05 · 처음 보는 사람용 · q3tts_koft 의 같은 문서를 옮김 · 모음 첫 장: [README](README.md)
> 사용자가 **"학습 끝났어"**, **"docs · 인수인계 업데이트해줘"**, **"학습 준비"** 처럼 짧게 말해도 Claude 가 아래 순서를 **빠짐없이** 하도록 약속해 둡니다.
> 사용자가 따로 지시하면 그 지시가 이 문서보다 우선합니다.

---

## 1. "`<run>` 학습 끝 — §1 대로"

| # | 할 일 | 도구 · 위치 |
|---|---|---|
| 1 | 정상 종료 확인 — 로그 `Training End` · `step_*/_COMPLETE` 개수 · `_EXPORT_FAILED` 없음 · 오류 · 경고 · 재개용 파일이 마지막 1개만 남았는지 | `~/workspaces/outputs/vox_koft/ckpt/<run>/logs/*.log` |
| 1 | 학습 뒤 백업 — `ckpt/<run>/configs/merged.yaml` == 학습 전 백업 확인 · 로그 · TB 수치 → `exp_reports/<sft\|lora>/<run>/run_backup/` | (도구는 첫 본 학습 뒤 만듦) |
| 2 | 분석: loss_diff · loss_stop · val · val_unseen · grad · 속도 · epoch · **짝 run 과 같은 step 비교** → 보고서 + 채팅 요약(좁은 표) | `docs/reports/<날짜>_<주제>.md` |
| 3 | 청음 — 원본 vs 후보 step(처음 보는 화자 무감정 ref × 7칸) 페이지 또는 노트북 | [notebooks.md](notebooks.md) · [eval_listening.md](eval_listening.md) |
| 4 | **D안 재측정**(성공 판정) — 고른 step 의 export 를 `d_eval.py vox → sv → ser → page` 로 추가해 원본과 비교 | `voxko/eval/d_eval.py` · [plan](../plan/20261105_success_criteria_and_next_runs.md) |

## 2. "docs · 인수인계 업데이트 — §2"

| 경우 | 할 일 |
|---|---|
| 바뀐 게 적을 때 | 기존 문서에 **보완 · 수정만** |
| 새 주제가 생겼을 때 | 폴더 = 문서 성격 · 이름 = `<YYYYMMDD>_<주제>.md`(데이터 · 조사 · 규칙은 날짜 없이) |
| 항상 같이 | 색인 `docs/README.md` · `CLAUDE.md` 지형 · 링크 검사 |
| 인수인계 | `_handoff/YYYYMMDD_HHMM_handoff_main.md`(KST) — 마지막 인수인계에서 이어서: 지금 상태 표 · 한 일 · 다음 할 일 · 남은 사용자 결정 · 기술 메모 |

## 3. "`<run>` 학습 준비 — §3"

| # | 할 일 |
|---|---|
| 1 | 짝 run 설정을 복사해 **바뀌는 키만** 고친 초안 → **diff 를 보여 주고** 모르는 값은 먼저 묻기 |
| 2 | 결정되면 학습 전 백업(`exp_reports/<sft\|lora>/<run>/config_backup/` · merged · md5) |
| 3 | 필요한 코드 수정(VoxCPM/ 은 건드리지 않음) → `pytest tests/` · **smoke(빈 GPU)** |
| 4 | **byobu 명령(코드 블록)** 전달 → 시작 뒤 프로세스 묶음 · 첫 step 확인 |

- run 이름 규칙([run_naming.md](run_naming.md))으로 방법을 짐작하되, **새 방법 · 새 데이터 · 새 지시문 표**면 먼저 묻습니다.
- GPU: 빈 GPU 먼저. 없으면 학습에 영향이 없는 한 다른 작업과 함께 써도 됩니다(시간은 늘 수 있음).
