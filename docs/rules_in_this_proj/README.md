# 이 프로젝트의 규칙 모음 (rules_in_this_proj)

> 📅 2026-11-05 · 처음 보는 사람용 · 자매 프로젝트 `~/workspaces/q3tts_koft/docs/rules_in_this_proj/` 를 vox_koft 에 맞게 옮김
> **"이 프로젝트에서는 이렇게 한다"** 를 한곳에 모았습니다. 규칙이 바뀌면 **여기를 먼저 고치고**, 다른 문서에는 링크만 둡니다.

---

## 0. 한눈에

| 문서 | 무엇을 정하나 | 예 |
|---|---|---|
| [run_naming.md](run_naming.md) | **run 이름** 짓는 법 | `A7N_ref_FT_01` · `A7N_ref_LoRA_01_r32` |
| [run_groups.md](run_groups.md) | run **계열**과 폴더 · GPU · 포트 대응 | 설정 · 기록 · 노트북 · ckpt 위치 |
| [training_workflow.md](training_workflow.md) | 학습 **한 번의 순서** | smoke → 백업 → 사용자 실행 → 종료 확인 → 평가 → resume 정리 |
| [notebooks.md](notebooks.md) | **청음 노트북** 만들기 | 커널 "Python (voxcpm)" · 사용자 수정본 보호 |
| [eval_listening.md](eval_listening.md) | **평가 · 청음 · step 고르기** | 청음이 정본 · 비교 페이지 |
| [docs_handoff.md](docs_handoff.md) | **문서 · 인수인계 · git** | 문서 위치 · 이름 · 커밋은 요청 시 |
| [request_templates.md](request_templates.md) | **자주 하는 요청의 약속** | "학습 끝났어" · "docs · 인수인계" · "학습 준비" |

## 1. 가장 중요한 다섯 가지
1. 🗣️ 사용자와는 **한국어**로, 진행 안내까지 모두. 문서는 **처음 보는 사람용**으로. 시각은 **KST**(서버 UTC + 9).
2. 🙋 제안 · 질문은 **먼저 하고 사용자가 고른다.**
3. 🏃 **본 학습은 사용자가 byobu 에서** 실행한다. Claude 는 smoke · 짧은 검증 · 백업까지 하고 **실행 명령을 알려 준다.** 작성한 코드는 **실제로 돌려서 검증**한다.
4. 🗑️ 삭제 · 다운로드 · 긴 작업 · 대량 생성은 **승인 뒤**. 추론 · 청음에 쓰는 `export_ft/` · `export_lora/` 는 지우지 않는다.
5. 🎧 지표는 후보를 좁히는 도구일 뿐, **확정은 귀(청음)**.

## 2. 이 폴더 밖에 있는 규칙
- 매 세션 자동으로 읽는 요약: [`../../CLAUDE.md`](../../CLAUDE.md) §1
- 코드 수준 규칙(설정 스키마 · 재진입 거부 · resume 정확 복원 · export 검증): [`../guides/sft_training.md`](../guides/sft_training.md) · [`../guides/lora_training.md`](../guides/lora_training.md)
- 공식 학습 코드와 다른 점: [`../reference/voxcpm2_training_code_walkthrough.md`](../reference/voxcpm2_training_code_walkthrough.md)
