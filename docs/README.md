# vox_koft 문서 색인

> 처음 오셨다면 이 순서로 읽으세요.
> 1. [규칙 모음](rules_in_this_proj/README.md)
> 2. [VoxCPM2 조사](reference/voxcpm2_overview_finetune.md)
> 3. [공식 학습 코드 해설 · 우리가 바꾼 점](reference/voxcpm2_training_code_walkthrough.md)
> 4. [전체 FT 안내](guides/sft_training.md) · [LoRA 안내](guides/lora_training.md)
> 5. 최신 [`_handoff/`](../_handoff/)

| 폴더 | 무엇 | 문서 |
|---|---|---|
| `rules_in_this_proj/` | **이 프로젝트의 규칙** | [README](rules_in_this_proj/README.md) · [run 이름](rules_in_this_proj/run_naming.md) · [계열 · GPU · 포트](rules_in_this_proj/run_groups.md) · [학습 순서](rules_in_this_proj/training_workflow.md) · [노트북](rules_in_this_proj/notebooks.md) · [평가 · 청음](rules_in_this_proj/eval_listening.md) · [문서 · 인수인계](rules_in_this_proj/docs_handoff.md) · [짧은 요청 약속](rules_in_this_proj/request_templates.md) |
| `reference/` | 외부 자료 · 조사 · 공식 코드 해설 | [voxcpm2_overview_finetune.md](reference/voxcpm2_overview_finetune.md) · [voxcpm2_training_code_walkthrough.md](reference/voxcpm2_training_code_walkthrough.md) · [voxcpm2_synthetic_dataset.md](reference/voxcpm2_synthetic_dataset.md)(SREF 합성 · 감정 증강 시범) |
| `guides/` | 실행 안내 | [sft_training.md](guides/sft_training.md) · [lora_training.md](guides/lora_training.md) · [tensorboard_groups.md](guides/tensorboard_groups.md)(TB 그룹 보기 · 맥북) · [agents_creation_guide.md](guides/agents_creation_guide.md) |
| `datasets/` | 데이터 정본 | [A7N_vox_koft.md](datasets/A7N_vox_koft.md) · 🆕 [A7NX_vox_koft.md](datasets/A7NX_vox_koft.md) — 같은 사람 다른 칸 ref |
| `reports/` | 실험 · 검증 결과 | [20261105_training_infra_smoke.md](reports/20261105_training_infra_smoke.md) — 전체 FT · LoRA smoke · resume · 2GPU · 합성 · [20261106_four_runs_preflight.md](reports/20261106_four_runs_preflight.md) — 본 학습 4개 사전 점검 · 🆕 [20261106_A7N_4runs_results.md](reports/20261106_A7N_4runs_results.md) — **본 학습 4개 결과 · D 재측정 · 진짜 녹음 감정 기준선** · 🆕 [20261106_neutral_cell_virtual_speaker.md](reports/20261106_neutral_cell_virtual_speaker.md) — **중립 칸 · 가상 화자 일관성 · REF 음향 · 중립 시범 생성 · best 5 / 상위 5 표** · 🆕 [20261106_cell_voice_drift_analysis.md](reports/20261106_cell_voice_drift_analysis.md) — **감정 칸별 목소리 이동 · run 별 순위(모두 s1500) · FT01 ≈ FT02(Base 아님)** · 🆕 [20261106_bestofn_voice_filter.md](reports/20261106_bestofn_voice_filter.md) — **센 감정 칸 목소리 흔들림: 여러 번 만들어 고르기 · 문턱 · cfg/prompt 노브** · 🆕 [20261107_A7NX_A7NX3_results.md](reports/20261107_A7NX_A7NX3_results.md) — **다른 칸 ref 1·2차 · val_xref · 감정 ref 판 · 칸별 이동** · 🆕 [20261107_A7NX2R_results.md](reports/20261107_A7NX2R_results.md) — **A7NX2R lr 두 판: 목소리 최고 · 감정 약해짐** · 🆕 [20261107_cfg_test_emo_voice.md](reports/20261107_cfg_test_emo_voice.md) — **CFG 시험(감정 효과 없음) · 감정 ↔ 목소리 그림 · 감정 ref 판 청음 페이지** · ⏳ `20261107_A7NX2Rs_results.md` — A7NX2Rs · A7NX2Rs6 결과(D 재측정 중 · 예정) |
| `plan/` | 계획 · 판정 기준 | [20261105_success_criteria_and_next_runs.md](plan/20261105_success_criteria_and_next_runs.md) · 🆕 [20261106_xref_lora_runs.md](plan/20261106_xref_lora_runs.md) — 다른 칸 ref LoRA 2개 — **성공 판정(D안 재측정 · 원본 기준선)** · 학습 형식 확인 · ref 0.3 = ref 없음 · 둘째 run 후보(ref 1.0 · +542) |
| `history/` | 날짜별 기록 | [20261105_project_setup.md](history/20261105_project_setup.md) |
| `decisions/` · `explanations/` | 결정 · 개념 | [explanations/20261107_why_mix_542.md](explanations/20261107_why_mix_542.md) — 왜 542 무감정을 섞나(q3tts 세션) |

관련(자매 프로젝트 q3tts_koft — 읽기만): `~/workspaces/q3tts_koft/docs/reference/voxcpm2_finetune_license.md`(라이선스 · 서빙 · 속도) · `docs/plan/20261104_codeswitch_fix_plan.md`(코드스위치 해결 계획) · `docs/datasets/README.md`(A7N 원천).
