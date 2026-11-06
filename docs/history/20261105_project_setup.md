# 2026-11-05 — vox_koft 프로젝트 틀 · 학습 인프라 · 성공 판정 기준

> 📅 2026-11-05 · 처음 보는 사람용 · 날짜별 요약 기록(자세한 것은 링크)

---

## 0. 한눈에
| 시각(KST) | 한 일 |
|---|---|
| 10:20 | env `voxcpm` · VoxCPM2 받기 · 조사 문서 ([handoff](../../_handoff/20261105_1020_handoff_main.md)) |
| 14:05 | 프로젝트 틀 지시문 (q3tts_koft 세션에서 작성 · [원문](../../_handoff/20261105_1405_prompt_ft_lora_setup.md)) |
| 14:30~15:50 | 서브 에이전트 10종 · 규칙 8개 · `voxko`(전체 FT · LoRA) · 첫 검증 통과 — [보고서](../reports/20261105_training_infra_smoke.md) |
| 16:00 | A7N 전체 16kHz 변환 · 가장 긴 배치 메모리 측정 → FT 4×4 · LoRA 8×2 · 본 학습 yaml 확정 · 학습 전 백업 |
| 16:30 | q3tts 점검 반영: **성공 판정 = D안 재측정** · 원본 기준선 · 학습 형식 확인 · 둘째 run 후보 — [plan](../plan/20261105_success_criteria_and_next_runs.md) |
| 16:40 | 542 전체 16kHz 변환(둘째 run 후보 B 용) |
| 16:50 | 본 학습은 q3tts 학습이 GPU 를 쓰는 동안 대기 |
| 00:18 | 🏃 본 학습 4개 연속 실행 시작(사용자 · byobu · GPU 2,3) — 1번 step 180 에서 1.2초/step · 정상 |
| 03:32 | ✅ 4개 모두 정상 종료(64 · 33 · 62 · 32분) · 학습 뒤 백업 · run.sh 새 판 교체 |
| 03:43~04:15 | D 재측정 18개 모델 + 진짜 녹음 감정 기준선 · 노트북 4개(smoke OK) · [결과 보고서](../reports/20261106_A7N_4runs_results.md) — 목소리 · 발음 · 깨짐은 원본보다 좋아짐, 감정은 원본의 과장이 줄고 A7N 진짜 녹음 세기에 맞춰짐(청음 판정 대기) |
| 23:36~00:20 | 본 학습 **4개**로 확정(웹 조사판 3 · 4번 추가) · v2 지시문 표 · drop_only · expandable_segments · TB 그룹 · 연속 실행 스크립트 · run.sh 로그 위치 버그 고침 — [사전 점검](../reports/20261106_four_runs_preflight.md) |

## 1. 사용자 결정
- 우리 트레이너(HF accelerate) + 공식 부품
- 16kHz 미리 변환
- Neutrality 에도 지시문 · 지시문 표 v1(새로 작성)
- ref = 같은 화자 무감정 0.7 · 없는 26명은 ref 없이
- 542 는 첫 run 에서 빼고 기능만 준비
- ~~첫 본 학습은 FT · LoRA 를 동시에 2GPU 씩~~ → (11/06) 4개를 GPU 2,3 에서 차례로 · 학습은 사용자가 byobu 에서 실행
- (11/06) v2 지시문 표 확정 · 지시문 0.9 = 전체의 10% 를 Neutrality 에서만 · run 이름 `_iv2` · 끝 무음 그대로 · expandable_segments 기본으로 켬
