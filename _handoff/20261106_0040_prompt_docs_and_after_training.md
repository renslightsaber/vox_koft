# vox_koft — 지금 docs · 인수인계(§2) + 학습 끝나면 §1(청음 노트북 · 백업 · 종합 평가) → 다시 §2

> 📅 2026-11-06 00:40 KST · q3tts_koft 세션에서 작성한 지시문(사용자가 자기 전에 이 폴더의 Claude Code 에 보냄 · 사용자는 아침까지 답하지 못함)

사용자는 자는 중입니다. 아래를 순서대로 진행하고, **사용자 결정이 꼭 필요한 것만 모아 두었다가** 아침에 한 번에 물어봐 주세요(그 전까지는 결정이 필요 없는 일만). 학습 중인 byobu 창 · GPU 2,3 의 학습 프로세스는 절대 건드리지 마세요(중단 · 재시작 · 설정 변경 금지).

## 1. 지금 — docs · 인수인계 업데이트(request_templates §2)
- 좀 전에도 §2 를 했으니, **그 뒤로 바뀐 것이 있는지 먼저 확인**하고 있는 것만 반영해 주세요(없으면 "변경 없음"으로 짧게).
  - 확인할 것: 학습 4개 && 진행 상태(00:38 KST 기준 `A7N_ref_FT_01` step 940/3000 · val loss 0.792 → 0.787 · 1.22 s/step · GPU 2,3 약 76~84GB) · TensorBoard 그룹(`runs_grouped` · `scripts/tb.sh` · 포트 6008) 반영 여부 · q3tts_koft 쪽 TensorBoard 규칙(아래)
  - q3tts_koft 의 새 TensorBoard 규칙(11/06): `scripts/run.sh` 가 학습 시작 때 그룹 경로를 안내하고 약 2분 뒤 그룹 트리에 자동 연결 · 새 학습 방식 이름은 `tools/tb_group.py` 의 `method_of()` 에 추가(`--where <run>` 으로 미리 확인). vox_koft 에도 같은 규칙을 맞춰 주세요(학습 중인 프로세스에는 영향 없게 — run.sh 수정은 다음 실행부터 적용).
- 인수인계 `_handoff/YYYYMMDD_HHMM_handoff_main.md`(KST).

## 2. 학습 4개가 모두 끝나면 — request_templates §1
- 순서: `A7N_ref_FT_01` → `A7N_ref_LoRA_01_r32` → `A7N_ref_FT_02_iv2` → `A7N_ref_LoRA_02_r32_iv2`(GPU 2,3 · `&&`). 예상 03:30~04:00 KST(추정). 주기적으로(예: 20~30분) 확인하되, **4개가 모두 끝난 뒤** 아래를 시작해 주세요. 중간 run 이 실패해서 뒤 run 이 안 돌았으면, 다시 띄우지 말고 원인만 조사해 아침 보고에 넣어 주세요.
- GPU: 학습이 끝난 뒤에는 **GPU 2,3** 을 씁니다. GPU 0,1 은 q3tts_koft SREF 평가가 쓸 수 있으니 `nvidia-smi` 로 확인하고 비어 있을 때만 추가로 쓰세요.
1. **종료 확인** — 로그 Training End · `_COMPLETE` · 오류 · 경고 · 재개용 파일 개수
2. **학습 설정값 · 로그 백업**(학습 전 백업과 같은지 확인) · `curves.png` · TB 수치 CSV
3. **종합 평가**(run 마다 저장 step 전부 또는 대표 step — 시간이 많이 걸리면 step 수를 줄이고 이유를 기록):
   - **D 재측정(`voxko/eval/d_eval.py`)**: 원본 기준선(목표 1위 67% · 감정 47%) 옆에 열로 — 이 FT 의 성공 판정 기준
   - 감정: SER(emotion2vec · arousal) — 칸별
   - 발음: ASR(whisper) → CER
   - 목소리: 목표 화자 유사도(WavLM-large) · 공정한 잣대
   - **깨짐 · 이상**: SQUIM(torchaudio, 정답 없이 STOI · PESQ · SI-SDR 추정) · 문장 중간 긴 무음(≥ 0.8초) · 클리핑 · 끝 잘림 · 빈 합성 · 안 끝나는 합성(최대 길이 도달) · 말 속도
     - 참고 구현: `~/workspaces/vox_koft/tools/synth_dataset/quality_check.py` · q3tts_koft `tools/diag/sref_eval.py`(같은 시간에 만든 SREF 평가 도구)
   - 4 run 비교: FT vs LoRA · v1(지시문 1문구 · ref 0.7) vs iv2(지시문 4문구 · 지시문 0.9 · ref 0.5) · step 별 best 5
4. **청음 노트북** run 마다 1개(best 5 주석 · smoke 실행 확인) + D 재측정 청음 페이지(`http://127.0.0.1:8765/vox_koft/d_eval/index.html`)에 FT · LoRA 열 추가
5. 분석 보고서 `docs/reports/<날짜>_<주제>.md`(처음 보는 사람용 · 표는 좁게) — 핵심: **D안 데이터 생성에 쓸 수 있는가**(감정이 원본보다 확실히 세지고 목소리는 원본만큼 유지되는가), 어느 run · step 이 가장 나은가
6. 다시 **docs · 인수인계 업데이트(§2)** — 아침에 사용자가 볼 요약(한눈에 표 · 결정할 것 목록)을 인수인계 맨 위에

## 3. 지켜 주세요
- 한국어 · 처음 보는 사람용 문서 · 시각 KST · git commit/push 금지(사용자가 요청할 때만) · 삭제 금지 · `VoxCPM/` 수정 금지
- 청음은 사용자가 아침에 합니다 — 청음 판정을 대신하지 말고, 숫자는 "참고용"으로 표시
