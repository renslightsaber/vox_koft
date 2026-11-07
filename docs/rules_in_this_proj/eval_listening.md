# 평가 · 청음 · step 고르기 규칙

> 📅 2026-11-05 · 처음 보는 사람용 · q3tts_koft 규칙을 옮김(평가 도구는 첫 본 학습 뒤 사용자와 정함)
> 모음 첫 장: [README](README.md)

---

## 0. 한눈에
- 🎧 **청음이 정본**입니다. 지표(TB · 자동 평가)는 **들을 후보를 좁히는 도구**입니다.
- 이 프로젝트의 핵심 질문 두 가지
  1. **지시문으로 감정이 확실히 나오나** — 원본은 지시문만으로는 감정이 약했음
  2. **목소리가 ref 를 따르나** — 무감정 ref + 지시문으로 합성했을 때 감정 ref 처럼 목소리가 섞이지 않는지(D안)

## 0-1. 🔴 성공 판정 = D안 재측정 (2026-11-05)
- `voxko/eval/d_eval.py`: 처음 보는 화자 16명 무감정 reference + `"(지시문)문장"` · 6칸 × 2문장 · **원본 VoxCPM2 와 같은 입력으로 비교**
  - 잣대: WavLM-large 목표 1위 · emotion2vec · arousal · CER
  - 청음 페이지 `tmp_listen/vox_koft/d_eval/`
- 기준과 원본 기준선: [plan/20261105_success_criteria_and_next_runs.md](../plan/20261105_success_criteria_and_next_runs.md)

## 1. 평가 (후보 도구 — 쓰기 전에 사용자와 정함)
| 무엇 | 도구(q3tts_koft 와 같은 것 우선) | 상태 |
|---|---|---|
| 발음 | whisper large-v3-turbo → ko_tts_norm → CER | `voxko/eval/metrics.py` 의 `cer` 만 |
| 목소리 | WavLM-large SV(q3tts `q3ko.eval.sv`, env q3tts) · 귀 | ✅ `d_eval.py sv` |
| 감정 | SER(emotion2vec · arousal/valence) — **별도 env**(`q3tts_ser` 를 쓰기만) | ✅ `d_eval.py ser` |
| 끝 잘림 | 끝 무음 < 50ms & 마지막 50ms > −45dB = 잘림 의심 | ✅ `voxko/eval/metrics.py tail_metrics` |
| 목소리 공정한 잣대 | WavLM-large + ERes2Net 두 잣대의 목표 1위 평균 | ✅ `d_eval.py sv`(2026-11-06) |
| 깨짐 · 이상 | SQUIM(STOI · PESQ · SI-SDR, 원본 하위 5% 보다 낮으면) · 중간 무음 ≥ 0.8초 · 클리핑 · 끝 잘림 · 빈 합성 · 최대 길이 도달 · 말 속도 | ✅ `d_eval.py quality`(2026-11-06) |
| 일괄 실행 | run 마다 대표 step(750 · 1500 · 2250 · 3000 + TB 처음 보는 화자 val 최저) → GPU 2장에 나눠 | ✅ `tools/eval_runs.py` |
| 중립 칸 | 7칸째 · 학습 텍스트에 없는 A7N 계열 중립 2문장(A7N val 문장은 텍스트가 학습에 있음) | ✅ `d_eval.py prep --add_neutral`(2026-11-06) |
| 가상 화자 일관성 | 같은 모델 · 같은 목표 화자의 중립 ↔ 감정이 서로 한 사람인가(세트 1위 · 감정끼리 · 천장 · 갈라짐) + REF 음향 | ✅ `tools/vspk.py`(2026-11-06) |
| 표 | run 별 best 5 · 통합 상위 5 · 성공 기준 ✅/❌ · 일관성 순위 → 페이지 맨 위 | ✅ `tools/d_eval_tables.py`(2026-11-06) |
| 여러 번 만들어 고르기 | seed N개 중 생성 중립에 가장 가까운 것(조건: CER · 이상 · 감정) + 진짜 녹음 문턱 · cfg/prompt 노브 → 청음 페이지 `bestofn/` | ✅ `tools/bestofn.py`(2026-11-06) |
| 감정 ref 판 D 재측정 | unseen8 8명의 기쁨 · 분노 녹음을 ref 로 7칸 — 지시문 감정 · ref 감정 새어 나옴 · arousal 끌림 · 목소리(무감정 ref 판과 짝 비교) | ✅ `tools/d_eval_eref.py`(2026-11-07 · `page` = 작은 청음 페이지 8줄 × 모델 4) |
| val_xref 다시 재기 | 저장된 ckpt 로 감정 ref 판 val loss(트레이너 검증과 같은 계산 · 잡음 seed 3개 평균) — 이전 run 비교 | ✅ `tools/val_xref_loss.py`(2026-11-07) |
| 칸별 목소리 이동 | 생성 중립 2개 평균 ↔ 감정 칸 유사도(두 잣대) · 진짜 녹음 천장 | ✅ `tools/cell_drift.py`(2026-11-07) |
| CFG 시험 | 다시 학습 없이 추론 cfg(2.0 · 2.5 · 3.0)만 바꿔 D 재측정 입력 224 채점 · 작은 청음 페이지(4칸 × 2명 × cfg 3 × step 2) | ✅ `tools/cfg_test.py`(2026-11-07) |
| 감정 ↔ 목소리 그림 | run · step 별 (e2v 6칸, 목표 유사도) 점 · 선 · 원본 · CFG 점 · ±1 표준오차 → 맞바꿈 곡선을 넘었는가 | ✅ `tools/emo_voice_plot.py`(2026-11-07 · `--group 542|unseen8 --small` = 묶음별 작은 그림) |
| 542 vs unseen8 나눠 보기 | D 재측정을 화자 묶음(542 8명 = 무감정 낭독만 · unseen8 8명 = 감정 녹음 있음) × A칸 · B칸으로 — e2v · arousal · 목표 유사도 · 1위 · CER · 이상 + 화자별 e2v | ✅ `tools/group_split.py`(2026-11-07) |
| 진짜 사람 녹음 열 | D 재측정 청음 페이지 unseen8 표에 칸마다 진짜 감정 녹음 1개(비교용 · 말투 · 문장 다름) — q3_extra val_unseen8 → 없으면 71349 metadata(강도 1 → A · 3 → B · 투표 ≥ 4.5 · 2~10초) → 없음 · 평가 문장 · ref 녹음 제외 · seed 1986 | ✅ `tools/real_cells.py` → `d_eval/real_cells.json`(2026-11-07 · 35/56칸) |
- 비교 규칙: 같은 문장 · 같은 ref · 같은 지시문 표 · cfg 2.0 · timesteps 10 · seed 1986/1987/1988 · **준비 합성 1회 뒤**.
- 평가 화자: 처음 보는 화자 8명(`A7N/val_unseen8`) + 학습 화자 일부.

## 2. 청음
- **꼭 필요할 때만 · 적은 줄 · HTML 버튼 판정.** 페이지는 `~/workspaces/tmp_listen/vox_koft/` 에 만들고 8765 서버로 엽니다.
- 페이지 형식(q3tts_koft 와 같음): 맨 위 청음 가이드 → 줄 = ref × 칸, 열 = 원본 ref → 비교 기준 → 후보 → 줄마다 판정 + 메모 · localStorage · CSV · **자동 선택 결과는 `<details>` 로 접어 둠**.
- **재현 확인 뒤 공개**: 페이지의 wav 가 노트북 · 엔진으로 같은 seed 에서 똑같이 나오는지 먼저 봅니다.
- 판정 규칙: 칸당 표본 1개면 **확실한 칸이 3칸 이상**일 때만 차이가 있다고 봅니다.

## 3. step 고르기
- TB 처음 보는 화자 val(`val_unseen/*`) 상위 + 평가 상위 → 귀로 확정(`PICK_STEP`).
- val loss 가 좋다고 감정 · 목소리가 좋은 것은 아닙니다(q3tts_koft 에서 거듭 확인).
