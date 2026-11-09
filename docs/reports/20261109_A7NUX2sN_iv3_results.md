# A7NUX2sN iv3 — 학습 문구에서 "shouting · loud" 를 빼면 분노B 소리침이 줄어드는가 (2026-11-09)

> 지시문: [`_handoff/20261108_2325_prompt_overnight_full.md`](../../_handoff/20261108_2325_prompt_overnight_full.md) §2 · §6(q3tts 세션 작성 · 사용자 승인 11/08 23:2x · 학습도 vox 세션이 nohup 으로)
> run: `A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3` · 이름표 `NUX2sNiv3lr5e5_s*`(v3 문구로 잼) · `NUX2sNiv3v1_s*`(같은 ckpt · v1 문구 · 참고)
> 함께 볼 것: [분노B 정답 녹음 높이](20261109_angryB_pitch_in_data.md) · [11/08 시험 A](20261108_listen_pages_and_instr_test.md) §3
> 코드: [`tools/iv3_compare.py`](../../tools/iv3_compare.py)(분노B F0 · 지표 · C2 비교 합성) · 숫자 `~/workspaces/tmp_listen/vox_koft/iv3_cmp/table.json`
> 11/09 10:13: 재개용 파일(step_4500)을 사용자 승인으로 지웠습니다(9.4GB → 624MB). 청음 · 추론은 `export_lora` 로 그대로 됩니다.

## 한 줄 답
**재학습(문구만 바꾸기)으로는 분노B 가 내려가지 않습니다. 서비스 때 추론 문구만 바꾸는 쪽이 낫습니다.**
- v3 로 학습한 모델에 v3 문구를 주면 분노B 는 REF 대비 **+8.2반음**입니다(s4000). 지금 모델(A7NUX2sN s3000 · v1 문구)은 +7.9반음입니다. 9 step 평균도 +7.6 vs +8.2 로 거의 같습니다.
  - 학습 초반(s500 · s1000)에는 +5.1 · +6.6 으로 낮았습니다. 원본 모델이 "without shouting" 의 뜻을 아직 읽기 때문입니다.
  - s2000 부터는 +8~9 로 올라갑니다. 학습하면서 문구가 "높은 분노B 녹음"을 가리키는 **이름표**가 되어 버렸습니다.
  - 목소리는 조금 나빠졌습니다(분노B 목표 유사도 9 step 평균 0.590 vs 0.601).
- 같은 문항에서 **A7NUX2sN s3000 + 추론 문구 C2(분노A 문구)** 는 **+3.9반음** · 유사도 **0.632** 입니다. 대신 angry 맞힘이 88 → 66% 로 줄어듭니다.
- 다른 6칸 · 통합 점수는 크게 떨어지지 않았습니다(9 step 평균 138.8 vs 140.2 · e2v 54.8 vs 54.7%).
- 결론
  - 학습 문구를 바꾸는 재학습은 필요 없습니다(효과 없음).
  - 당장은 **서비스 문구로 조절**합니다. C2 는 "누른 분노"로 들릴 수 있어서 청음으로 세기를 봐야 합니다.
  - 근본 대책은 데이터입니다. 분노B 정답 녹음 자체가 무감정보다 +5.4반음 높습니다([높이 조사](20261109_angryB_pitch_in_data.md)).

## 1. 무엇을 바꿨나
- 감정 표 `configs/emotion/a7n_vox_v3.yaml`: v1 에서 **분노B 한 줄만** 바꿨습니다(나머지 6칸은 글자까지 같음).
  - v1: `Furious and shouting, strong anger, loud and harsh`
  - v3: `Strong anger, intense and forceful, firm voice without shouting`
- 설정 diff(`A7NUX2sN_xref_LoRA_01_r32_lr5e5.yaml` → `A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3.yaml` · 주석 줄 빼고)
```diff
-  run_name: A7NUX2sN_xref_LoRA_01_r32_lr5e5
+  run_name: A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3
+  emotion:
+    map_file: configs/emotion/a7n_vox_v3.yaml   # ← v3(분노B 만 v1 과 다름)
```
- 데이터 · lr 5e-5 · r32 · 4,500 step · 저장 · 평가 500 은 그대로입니다.
- 학습: 11/08 23:29 ~ 11/09 00:22 KST(52분 · GPU 2,3 · 1.5 step/s · 최대 메모리 47.9GB)
  - 학습 전 백업 → smoke(3 step · 25.8GB · media 3줄 REF 있음) → nohup 학습
  - 학습 뒤 백업: export 9개 · 실패 0 · merged.yaml = 학습 전 백업 · 경고 1개(토크나이저 안내 · 늘 나오는 것)
- 평가 도구 수정(이번에 더함)
  - `voxko/eval/d_eval.py vox --emotion_map`(기본 v1) · `vox_<tag>.json` 에 `emotion_map` 을 적음
  - `tools/eval_runs.py --emotion_map · --short`
  - 청음 노트북(`voxko/listen.py`)은 run 의 merged.yaml 에 적힌 표로 합성

## 2. TensorBoard (참고)
| run | val s4500 | val_unseen 최저 · s4500 | val_xref s4500 | val_stop s4500 |
|---|---|---|---|---|
| A7NUX2sN | 0.7911 | s1000 0.8071 · 0.8076 | 0.7808 | 0.0096 |
| **iv3** | 0.7942 | s1000 0.8078 · 0.8084 | 0.7838 | 0.0100 |

- iv3 가 val 에서 약 0.003 높습니다. 곡선 모양은 같습니다.
- 🔧 **TB 미리듣기(media)를 이번 run 부터 고쳤습니다.** 3줄 모두 REF 가 있는 행이고, REF 소리 · 이름이 함께 기록됩니다.
  - step 4500 의 pred ↔ REF WavLM 코사인: 0.645 · 0.657 · 0.745(같은 화자 진짜 녹음끼리 0.52~0.83 수준)
  - 예전 run(A7NUX2sN_01)의 media 는 REF 없이 만들어져 pred ↔ 정답이 0.14 · 0.30(다른 사람)이었습니다.
  - 안내: [tensorboard_groups.md](../guides/tensorboard_groups.md) "media 소리" 절

## 3. 분노B — 같은 D 재측정 32문항(16명 × 2문장)
F0 = REF 대비 반음(시험 A 와 같은 pyin 범위 규칙 · 유성 0.3초 미만 제외).

| 모델 · 문구 | F0 중앙 | (542 · unseen8) | +8 넘음 | 유사도 WavLM · ERes | 목표 1위 | angry |
|---|---|---|---|---|---|---|
| A7NUX2sN s3000 · v1(지금) | +7.9 | +8.8 · +7.4 | 45% | 0.593 · 0.583 | 86% | 88% |
| **A7NUX2sN s3000 · C2 문구** | **+3.9** | +5.0 · +0.9 | **12%** | **0.632 · 0.633** | 84% | **66%** |
| iv3 s4000 · v3(학습과 같음) | +8.2 | +7.2 · +8.7 | 55% | 0.588 · 0.545 | 81% | 91% |
| iv3 s1000 · v3 | +6.6 | +7.1 · +5.5 | 32% | 0.630 · 0.629 | 92% | 88% |
| iv3 s4000 · v1(참고) | +9.5 | +9.4 · +11.4 | 69% | 0.552 · 0.547 | 73% | 91% |
| iv3 s1000 · v1(참고) | +9.0 | +7.7 · +9.5 | 57% | 0.565 · 0.603 | 84% | 88% |

- step 별(v3 문구 · F0 중앙): s500 +5.1 · s1000 +6.6 · s1500 +6.7 · s2000 +8.0 · s2500 +8.7 · s3000 +8.1 · s3500 +8.2 · s4000 +8.2 · s4500 +8.9 → **9 step 평균 +7.6**
- 같은 step 의 A7NUX2sN(v1): +10.2 · +9.8 · +7.6 · +7.8 · +7.5 · +7.9 · +7.8 · +7.3 · +7.7 → **9 step 평균 +8.2**
  - 둘 다 초반에 높낮이가 흔들리고, s2000 뒤로는 +7.5~9 에 모입니다. 뒤쪽 step 은 iv3 가 오히려 조금 높습니다.
- 분노B 목표 유사도 9 step 평균(WavLM): iv3 0.590 · A7NUX2sN 0.601. angry 맞힘 89.2 vs 90.6%.
- 이 모델에 v1 문구("shouting")를 주면 +9~9.5 로 더 올라갑니다. v3 로 학습했어도 모델은 "shouting" 의 뜻을 여전히 읽습니다.
- 확인: 분노A 는 iv3 · v3 와 iv3 · v1 이 같은 값입니다(두 표에서 분노A 문구가 같으므로 · 합성이 결정적이라는 확인).

## 4. 다른 6칸 · 통합 점수 (9 step 평균)
| | 통합 점수 | e2v 7칸 | 공정 1위 | 목표 유사도 | CER | 이상 |
|---|---|---|---|---|---|---|
| A7NUX2sN | 140.2 | 54.7% | 90.5% | 0.643 | 2.1% | 2.9% |
| iv3 | 138.8 | 54.8% | 88.6% | 0.639 | 2.1% | 2.6% |

- 칸별 e2v(9 step 평균 · A7NUX2sN → iv3)
  - 중립 52.8 → 55.9 · 기쁨A 48.6 → 51.4 · 기쁨B 22.9 → 25.0 · 슬픔A 25.3 → 28.8
  - **슬픔B 63.2 → 54.9** · 분노A 79.2 → 78.5 · 분노B 90.6 → 89.2
  - 슬픔B −8%p 는 칸마다 32문항이라 잡음 범위(±8%p)의 끝입니다.
- 542 / unseen8(group_split · [표](figs/20261109_A7NUX2sN_iv3_group_split.md))
  - 542 A칸 e2v 9 step 평균 42.3%(A7NUX2sN 41.0%) · unseen8 A 63.8%(A7NUX2sN 약 61%) → 같은 수준
- 통합 점수 1위 step: s4000 142.9 · s4500 141.6 · s1000 141.8 · s3000 141.2 → best 5 = 4000 · 1000 · 4500 · 3000 · 3500
  - A7NUX2sN 최고(s3000 144.9)보다 2점 낮습니다.

## 5. 판단
- "학습 문구를 바꾸면 소리침이 줄어든다"는 가설은 **아닙니다**. 문구는 학습하면서 데이터(분노B 녹음)에 묶입니다. 그 녹음이 높으면 문구가 무엇이든 높아집니다.
- 추론 문구는 아직 효과가 있습니다. 모델이 학습하지 않은 문구(분노A 문구 · "shouting")는 원본 모델이 아는 뜻대로 움직입니다.
- 그래서
  - **재학습 불필요.** v1 모델(A7NUX2sN)을 그대로 쓰고, 분노B 서비스 문구를 C2 쪽으로 바꾸는 것이 가장 쌉니다. 다만 angry 판정이 66% 로 줄어 "분노B 다운 세기"가 남는지 청음이 필요합니다.
  - 학습으로 고치려면 데이터를 바꿔야 합니다(분노B 정답 중 +8반음 넘는 녹음 25% 를 빼거나 줄이기).

## 6. 청음
- 청음 노트북: `notebooks/lora/A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3_listening.ipynb`(best 5 · GPU 1 · v3 문구로 합성)
- run 별 페이지: http://127.0.0.1:8765/vox_koft/listen_runs/NUX2sNiv3lr5e5.html(공통 세트 6명 × 9줄 · 상위 3 step · v3 문구)
- 비교로 들을 것: 11/08 시험 A 페이지(C0 vs C2) http://127.0.0.1:8765/vox_koft/instr_test/index.html

## 다시 돌리기
```bash
V=~/miniforge3/envs/voxcpm/bin/python
$V tools/eval_runs.py --runs A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3 --steps 500 1000 1500 2000 2500 3000 3500 4000 4500 --gpus 2 3 --only vox --emotion_map configs/emotion/a7n_vox_v3.yaml
$V tools/eval_runs.py --runs A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3 --steps 1000 4000 --gpus 2 3 --only vox --short NUX2sNiv3v1
# 점수(sv · ser · quality)는 d_eval.py 를 tag 로 · C2 비교는 tools/iv3_compare.py 머리말 순서대로
```
