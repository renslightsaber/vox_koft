# A7NUX2sNn — 542 지름길의 단서는 "542 화자"인가, "무감정 ref → 무감정 정답" 행 모양인가 (2026-11-09)

> 지시문: [`_handoff/20261108_2325_prompt_overnight_full.md`](../../_handoff/20261108_2325_prompt_overnight_full.md) §5 · §6(q3tts 세션 작성 · 사용자 승인 11/08 23:2x)
> run: `A7NUX2sNn_xref_LoRA_01_r32_lr5e5` · 이름표 `NUX2sNnlr5e5_s*` · 데이터 [A7NX_vox_koft.md](../datasets/A7NX_vox_koft.md) A7NUX2sNn 절
> 앞 판: [A7NUX2sN](20261107_A7NUX2sN_results.md) · [A7NU 두 판](20261107_A7NU_results.md)
> 11/09 10:13: 재개용 파일(step_4500)을 사용자 승인으로 지웠습니다(9.4GB → 624MB). 청음 · 추론은 `export_lora` 로 그대로 됩니다.

## 한 줄 답
**(가) 542 화자 자체가 단서입니다.** "무감정 ref → 무감정 정답"이라는 행 모양은 단서가 아닙니다.
- 542 화자 A칸 e2v 9 step 평균: **39.6%**
  - A7NUX2sN(추가 행 ref = 감정 녹음) 41.0% 와 같은 수준입니다(차이 1.4%p · 잡음 안).
  - A7NUX2Rs6(542 행 있음) 31.7% 와는 멉니다.
- 542 화자 목소리 유사도도 sN 쪽입니다(A칸 0.671 · sN 0.668 · Rs6 0.687).
- 다른 칸 · 통합 점수는 sN 과 비슷합니다(9 step 평균 138.4 vs 140.2).
- 뜻: 모델은 "무감정 ref 를 받으면 무감정으로 말한다"를 배운 것이 아니라, **"542 녹음 같은 목소리(낭독 · 무감정만 녹음한 사람)면 감정을 약하게"**를 배웠습니다.

## 1. 무엇이 다른가
- 데이터: A7NUX2sN 과 행 수(206,570) · 무감정 비율(10.69%) · 추가 무감정 정답 5,941개가 모두 같습니다.
  - 다른 점은 하나: 추가 행의 ref 가 **감정 녹음(sN)** 이냐 **같은 화자의 다른 무감정 녹음(sNn)** 이냐.
  - 만든 방법 · 검사: [데이터 문서](../datasets/A7NX_vox_koft.md) A7NUX2sNn 절(`tools/make_a7nux2rs6.py --n542 0 --neu_fill_neutral`)
- 설정 diff(`A7NUX2sN_xref_LoRA_01_r32_lr5e5.yaml` → `A7NUX2sNn_xref_LoRA_01_r32_lr5e5.yaml` · 주석 줄 빼고)
```diff
-  run_name: A7NUX2sN_xref_LoRA_01_r32_lr5e5
+  run_name: A7NUX2sNn_xref_LoRA_01_r32_lr5e5
-  train_jsonl: A7NUX2sN/train.jsonl
+  train_jsonl: A7NUX2sNn/train.jsonl
-  val_xref_jsonl: A7NUX2sN/val_xref.jsonl
+  val_xref_jsonl: A7NUX2sNn/val_xref.jsonl   # 내용은 같음
```
- 학습: 11/09 00:23 ~ 01:47 KST(GPU 2,3 · 1시간 23분)
  - 학습 ① 평가를 같은 GPU 에서 함께 돌려서 평소(50분)보다 느렸습니다. 결과에는 영향이 없습니다(학습은 결정적).
  - 학습 전 백업 → smoke(3 step · 25.7GB · media 3줄 REF 있음) → nohup 학습
  - 학습 뒤 백업: export 9개 · 실패 0 · merged.yaml = 학습 전 백업 · 경고 1개(토크나이저 안내)
- 평가: D 재측정 9 step(02:13 끝) · 재현 9/9 · v1 문구

## 2. TensorBoard (참고)
| run | val s4500 | val_unseen 최저 · s4500 | val_xref s4500 | val_stop s4500 |
|---|---|---|---|---|
| A7NUX2Rs6 | 0.7916 | s3000 0.8072 · 0.8075 | 0.7809 | 0.0101 |
| A7NUX2sN | 0.7911 | s1000 0.8071 · 0.8076 | 0.7808 | 0.0096 |
| **A7NUX2sNn** | 0.7909 | s2000 0.8067 · 0.8071 | 0.7808 | 0.0094 |

- 곡선은 거의 겹칩니다. loss 로는 가를 수 없습니다.
- 새 TB 미리듣기(media): step 4500 의 pred ↔ REF WavLM 0.648 · 0.650 · 0.773 → 3줄 모두 REF 와 같은 목소리입니다.

## 3. 542 / unseen8 나눠 보기 (9 step 평균 · e2v 맞힘 · 칸마다 48개)
| run | 542 A | 542 B | unseen8 A | unseen8 B | 542 A 목소리 | 통합 점수 |
|---|---|---|---|---|---|---|
| A7NUX2Rs6(542 행 있음) | 31.7% | 51.4% | 63.0% | 61.1% | 0.687 | 137.5 |
| A7NUX2s(542 0 · 무감정 8%) | 42.1% | 56.9% | 55.8% | 63.0% | 0.669 | 139.3 |
| A7NUX2sN(추가 행 ref 감정) | 41.0% | 55.6% | 61.1% | 62.3% | 0.668 | 140.2 |
| **A7NUX2sNn(추가 행 ref 무감정)** | **39.6%** | 50.5% | 61.8% | 60.2% | 0.671 | 138.4 |

- 목소리 = WavLM · ERes2Net 목표 유사도 평균.
- step 별 542 A: s500 35 · s1000 33 · s1500 31 · s2000 42 · s2500 44 · s3000 42 · s3500 46 · s4000 38 · s4500 46%
  - 초반 3 step 은 Rs6 수준(31~35%)이고, s2000 부터 sN 수준(42~46%)입니다. 9 step 평균으로 판단합니다(한 점 ±7%p 는 잡음).
  - 초반이 낮은 것은 "행 모양"도 아주 조금 영향이 있을 수 있다는 뜻입니다. 그러나 뒤쪽 6 step(s2000~4500) 평균은 43.0% 로 sN 의 같은 6 step(43.8%)과 같습니다.
- 542 B 는 50.5% 로 sN(55.6%)보다 5%p 낮습니다. 칸마다 48개라 잡음 범위 안입니다.
- 전체 표 · 화자별: [figs/20261109_A7NUX2sNn_group_split.md](figs/20261109_A7NUX2sNn_group_split.md)

## 4. 전체 점수
- 9 step 평균: 통합 138.4 · e2v 53.0% · 공정 1위 90.2% · 목표 유사도 0.643 (sN: 140.2 · 54.7% · 90.5% · 0.643)
- step 별 통합 점수: s500 128.0 · s1000 134.0 · s1500 139.7 · s2000 139.1 · s2500 140.9 · s3000 141.3 · s3500 140.6 · s4000 140.5 · s4500 141.5
- best 5: 4500 · 3000 · 2500 · 3500 · 4000. 최고 141.5 는 sN s3000(144.9)보다 3점 낮습니다.
- s2000 에 끝 잘림 5개가 있습니다(다른 step 은 0~2개).

## 5. 판단 · 다음
- 542 지름길의 단서는 **542 화자(목소리 · 녹음 느낌) 자체**입니다. 무감정 ref 를 무감정 정답에 붙이는 행 모양은 문제가 아닙니다.
- 그래서
  - 542 같은 "무감정만 녹음한 화자"를 넣으면, 그런 목소리의 처음 보는 화자에게서 감정이 약해집니다.
  - 목소리를 늘리려고 542 를 넣는 것과, 그 목소리 화자의 감정을 살리는 것은 서로 부딪칩니다.
  - 무감정 ref 를 써서 무감정 행을 늘리는 것 자체는 괜찮습니다.
- 다음 판의 기본 데이터로는 A7NUX2sN 과 A7NUX2sNn 둘 다 됩니다. 점수는 sN 이 조금 높습니다.
- 청음
  - 노트북: `notebooks/lora/A7NUX2sNn_xref_LoRA_01_r32_lr5e5_listening.ipynb`(best 5 · GPU 1)
  - run 별 페이지: http://127.0.0.1:8765/vox_koft/listen_runs/NUX2sNnlr5e5.html(4 step · 4위가 1위와 2점 안)

## 다시 돌리기
```bash
~/miniforge3/envs/q3tts/bin/python tools/make_a7nux2rs6.py --n542 0 --neu_fill_neutral
./scripts/run.sh train lora --exp configs/experiments/lora/A7NUX2sNn_xref_LoRA_01_r32_lr5e5.yaml --hw rtxpro6000_2gpu --accel 2gpu_b
~/miniforge3/envs/voxcpm/bin/python tools/eval_runs.py --runs A7NUX2sNn_xref_LoRA_01_r32_lr5e5 --steps 500 1000 1500 2000 2500 3000 3500 4000 4500 --gpus 2 3 --only vox
~/miniforge3/envs/voxcpm/bin/python tools/group_split.py --tags NUX2sNnlr5e5_s500 … --out docs/reports/figs/20261109_A7NUX2sNn_group_split.md
```
