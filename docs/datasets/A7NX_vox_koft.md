# A7NX · A7NX3 — "같은 사람 · 다른 칸 ref" 학습 데이터 (vox_koft)

> 📅 2026-11-06 16:45 KST · 처음 보는 사람용 · 만든 곳: q3tts_koft 세션 `tools/make_a7nx.py`(이 폴더에 둠 · seed 1986 · 결정적)
> 정본 숫자: `~/workspaces/preprocessed/vox_koft/{A7NX,A7NX3}/dataset_info.json` · 바탕 데이터: [A7N_vox_koft.md](A7N_vox_koft.md)

---

## 0. 한눈에
- A7N 학습 문장(정답)마다, **같은 사람의 다른 칸 녹음 하나를 ref** 로 붙인 메타데이터입니다. wav 는 새로 만들지 않고 A7N 16kHz 파일을 가리킵니다.
- 지시문은 학습 때 **정답 칸 것**을 붙입니다(기존 A7N 과 같은 표 · 같은 비율).
- 노리는 것: "목소리는 ref, 감정은 지시문"으로 떼어 놓기 · A 만 · B 만 녹음한 화자의 빈칸 채우기(D안 데이터).
  - 배경: [감정 칸별 목소리 이동](../reports/20261106_cell_voice_drift_analysis.md) · [여러 번 만들어 고르기](../reports/20261106_bestofn_voice_filter.md)

| 데이터 | 행 | 정답 | 화자 | ref |
|---|---|---|---|---|
| `A7NX` | 104,664 | 104,664 | 96 | 정답마다 1개 고정 |
| `A7NX3` | 313,992 | 104,664 | 96 | 정답마다 서로 다른 3개(행 3개) |

## 1. ref 규칙
- 같은 화자이고, **정답과 다른 칸**이며, 길이 2~10초이고, 자기 자신이 아닙니다. 모든 행에 ref 가 있습니다.
- 묶음 고르기 비율은 무감정 0.4 · A 0.4 · B 0.2 입니다. 화자에게 없는 묶음은 빼고 다시 나누므로, 실제 비율은 **N 36.6% · A 37.7% · B 25.7%** 입니다.
- 무감정 후보 = A7N 학습 Neutrality + 466 무감정 ref 풀입니다.
- 칸(중립 포함)이 1개뿐인 화자 16명(1,274행)은 뺐습니다.
- 검사 결과: 같은 칸 ref 0 · 자기 자신 0 · 다른 화자 0. vox 쪽에서도 A7NX3 전체 행을 다시 세어 같은 칸 · 자기 자신이 0 인 것을 확인했습니다.
- A7NX 가 "고정 1개"인 이유: 3,000 step(배치 32)은 약 0.9 epoch 라서, 매 step 무작위로 뽑는 것과 사실상 같습니다. 1차와 2차가 같은 코드 경로를 씁니다.

## 2. 행 필드 (A7N 행 + 아래)
| 필드 | 뜻 |
|---|---|
| `utt_x` | 행 이름(`<utt>__x<k>`) — A7NX3 는 같은 정답이 행 3개라서 난수 키 · 로그에 이것을 씀 |
| `ref_utt` · `ref_audio` | ref 녹음(16kHz) |
| `ref_audio24` | 원본(24kHz) — q3tts 용 |
| `ref_emotion` · `ref_kind` | ref 칸 · 묶음(N/A/B) |
| `ref_duration` | ref 길이(초) — 학습 길이 계산에 씀 |

- `val_xref.jsonl`(192행): 감정 ref 판 검증 세트입니다. 학습 설정의 `data.val_xref_jsonl` 로 넣으면 TB `val_xref/*` 에 나옵니다.

## 3. 학습에서 쓰는 법 (2026-11-06 코드)
- `voxko/data/dataset.py`
  - 행에 `ref_audio` 가 있으면 그 ref 를 **항상** 씁니다(`data.ref.ratio` · 후보 풀 무시). 길이도 그 ref 로 계산합니다.
  - 없는 행은 이전과 완전히 같습니다(테스트 `test_fixed_ref_row_always_used_and_old_rows_unchanged`).
- 검증: 기존 `A7N/val.jsonl` + 무감정 풀(LoRA_01 과 같은 val loss) + `val_xref`.
- 설정: `configs/experiments/lora/A7NX_xref_LoRA_01_r32.yaml`(3,000 step) · `A7NX3_xref_LoRA_01_r32.yaml`(9,000 step).
- 메모리: 2GPU 최장 배치(정답 21~23초 + ref 6.7~8.8초)에서 실제 점유가 최대 54.1GB 입니다(배치 8 × accum 2 · LoRA_01 은 55.7GB).
- 다시 만들기: `~/miniforge3/envs/q3tts/bin/python tools/make_a7nx.py`. 고칠 때는 이유를 기록합니다.

## 🆕 A7NX2R — A7NX3 수정안(11/07 00:35 · q3tts 세션 작성 · 사용자 결정)
- ref 3개 → **2개** + **542 무감정 992명 × 12문장**(같은 화자 다른 문장 ref 2개 · Neutrality 지시문) + 음질 나쁜 정답 · ref 제외
- 229,820행(A7N 206,310 + 542 23,510) · 무감정 비율 23.2%(A7NX 14.5%) · 검사 모두 0 · `~/workspaces/preprocessed/vox_koft/A7NX2R/` · 도구 `tools/make_a7nx2r.py`
- 자세히: `~/workspaces/q3tts_koft/docs/datasets/A7NX_q3tts.md` §8 · 왜 542 인가: [docs/explanations/20261107_why_mix_542.md](../explanations/20261107_why_mix_542.md)
- 학습 지시문: [_handoff/20261107_0035_prompt_a7nx_s1_and_a7nx2r.md](../../_handoff/20261107_0035_prompt_a7nx_s1_and_a7nx2r.md)

## 🆕 A7NX2Rs — A7NX2R 수정안: 감정 되찾기(11/07 10시 · 사용자 결정)
**왜**
- A7NX2R 두 판은 목소리가 지금까지 중 가장 좋았지만 감정이 약해졌습니다(e2v 6칸 47~54% vs LoRA01 52~57%).
- 원인을 **무감정 비율 14.5% → 23.2%** 로 봅니다. 4,500 step 동안 감정 정답을 연습한 횟수가 A7NX3 보다 약 10% 적었습니다.
- 자세한 의견: [A7NX3 · A7NX2R 의견](../explanations/20261107_opinion_A7NX2R_A7NX3.md)(q3tts_koft 세션 작성 · 복사본)

**바꾼 것은 하나뿐** — 무감정 정답(A7N 무감정 · 542)의 ref 를 2개 → **1개**
- 감정 정답 176,400행은 A7NX2R 과 **똑같습니다**(같은 정답 · 같은 ref, 코드로 확인).
- 542 화자 992명 · 문장 11,755개도 그대로입니다. 반복만 줄였습니다.
- 비유: 성대모사 학원에서 "1,000명 명단과 대본은 그대로, 평소 말투 대본만 두 번 → 한 번 연습". 남는 수업 시간은 감정 연기로 갑니다.

| | A7NX3 | A7NX2R | **A7NX2Rs** |
|---|---|---|---|
| 행 수 | 313,992 | 229,820 | **203,110** |
| 정답당 ref | 3 | 2 | 감정 2 · 무감정 1 |
| 542 | 없음 | 992명 × 12 · ref 2 | 992명 × 12 · **ref 1** |
| 무감정 비율 | 14.5% | 23.2% | **13.2%** |
| 542 비중 | 0 | 10.2% | **5.8%** |
| 4,500 step 동안 감정 정답 연습 | 약 12.3만 | 약 11.1만 | **약 12.5만** |
| 542 화자당 연습(4,500 step) | 0 | 약 15번 | 약 8.5번 |

**기대와 리스크**
- 기대: 감정이 A7NX3 수준으로 돌아오고, 처음 보는 화자 목소리 이득은 남음.
- 리스크 1: 목소리 이득이 "542 를 많이 본 양" 덕이었다면 함께 줄어 A7NX3 와 비슷해질 수 있습니다(그러면 "542 는 양이 필요하다"를 배움).
- 리스크 2: 무감정 정답의 ref 가 1개라 "감정 ref → 무감정으로 말하기" 연습이 절반입니다. 감정 ref 판의 "중립 칸 중립" 지표를 봐야 합니다.

**파일 · 검증**
- `~/workspaces/preprocessed/{vox_koft,q3tts_koft}/A7NX2Rs/{train.jsonl, val_xref.jsonl, dataset_info.json, _READY}` · 도구 `~/workspaces/vox_koft/tools/make_a7nx2rs.py`
- 검사 7개(같은 칸 ref · 자기 자신 · 음질 걸림 정답 · ref · 처음 보는 화자 누출 · 중복 · 다른 화자 ref) 모두 0 · 두 프로젝트 행 순서 같음 · 파일 표본 있음
- val_xref = A7NX 것 그대로(192)
- 학습: vox_koft `A7NX2Rs_xref_LoRA_01_r32_lr5e5`(lr 5e-5 한 판 · 사용자 결정) — vox 지시문 [_handoff/20261107_1145_prompt_x2r_cleanup_cfg_a7nx2rs.md](../../_handoff/20261107_1145_prompt_x2r_cleanup_cfg_a7nx2rs.md) · 11/07 15:38 ~ 16:27 KST 학습 끝(Claude nohup · GPU 2,3)

## 🆕 A7NX2Rs6 — 542 를 절반으로(11/07 15시 · q3tts 세션 작성 · 사용자 결정)
- 왜: X2R02 의 감정 ↓ 는 **542 화자에 몰려 있습니다**(D 재측정 542 A칸 e2v LoRA01 42% → X2R02 s4500 19% · unseen8 은 ↑). 가설 = 542 행("낭독 목소리 ref → 늘 무감정")이 만든 지름길.
- 바꾼 것: A7NX2Rs 에서 **542 만 화자당 12 → 6문장**(992명 그대로) · A7N 행은 완전히 같음 · 542 행은 A7NX2Rs 의 부분집합
- 수량: 197,296행 = A7N 191,355 + 542 5,941 · 542 **3.0%**(A7NX2Rs 5.8%) · 무감정 **10.6%**(13.2%, 542 가 무감정이라 함께 줄어듦)
- 파일: `~/workspaces/preprocessed/vox_koft/A7NX2Rs6/` · 도구 `tools/make_a7nx2rs6.py` · 검사 7개 0
- 자세한 설명: q3tts_koft `docs/datasets/A7NX_q3tts.md` §10 · 지시문 `_handoff/20261107_1530_prompt_a7nx2rs_x2_full.md`
- 학습: vox_koft `A7NX2Rs6_xref_LoRA_01_r32_lr5e5`(설정은 A7NX2Rs 판과 데이터 경로만 다름) · 11/07 16:28 ~ 17:17 KST 학습 끝(Claude nohup · GPU 2,3)

## 🆕 A7NU — 안 쓴 71349 화자에서 새 5목소리(11/07 18:38 · 사용자 결정 · q3tts 세션 작성)
- 왜
  - 542 를 줄여도(A7NX2Rs6) 542 화자 A칸 감정이 돌아오지 않았습니다([결과](../reports/20261107_A7NX2Rs_results.md)).
  - 그래서 감정 녹음이 있는 화자를 늘리고, 542 있음/없음을 따로 가릅니다.
- 고른 목소리(사용자 26명 말투 청음 · 페이지 `~/workspaces/tmp_listen/datasets/a71349_unused26/` · 판정 CSV `a71349_unused26_v1_판정.csv`)
  - 애니체 4명: `aihub_0006_ani`(여아) · `aihub_0019_ani`(6~7세 남아) · `aihub_0030_ani`(할머니) · `aihub_0036_ani`(아줌마)
  - 친절체 1명: `aihub_0019`
  - 애니체 성우는 사람마다 배역이 하나뿐입니다. 그래서 "윗줄과 목소리가 다르다"는 메모는 감정 연기 때문으로 봅니다.
    - 근거: 같은 화자 안 무감정 ↔ 감정 평균 목소리 유사도가 0.977~0.994 입니다(Qwen 화자 인코더). A7N 화자(중앙값 0.99 · 최저 0.971) 범위 안입니다.
  - **강도 2 → B**: A7N 도 예전 대화체의 강도 2 를 B 로 둡니다. 이 5목소리는 강도 3 이 없어서 B칸이 모두 강도 2 입니다.
  - 무감정 중 0019 애니체("딱딱함")와 0036 애니체("애매")는 정답에서 빼고 ref 로만 씁니다.
  - ⚠️ 0019 친절체 분노A(93개)는 상담 대본이라 분노가 약할 수 있습니다(q3tts 세션 메모).
- 새 행: 9,274행 · 정답 5,234개(화자 · 칸별 수는 `A7NUX2Rs6/dataset_info.json` counts.a7nu_by_speaker)
- 두 데이터(A7N · 542 행 · val · val_xref 는 A7NX2Rs6 과 똑같음 · 검사 8개 모두 0 · vox 세션이 코드로 다시 확인)

| 데이터 | 행 | 542 | 무감정 | run |
|---|---|---|---|---|
| `A7NUX2Rs6` | 206,570 | 5,941(2.9%) | 10.7% | `A7NUX2Rs6_xref_LoRA_01_r32_lr5e5` |
| `A7NUX2s` | 200,629 | 0 | 8.05% | `A7NUX2s_xref_LoRA_01_r32_lr5e5` |

- 파일: `~/workspaces/preprocessed/vox_koft/{A7NU, A7NUX2Rs6, A7NUX2s}/` · 도구 `tools/make_a7nux2rs6.py --n542 6 | 0`
- 학습: 11/07 18:44 ~ 20:26 KST 두 판 연속(Claude nohup · GPU 2,3 · 사용자 승인 18:38) · 평가 21:21 끝
- 결과([보고서](../reports/20261107_A7NU_results.md)): 새 5목소리는 숫자로 효과 없음 · 542 를 0 으로 하면 542 화자 A칸 감정 e2v 32 → 42%(무감정 비율도 함께 줄어 원인은 아직 못 가름)
- 자세한 설명: q3tts_koft `docs/datasets/aihub_71349_unused26.md` · 지시문 `_handoff/20261107_1840_prompt_a7nu_full.md`

## 🆕 A7NUX2sN — 542 를 빼고 무감정 비율만 되돌린 판(11/07 21:36 · 사용자 결정 · q3tts 세션 작성)
- 왜: A7NUX2s(542 = 0)에서 542 화자 A칸 감정이 돌아왔지만, 무감정 비율도 10.7 → 8.05% 로 함께 줄었습니다. 그래서 원인이 둘 중 무엇인지 모릅니다.
  - (가) **542 지름길**: "낭독 목소리 ref → 늘 무감정 정답" 행이 "이런 목소리면 무감정"을 가르침
  - (나) **무감정 비율**: 무감정 연습이 줄어 감정 연습 비중이 커짐
- 만든 방법: `tools/make_a7nux2rs6.py --n542 0 --neu_fill`(q3tts 세션이 `--neu_fill` 을 더함)
  - A7NUX2s 의 모든 행 + **무감정 정답 5,941개**(= A7NUX2Rs6 의 542 행 수)에 같은 규칙으로 두 번째 ref 를 붙인 행
  - 추가 행: A7N 5,506 · A7NU 435 · 화자 53명 · ref 종류 A 3,405 / B 2,536
  - ⚠️ 무감정 정답의 ref 는 규칙상 다른 칸이라 **추가 행의 ref 는 모두 감정(A · B) 녹음**입니다. 542 행은 "무감정 ref → 무감정 정답"이었으므로 이 점도 다릅니다.
- vox 세션 확인(코드): 검사 8개 모두 0 · A7NUX2s ⊂ A7NUX2sN · A7NUX2Rs6 의 542 아닌 행 ⊂ A7NUX2sN · val_xref = A7NUX2Rs6 것과 같음 · 무작위 20행 소리(정답 + ref 40개) 16kHz 열림

| 데이터 | 행 | 542 행 | 무감정 비율 |
|---|---|---|---|
| A7NUX2Rs6 | 206,570 | 5,941 | 10.7% |
| A7NUX2s | 200,629 | 0 | 8.05% |
| **A7NUX2sN** | **206,570** | **0** | **10.69%** |

- run: `A7NUX2sN_xref_LoRA_01_r32_lr5e5`(A7NUX2s 설정 복제 · 데이터 경로 · run_name · 주석만 다름)
- 학습: 11/07 21:56 ~ 22:46 KST(Claude nohup · GPU 2,3 · 사용자 승인 21:36) · 평가 23:17 끝
- 결과([보고서](../reports/20261107_A7NUX2sN_results.md)): 542 A칸 e2v 9 step 평균 41.0%(A7NUX2s 42.1% · A7NUX2Rs6 31.7%) → 원인은 거의 **542 행**(+9.3%p) · 무감정 비율 효과는 +1.1%p(잡음)
- 지시문: `_handoff/20261107_2145_prompt_a7nux2sn_full.md`

## 🆕 A7NUX2sNn — 추가 무감정 행의 ref 만 무감정 녹음으로(11/09 00시 · 지시문 q3tts 세션 11/08 23:25 §5 · vox 세션이 만듦)
- 왜: A7NUX2sN 으로 "542 화자 A칸 감정 약화의 원인은 542 행 자체"로 결론 났습니다. 남은 물음은 **모델이 542 행의 무엇을 단서로 삼았나**입니다.
  - (가) 542 화자의 목소리 · 녹음 느낌 자체
  - (나) "무감정 ref → 무감정 정답"이라는 행 모양
  - A7NUX2sN 의 추가 행은 ref 가 모두 감정 녹음이라, 화자와 모양이 함께 바뀌어 가를 수 없었습니다.
- 만든 방법: `tools/make_a7nux2rs6.py --n542 0 --neu_fill_neutral`(이번에 더함 · 기존 `--neu_fill` 동작은 그대로)
  - A7NUX2sN 과 **같은 무감정 정답 5,941개**(`neufill` 표본 그대로)에 두 번째 행을 붙입니다.
  - 그 행의 ref = **같은 화자의 다른 무감정 녹음**입니다.
    - 후보: A7N/ref_pool · A7N · A7NU 무감정 녹음 · 2~10초 · 음질 bad 제외
    - seed `1986|neufill_neu|<utt>`
  - 그 화자에게 다른 무감정 녹음이 없으면 같은 규칙(`1986|neufill_neu_repl`)으로 다른 무감정 정답을 대신 뽑습니다 → **바꾼 것 0개**.
  - 추가 행에는 `ref_rule: "neufill_neu"` 를 적습니다. "같은 칸 ref" 검사는 이 행만 뺍니다(일부러 같은 칸이므로).
- vox 세션 확인(코드)
  - 기존 검사 8개 모두 0
  - 206,570행 · 무감정 10.69% → A7NUX2sN 과 같음
  - A7NUX2s ⊂ A7NUX2sNn · A7NUX2sN 과 겹치는 행 200,629(= A7NUX2s 전부)
  - 추가 5,941행의 정답 = sN 추가 행의 정답(같은 utt 집합) · 추가 행 ref 의 칸은 **모두 Neutrality**(sN 은 모두 감정 칸)
  - 추가 행 출처 A7N 5,506 · A7NU 435 · 화자 53명 · ref 길이 중앙 4.1초(sN 4.16초)
  - val_xref = A7NUX2sN 것과 같음
  - q3tts 쪽 사본 206,570행 · 행 순서 · ref 같음 · 추가 행 ref 소리 모두 있음(두 쪽)
  - 무작위 추가 20행 소리(정답 + ref 40개) 16kHz 열림
  - `_READY` 있음(두 쪽)
- 출력: `~/workspaces/preprocessed/{vox_koft,q3tts_koft}/A7NUX2sNn/{train.jsonl, val_xref.jsonl, dataset_info.json, _READY}`

| 데이터 | 행 | 542 행 | 무감정 비율 | 추가 무감정 행의 ref |
|---|---|---|---|---|
| A7NUX2Rs6 | 206,570 | 5,941 | 10.7% | (542 행 · 무감정 ref) |
| A7NUX2sN | 206,570 | 0 | 10.69% | 감정 녹음(A 3,405 · B 2,536) |
| **A7NUX2sNn** | **206,570** | **0** | **10.69%** | **같은 화자 무감정 녹음 5,941** |

- run: `A7NUX2sNn_xref_LoRA_01_r32_lr5e5`(A7NUX2sN 설정 복제 · 데이터 경로 · run_name · 주석만 다름 · 지시문 v1)
- 해석(542 A칸 e2v 9 step 평균): 약 41%(sN 수준) → (가) 542 화자 자체 · 약 32%(Rs6 수준) → (나) 행 모양 · 사이 → 둘 다
- 결과: [20261109_A7NUX2sNn_results.md](../reports/20261109_A7NUX2sNn_results.md)
