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
