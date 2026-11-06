# A7N (vox_koft 판) — VoxCPM2 학습 데이터

> 📅 2026-11-05 · 처음 보는 사람용 · 만드는 코드 `voxko/data/prepare.py` · 원천 설명 `~/workspaces/q3tts_koft/docs/datasets/README.md`

---

## 0. 한눈에
- 원천은 q3tts_koft 의 **A7N**입니다(감정 7칸 · 106,150문장 · 111 화자 번호 · 124h · 24kHz).
  - 7칸 = Neutrality · Happy-A/B · Sad-A/B · Angry-A/B
- 이 데이터를 **16kHz wav**(VoxCPM2 학습 입력)로 바꾸고 JSONL 을 새로 씁니다.
- **지시문은 여기서 굽지 않습니다.** 학습할 때 `configs/emotion/a7n_vox_v1.yaml` 로 `"(지시문)문장"` 을 만듭니다. 그래서 표를 바꿔도 다시 변환할 필요가 없습니다.
- **ref** 는 같은 화자의 **무감정** 문장입니다. 학습 때 문장 · epoch 마다 고르고, 비율은 `data.ref.ratio`(기본 0.7)입니다.
  - 나머지 0.3 은 **ref 없음**(지시문 + 문장만)이고, 다른 감정 ref 는 쓰지 않습니다. 근거와 1.0 판 제안은 [plan](../plan/20261105_success_criteria_and_next_runs.md) §3.

```bash
python -m voxko.data.prepare a7n --limit 100      # smoke → A7N_smoke100/
python -m voxko.data.prepare a7n --workers 32     # 전체 → A7N/   (대량 생성 · 승인 뒤)
python -m voxko.data.prepare r542                 # 542 무감정(섞기용 · 아직 안 씀)
```

## 1. 산출물 (`~/workspaces/preprocessed/vox_koft/A7N/`)
| 파일 | 내용 |
|---|---|
| `wavs16k/<utt>.wav` | 16kHz · mono · PCM_16. 24kHz 원본을 soxr HQ 로 변환 |
| `train.jsonl` · `val.jsonl` | `utt · audio · text · raw_text · emotion · speaker · gender · age · src_corpus · person · duration · src_audio` |
| `ref_pool.jsonl` | 무감정 ref 후보(`utt · audio · speaker · duration · text · source`). 길이 거르기는 학습 때 |
| `val_unseen8/{val,ref_pool}.jsonl` | 처음 보는 화자 8명(평가 전용 · 76문장 · ref 8). 학습에서 제외 |
| `dataset_info.json` | 건수 · 시간 · 화자 · 감정 분포 · ref 풀이 없는 화자 목록 |

## 2. 규칙 (2026-11-05 사용자 결정)
1. **val**: 화자마다 고정 표본 2문장(seed 1986 · 서로 다른 감정 먼저). 문장이 10개 미만인 화자(aihub_0024_ani · 2문장)는 val 을 뽑지 않습니다(전부 val 로 가서 train 에서 사라지지 않게). val 문장은 train · ref 풀에서 빠집니다.
2. **ref 풀**
   - A7N 안에 있는 같은 화자의 Neutrality 문장을 씁니다(화자 50명).
   - 여기에 `q3_extra/ref_neutral466`(466 화자 무감정 ref 전용, 화자당 30개)를 더합니다(화자 35명).
3. **ref 가 없는 화자 26명은 ref 없이**(지시문 + 문장만) 학습합니다.
   - `_ani`(애니체) 배역 18개: 본인의 평소 목소리를 빌리지 않습니다. 목소리가 다른 ref 로 잘못 가르치지 않기 위해서입니다.
   - 무감정 녹음이 전혀 없는 화자 8명
4. **처음 보는 화자 8명**(`q3_extra/val_unseen8/dataset_info.json`)이 train 에 있으면 변환이 멈춥니다(실제로는 없음).
5. 본문에 괄호 `( )` 가 있으면 멈춥니다(지시문과 헷갈리므로). A7N 에는 0개입니다.
6. 텍스트는 A7N 의 정규화된 `text` 를 그대로 씁니다(ko_tts_norm qwen3tts 프로필). 학습 때 공백만 추론과 같게 정리합니다.

## 3. 숫자
| 판 | train | val | ref 풀 | 비고 |
|---|---|---|---|---|
| `A7N_smoke100` | 88(화자 6) | 12 | 77(화자 5) | 2026-11-05 · 원본과 길이 차 ≤ 0.00002초 · 상관 > 0.999 |
| `A7N` | 105,938(화자 111 · 124.0h) | 212(화자 106) | 16,300(화자 86 · A7N 15,220 + 466 1,080) | 2026-11-05 · 변환 1분 48초 · 실패 0 · 14GB · 처음 보는 화자 누출 0 · 표본 500개 16kHz mono |
| `542` (섞기용) | 26,484(화자 1,000 · 28.0h · 모두 Neutrality) | — | 같은 화자 다른 문장 | 2026-11-05 · 45초 · 실패 0 · 3.1GB · 처음 보는 화자(val_unseen8 · val_unseen542 142명) 누출 검사 통과 |

- ref 가 없는 화자 26명의 문장 = train 의 **7,308문장(6.9%)** → 실제 ref 비율은 약 0.7 × 0.93 ≈ 65%.

## 4. 품질 점검 (2026-11-06 · 정본 [reports/20261106_four_runs_preflight.md](../reports/20261106_four_runs_preflight.md))
- **끝 무음**: A7N 중앙 0.27초 · p99 0.37초 · 0.5초 넘는 파일 0.02~0.10%. 542 는 0.01~0.18%. 사용자 결정으로 **그대로 둡니다**.
- **정규화기 변경(q3tts 쪽 `2be0315` · `d6de874`)**: 지금 qwen3tts 프로필로 다시 정규화해도 A7N · 542 저장 텍스트와 13만 2천 문장 모두 같습니다(영문자 0). 영향 없음.
