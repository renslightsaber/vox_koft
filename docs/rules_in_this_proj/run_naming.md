# run 이름 규칙

> 📅 2026-11-05 · 처음 보는 사람용 · q3tts_koft 규칙을 옮김
> 모음 첫 장: [README](README.md) · 계열 · 폴더: [run_groups.md](run_groups.md)

---

## 0. 한눈에
- **run 이름 = 설정 파일 이름**(`configs/experiments/<sft|lora>/<run>.yaml`)이고, 설정 안의 `logging.run_name` 도 같아야 합니다(로더가 검사).
- 이름은 왼쪽부터 **데이터 → 조건 → 학습 방식 → 번호 → 덧붙임** 순서입니다.

```
A7N _ ref _ LoRA _ 01 _ r32
 │     │     │      │    └ 덧붙임: 짝 run 과 다른 점(r · lr · ref 비율 …)
 │     │     │      └ 번호: 같은 계열 안의 변형
 │     │     └ 학습 방식: FT(전체) · LoRA
 │     └ 조건: ref = 무감정 ref + 지시문 · noref = 지시문만 · inst0 = 지시문 없이
 └ 데이터: A7N · A7N542(542 섞음) · A7N100(smoke 100문장)
```

## 1. 칸별 뜻

### 데이터
| 값 | 뜻 | 정본 |
|---|---|---|
| `A7N` | q3tts_koft A7N 7칸 → 16kHz · 지시문은 학습 때 | [datasets/A7N_vox_koft.md](../datasets/A7N_vox_koft.md) |
| `A7N100` | A7N 100문장 smoke(화자 6명) — 인프라 검증용 | 같음 |
| `A7N542` | A7N + AIHub 542 무감정(`data.extra_sources`) — 아직 안 씀 | 같음 |

### 조건
- `ref` = `data.ref.ratio > 0`(같은 화자 무감정 ref) + 지시문 · `noref` = ref 없이 지시문만 · 지시문 표가 기본(v1)과 다르면 덧붙임 `_iv2` 처럼.

### 학습 방식
| 값 | 뜻 | 트레이너 |
|---|---|---|
| `FT` | 전체 FT(2.29B 모두) | `voxko/sft` · `run.sh train sft` |
| `LoRA` | LoRA(LM · DiT q/k/v/o) | `voxko/lora` · `run.sh train lora` |

### 덧붙임 (짝 run 과 다른 점만)
| 모양 | 뜻 | 예 |
|---|---|---|
| `_r16` · `_r64` | LoRA rank | `A7N_ref_LoRA_01_r16` |
| `_lr5e6` · `_lr5e5` | 학습률 | `A7N_ref_FT_01_lr5e6` |
| `_ref05` | ref 비율 0.5 | `A7N_ref_FT_02_ref05` |
| `_iv2` | **웹 조사판**: 지시문 표 v2(감정마다 4문구) · 지시문 0.9(빼는 10% 는 Neutrality 만) · ref 0.5 — 2026-11-06 | `A7N_ref_FT_02_iv2` · `A7N_ref_LoRA_02_r32_iv2` |
| `_smoke` | 짧은 동작 확인용 | `A7N100_ref_FT_00_smoke` |

## 2. 규칙
1. **이름을 바꾸면 설정 파일 이름 · `logging.run_name` · 머리 주석의 실행 줄 · 백업 폴더를 같이** 바꿉니다.
2. 이미 학습한 run 이름은 **다시 쓰지 않습니다**(재진입 거부 · resume 파일을 지운 run 은 `--resume` 불가).
3. 덧붙임은 **짝 run(비교 기준)과 다른 점만** 적습니다.
4. `_` 로 시작하는 실험 파일은 템플릿 — 로더가 실행을 거부합니다.
