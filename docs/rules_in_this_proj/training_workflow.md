# 학습 한 번의 순서 (Claude · 사용자 역할)

> 📅 2026-11-05 · 처음 보는 사람용 · q3tts_koft 규칙을 옮김
> 모음 첫 장: [README](README.md) · 도구 사용법 정본: [guides/sft_training.md](../guides/sft_training.md) · [guides/lora_training.md](../guides/lora_training.md)

---

## 0. 한눈에

```
설정 만들기 → 검토(짝 run 과 diff) → smoke → 학습 전 백업 → 실행 명령 안내
   → (사용자) byobu 에서 실행 → 중복 실행 · 시작 확인
   → (끝나면) 종료 확인 → 학습 뒤 백업 → 평가 · 청음 → 보고서 → (승인 뒤) resume 정리 → (요청 시) 커밋
```

| 단계 | 누가 | 결과물 |
|---|---|---|
| 설정 · 검토 · smoke · 학습 전 백업 | Claude | `configs/experiments/...` · `exp_reports/<sft\|lora>/<run>/config_backup/` |
| **학습 실행** | **사용자(byobu)** | `ckpt/<run>/` · `runs/<run>/` |
| 종료 확인 · 학습 뒤 백업 · 평가 · 보고서 | Claude | 보고서 · 청음 페이지/노트북 |
| resume 정리 · 삭제 | Claude(**승인 뒤**) | `_RESUME_REMOVED` |

## 1. 단계별 규칙

### ① 설정
- 새 run 은 **짝 run(비교 기준) 설정을 복사**해서 다른 점만 바꿉니다. 머리 주석에 목적 · 짝 · 실행 줄을 적습니다.
- 병합 결과를 짝 run 과 **diff** 해서, 의도한 키만 다른지 확인합니다.

### ② smoke
- `./scripts/run.sh train <sft|lora> --exp <yaml> --hw rtxpro6000_1gpu --gpu N --smoke` — 3 step · gradient 검사 · val 2배치 · 합성 1개 · export 생성 · 검증.
- 확인: `rc=0` · `SMOKE OK` · peak mem · 배너의 "ref 풀 없는 문장 · 길이로 제외" 수를 사용자에게 알립니다.

### ③ 학습 전 백업
- `exp_reports/<sft|lora>/<run>/config_backup/` = 설정 원본들 + accelerate + **감정 지시문 표** + `merged.yaml` + `md5.txt`.
  (학습이 시작되면 같은 내용이 `ckpt/<run>/configs/` 에도 자동 스냅샷으로 남습니다.)

### ④ 실행 명령 안내 → 사용자 실행
- Claude 는 run 별 명령(GPU 포함)을 코드 블록으로 줍니다. 시작 뒤 같은 run 의 프로세스 묶음이 하나인지 · GPU 번호 · 첫 step 로그를 확인합니다.

### ⑤ 끝난 뒤
1. 종료 확인: 로그 `Training End` · `step_*/_COMPLETE` 개수 · 오류 · 경고
2. 평가 · 청음 → 보고서([eval_listening.md](eval_listening.md))
3. **재개용 파일**: `train.resume: {enable, every, keep}` — 끝나면 보통 마지막 step 1개만 남습니다.
   - 크기(실측): 전체 FT **약 27GB**(fp32 가중치 9GB + optimizer 18GB) + export 4.6GB · LoRA **약 9GB**(accelerate 가 frozen 포함 전체 가중치를 저장) + export 70MB.
4. **resume 정리(승인 뒤)**: 재개용 파일(`model.safetensors` · `optimizer.bin` · `random_states_*` · `custom_checkpoint_*`)만 지우고 `_RESUME_REMOVED` 를 남깁니다. **`export_ft/` · `export_lora/` 는 절대 지우지 않습니다.**

## 2. 항상 지키는 것
- seed **1986** · **bf16** · 한국어 `normalize=False` + `ko_tts_norm(profile="qwen3tts")`.
- 삭제 · 모델 다운로드 · 긴 학습 / 대량 생성은 **사용자 승인 뒤**.
- `VoxCPM/` 은 고치지 않습니다. `~/workspaces/{q3tts_koft,cv3_koft}` 는 읽기만.
- 실패 · 건너뛴 단계는 숨기지 않고 그대로 보고합니다.
