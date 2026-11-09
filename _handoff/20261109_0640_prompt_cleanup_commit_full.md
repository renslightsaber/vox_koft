> ⚠️ **→ [20261109_0930_prompt_bcell_ref_study_full.md](20261109_0930_prompt_bcell_ref_study_full.md) §0 으로 합쳐졌습니다. 이 문서는 따로 실행하지 않습니다.**

# vox 지시문 — 11/09 06:40 KST: 새 run 2개 재개용 파일 정리 · 커밋 · 푸쉬

> 작성: q3tts 세션 · 사용자 승인 11/09 06:3x("둘 다 정리" · "vox 커밋+푸쉬")
> 앞 문서: [20261109_0220_handoff_main.md](20261109_0220_handoff_main.md)

밤사이 작업 수고 많으셨습니다. 사용자가 결과를 보고 아래 두 가지를 승인했습니다. 학습 · 합성 · 새 실험은 하지 않습니다.

## 1. 재개용 파일 정리(삭제 승인됨)
- 대상 run 2개입니다. 둘 다 채택하지 않는 판입니다.
  - `~/workspaces/outputs/vox_koft/ckpt/A7NUX2sN_xref_LoRA_02_r32_lr5e5_iv3`
  - `~/workspaces/outputs/vox_koft/ckpt/A7NUX2sNn_xref_LoRA_01_r32_lr5e5`
- 지울 것: 각 run `step_4500/` 의 재개용 파일만 지웁니다(`model.safetensors` · `optimizer.bin` · `random_states_*.pkl` · `custom_checkpoint_*.pkl`, 약 8.8GB).
  - 다른 step 처럼 `_RESUME_REMOVED` 표시를 남깁니다. 이 저장소에 정리 도구나 정해진 방식이 있으면 그것을 씁니다.
- 🚫 **`export_*` · `_COMPLETE` · `configs/` · `logs/` · `_smoke/` 와 평가 산출물은 지우지 않습니다.**
- 지우기 전에 지울 파일 목록과 크기를 출력합니다. 지운 뒤 `du -sh` 로 두 run 의 크기를 보고합니다.
- 다른 run(A7NUX2sN_01 등)은 건드리지 않습니다.

## 2. 커밋 · 푸쉬(승인됨)
- 0220 인수인계 §5 의 "의미 단위 제안"대로 나눠 커밋합니다.
  - ① TB 미리듣기 고침 ② 평가 도구(지시문 표 선택) ③ 데이터 A7NUX2sNn ④ iv3 · sNn 설정 · exp_reports · 노트북 ⑤ 조사 코드 3개 + listen_pages · instr_test ⑥ docs ⑦ handoff(이 지시문 포함)
- 노트북은 이 저장소의 규칙(출력 지운 사본)을 따릅니다. 큰 산출물 · 음원 · ckpt 는 올리지 않습니다.
- 푸쉬는 **원격 2곳 모두**: `origin`(GitHub) · `gitlab`
- 커밋 메시지 끝에는 이 저장소의 평소 서명 줄을 붙입니다.

## 3. 보고(채팅 · 짧게)
- 정리: 지운 크기 · 두 run 의 남은 크기
- 커밋: 해시와 한 줄 제목 목록 · 두 원격의 푸쉬 결과
- 인수인계 `_handoff/20261109_HHMM_handoff_main.md`(KST)에 위 두 가지를 짧게 적습니다.

## 참고 — 사용자 결정 대기(이번에는 하지 않음)
- 분노B 대책: B칸 녹음 고르기 판(+8반음 넘는 분노B · 기쁨B 녹음 줄이기) 여부
- Qwen3-TTS 데이터셋 모델 조합(NUX2sN 하나 / 기쁨A 만 A7NX3 / 숫자 추천 (라) / 섞기 청음 뒤)
- 참고: 분노B 서비스 문구 C2 는 11/08 사용자 청음(instr_test)에서 이미 7/8 "못하다"였습니다. 그래서 C0 유지 쪽입니다.
