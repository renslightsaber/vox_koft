# vox 지시문 — 11/09 16:4x KST: 워크샵 전/후 음원 한 번 더 — REF 를 aihub_0156(남 · unseen8)으로

> 작성: q3tts 세션 · 사용자 결정 11/09 · 앞 지시문: [20261109_1600_prompt_workshop_audio_full.md](20261109_1600_prompt_workshop_audio_full.md)
> 서비스 REF 로 만든 `workshop/` 은 사용자 청음에서 **학습 모델이 서비스 목소리를 못 따라갔습니다.** 그래서 발표용으로는 학습 · 평가 때와 같은 종류의 화자(처음 보는 화자 aihub_0156)로 한 번 더 만듭니다. 기존 `workshop/` 은 지우지 않고 그대로 둡니다(나중에 확인 자료로 씀).

## 할 일
- 16:00 지시문과 **모든 것이 같고 REF 만 바꿉니다.**
  - Base(VoxCPM2 원본) vs NUX2sN s3000(`A7NUX2sN_xref_LoRA_01_r32_lr5e5` step_3000 `export_lora`)
  - 노트북 `notebooks/lora/A7NUX2sN_xref_LoRA_01_r32_lr5e5_listening.ipynb` 와 같은 설정
  - 문장 8개(N1 · N2 · HA · HB · SA · SB · AA · AB) · seed 1986 · 1987 · 1988
- **REF = aihub_0156**: D 재측정과 같은 REF 를 씁니다.
  - 파일: `~/workspaces/tmp_listen/vox_koft/d_eval/ref/aihub_0156_ref.wav`
  - 원본: `preprocessed/vox_koft/A7N/val_unseen8/wavs16k/71349_F-NX-Q-156-0315.wav`
  - 전사문: "시아버지는 나이가 많았지만 성격도 좋고 재산도 많았습니다." · 5.0초
  - D 재측정과 다르게 읽히는 부분(16k ↔ 24k 등)이 있으면 D 재측정 쪽을 따릅니다.
- 결과 폴더: `~/workspaces/tmp_listen/vox_koft/workshop_0156/` → `http://127.0.0.1:8765/vox_koft/workshop_0156/index.html`
  - 파일 이름 · manifest · 재현 확인 · 페이지 형식은 16:00 지시문과 같습니다.
- 페이지 맨 위 안내 한 줄: "REF = 처음 보는 남성 화자(학습에 없음) · 같은 문장을 Base 와 학습 모델로"
- 자동 검사(CER · 클리핑 · 같은 사람 점수)로 깨짐이 의심되는 것을 페이지와 채팅에 적습니다.
  - Base 와 학습 모델 각각 seed 1986 이 깨졌으면 교체 후보 seed 도 적습니다.
- 학습 · 삭제 · 커밋은 하지 않습니다. GPU 2,3.
- 끝나면 짧게 보고합니다: 주소 · 재현 · 깨짐 의심 id · 걸린 시간
