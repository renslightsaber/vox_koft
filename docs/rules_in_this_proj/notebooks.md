# 청음 노트북 만들기 규칙

> 📅 2026-11-05 · 처음 보는 사람용 · q3tts_koft 규칙을 옮김(빌더는 아직 없음)
> 모음 첫 장: [README](README.md)

---

## 0. 한눈에
- 청음 노트북은 **run 하나에 하나** — `notebooks/<sft|lora>/<run>_listening.ipynb`.
- 커널 **"Python (voxcpm)"** · 합성은 `voxko.engine.Engine`(순수 PyTorch · 준비 합성 1회 자동).
- 🔒 **사용자가 실행 · 수정한 노트북은 다시 만들지 않습니다.** 고쳐야 하면 셀 단위로 최소 수정하고 알립니다.
- 빌더: `~/miniforge3/envs/voxcpm/bin/python notebooks/utils/build_listening_nb.py --runs <run>`(2026-11-06) — 도우미 `voxko/listen.py` · 셀 2 에 best 5 주석(D 재측정 참고 점수 + TB 처음 보는 화자 val) · 이미 있으면 덮어쓰지 않음(`--force` 는 사용자 확인 뒤) · 셀 구성은 사용자가 고치면 그 판을 틀로.
- 검증: `LISTEN_SMOKE=1 CUDA_VISIBLE_DEVICES=N jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=voxcpm --output-dir <scratch> <노트북>` — 출력은 사본에만.

## 1. 노트북에 꼭 들어갈 것 (초안)
- `RUN_NAME` · `PICK_STEP`(들을 step) · `CUDA_VISIBLE_DEVICES`
- 비교 열: **원본 VoxCPM2 → 후보 step**(같은 문장 · 같은 ref · 같은 seed)
- 줄: 처음 보는 화자 8명의 무감정 ref × 감정 7칸 (지시문 표 = 학습과 같은 것)
- 랭킹 주석 셀: TB 처음 보는 화자 val loss 상위 5 + (있으면) 평가 상위 5 + 끝 잘림 비율

## 2. 검증 · 저장
- `nbformat.validate` · 가능하면 앞 셀 실행으로 확인.
- git 을 쓰게 되면 출력 지운 사본만 올립니다(사용자 결정).
