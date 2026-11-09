# vox 보충 지시 — 11/09 15:30 KST: 청음 규칙 하나 더(B 비교 페이지 열 추가)

> 작성: q3tts 세션 · 사용자 결정 11/09 15시 · 14:00 지시문(파일럿)이 끝난 뒤 실행합니다.

- `docs/rules_in_this_proj/request_templates.md` §1 의 4-1(C 형식 run 별 페이지) 바로 아래에 **4-2** 를 더합니다.
  - **여러 모델 비교 페이지(B 형식)에 열 추가**: 새 run 의 1위 step(사용자가 고르면 그 step)을 B 비교 페이지(`tmp_listen/vox_koft/listen_cmp/` · 공통 세트 54줄)에 **열 하나로 더합니다.**
  - 기존 열 음원과 판정 저장 키(`cmp4_judg`)는 그대로 둡니다. 사용자가 이미 한 판정이 사라지면 안 됩니다.
  - 새 열 음원은 노트북 엔진 재현 확인 뒤에 공개합니다.
  - 열 머리에는 run · step · 점수를 적습니다.
  - 도구는 `tools/listen_pages.py`(`St["B"]` 에 이름표 추가 · `page`)입니다.
- `eval_listening.md` §2-1 에 한 줄 덧붙입니다.
- (q3tts_koft 도 같은 규칙을 `request_templates.md` §1 3b 로 넣었습니다.)
- 이 변경은 다음 커밋 때 함께 올립니다(사용자 요청 시).
