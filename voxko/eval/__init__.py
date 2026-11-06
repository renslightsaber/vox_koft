"""평가(틀) — CER · 화자 유사도 · 감정(SER 별도 env, 파일로 주고받음) · 끝 잘림.

metrics.py: 끝 잘림(tail_metrics) · CER. d_eval.py: D안 재측정(성공 판정 — 목소리 WavLM-large · emotion2vec · arousal · CER · 청음 페이지). CER(whisper) · 화자 유사도(WavLM-SV) · SER 은 첫 본 학습 뒤 사용자와
도구를 정하고 붙인다(q3tts_koft 의 q3ko/eval · ser/ 를 참고만 — 수정 금지).
"""
