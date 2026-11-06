"""평가 지표. 끝 잘림 기준은 q3tts_koft 와 같다(끝 무음 = 피크 대비 −45dB 이하 10ms 프레임 꼬리,
잘림 의심 = 끝 무음 < 50ms & 마지막 50ms 음량 > −45dB)."""
from __future__ import annotations

import numpy as np


def tail_metrics(wav: np.ndarray, sr: int, thr_db: float = -45.0) -> dict:
    y = np.asarray(wav, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return {"tail_sil_ms": 0.0, "last50_db": -120.0, "cut_suspect": True}
    peak = float(np.max(np.abs(y))) + 1e-9
    hop = max(1, sr // 100)
    n = len(y) // hop
    fr = y[: n * hop].reshape(n, hop) if n else y.reshape(1, -1)
    db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) / peak + 1e-9)
    k = 0
    for v in db[::-1]:
        if v > thr_db:
            break
        k += 1
    last = y[-max(1, sr // 20):]
    last_db = float(20 * np.log10(np.sqrt(np.mean(last ** 2)) / peak + 1e-9))
    tail_ms = k * 10.0
    return {"tail_sil_ms": tail_ms, "last50_db": last_db, "cut_suspect": bool(tail_ms < 50 and last_db > thr_db)}


def cer(ref: str, hyp: str) -> float:
    """문자 오류율(공백 제외). 둘 다 같은 정규화를 거친 뒤 넣는다."""
    import jiwer
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    return float(jiwer.cer(r, h)) if r else float(len(h) > 0)
