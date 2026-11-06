"""난수 상태 저장/복원 — validation·media 합성이 학습 난수열을 바꾸지 않게 한다.

샘플링 합성(multinomial)은 토큰마다 CUDA RNG 를 소비한다. 함수 전체를 감싸 4종(torch CPU·CUDA,
python random, numpy)을 복원해야 media 주기를 바꿔도 학습이 재현된다.
"""
from __future__ import annotations

import contextlib
import random
from typing import Optional

import numpy as np
import torch


@contextlib.contextmanager
def preserved_rng(seed: Optional[int] = None):
    """with preserved_rng(1986): ... — 블록 안에서는 seed 고정(주면), 나오면 원래 상태로."""
    state = (torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             random.getstate(), np.random.get_state())
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)          # CUDA 전 디바이스 포함
        yield
    finally:
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state_all(state[1])
        random.setstate(state[2])
        np.random.set_state(state[3])
