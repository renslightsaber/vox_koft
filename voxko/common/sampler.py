"""배치 샘플러 — q3ko/sft/dataset.py 의 BucketBatchSampler · EvalBatchSampler · make_batches 를 그대로 옮김.

BucketBatchSampler: 길이 비슷한 것끼리 묶고 배치 순서를 섞어 rank 별로 나눈다. 순서 = seed + epoch 로 결정적
  → resume 때 set_epoch(e, start=k) 로 정확히 그 다음 배치부터. tag_epoch=True 면 (idx, epoch) 를 넘긴다
  (데이터셋이 epoch 마다 ref · 지시문을 다르게 고르되 worker 와 무관하게 결정적이도록).
EvalBatchSampler: GPU 수와 무관하게 같은 표본.
"""
from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional

from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[List[int]]):
    """[학습용] 길이가 비슷한 샘플끼리 묶고, 배치 단위로 섞어 랭크별로 나눠 준다.

    · batch_size 고정, 또는 max_tokens(패딩 포함 B × 최대길이 ≤ max_tokens) 동적 배치
    · 모든 랭크가 같은 수의 배치를 받는다(DDP collective 불일치 방지). multiple_of(= grad_accum) 의 배수로 잘라
      accumulation 창이 epoch 경계에 걸리지 않게 한다(잘린 몇 배치는 다음 epoch 에 다른 순서로 다시 나옴).
    · 순서는 seed + epoch 로 결정적 → resume 때 set_epoch(e) + start=k 로 **정확히 그 다음 배치부터** 이어간다.
    """

    def __init__(self, lengths: List[int], batch_size: int = 8, max_tokens: Optional[int] = None,
                 num_replicas: int = 1, rank: int = 0, seed: int = 0, bucket_mult: int = 50, multiple_of: int = 1,
                 tag_epoch: bool = False):
        self.tag_epoch = tag_epoch                     # True: (idx, epoch) 를 넘긴다(ref · 지시문이 epoch 마다 바뀜 · worker 와 무관)
        self.lengths, self.batch_size, self.max_tokens = lengths, batch_size, max_tokens
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.bucket_mult, self.multiple_of = bucket_mult, max(int(multiple_of), 1)
        self.epoch, self.start = 0, 0
        self._cache: Optional[tuple] = None

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        """epoch 의 순서를 정하고, 그 epoch 에서 이 rank 가 이미 소비한 배치 수(start)만큼 건너뛴다."""
        self.epoch, self.start = epoch, start

    def _all_batches(self) -> List[List[int]]:
        if self._cache and self._cache[0] == self.epoch:
            return self._cache[1]
        batches = make_batches(self.lengths, self.batch_size, self.max_tokens, self.bucket_mult,
                               random.Random(self.seed + self.epoch))
        per_rank = len(batches) // self.num_replicas
        per_rank -= per_rank % self.multiple_of
        if per_rank == 0 and batches:                  # 데이터가 아주 적으면 반복해 채운다(smoke)
            need = self.num_replicas * self.multiple_of
            batches = (batches * math.ceil(need / len(batches)))[:need]
            per_rank = self.multiple_of
        batches = batches[:per_rank * self.num_replicas]
        self._cache = (self.epoch, batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        mine = self._all_batches()[self.rank::self.num_replicas]
        for b in mine[self.start:]:
            yield [(i, self.epoch) for i in b] if self.tag_epoch else b

    def __len__(self) -> int:
        """이 rank 의 이번 epoch 전체 배치 수(start 와 무관)."""
        return len(self._all_batches()) // self.num_replicas


def make_batches(lengths: List[int], batch_size: int, max_tokens: Optional[int], bucket_mult: int,
                 rng: random.Random) -> List[List[int]]:
    """섞기 → bucket(batch_size × bucket_mult) 안에서 길이 정렬 → 배치 → 배치 순서 섞기."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    chunk = max(batch_size * bucket_mult, 1)
    batches: List[List[int]] = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: lengths[i])
        if max_tokens:
            cur: List[int] = []
            for i in part:
                if cur and (len(cur) + 1) * max(lengths[j] for j in cur + [i]) > max_tokens:
                    batches.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                batches.append(cur)
        else:
            batches += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


class EvalBatchSampler(Sampler[List[int]]):
    """[검증용] GPU 수와 무관하게 **같은 표본**을 평가한다.

    고정 seed 로 섞은 전체 배치 목록에서 앞 max_batches 개(전체 상한)를 고르고, 그걸 rank 에 나눠 준다.
    꼬리를 자르지 않는다 — 랭크마다 배치 수가 1 차이 날 수 있으므로, 검증은 DDP 를 거치지 않는
    (collective 없는) forward 로 돌리고 마지막에 합계만 all-reduce 한다.
    (이전 버그: 길이순 정렬 + 꼬리 자름 + rank 별 앞 50배치 → 짧은 발화만 평가, GPU 수에 따라 val 이 달라짐)
    """

    def __init__(self, lengths: List[int], batch_size: int = 8, max_tokens: Optional[int] = None,
                 num_replicas: int = 1, rank: int = 0, seed: int = 0, bucket_mult: int = 50,
                 max_batches: Optional[int] = None):
        batches = make_batches(lengths, batch_size, max_tokens, bucket_mult, random.Random(seed))
        self.total = len(batches) if not max_batches else min(len(batches), int(max_batches))
        self.mine = batches[:self.total][rank::num_replicas]

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.mine

    def __len__(self) -> int:
        return len(self.mine)
