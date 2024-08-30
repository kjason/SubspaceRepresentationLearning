"""
Created on Sat May 11 2024

@author: Kuan-Lin Chen

https://arxiv.org/abs/2408.16605
"""
from typing import Iterator, List
import torch

# consistent rank sampling, see Section IV-E in the paper
class ConsistentRankBatchSampler(torch.utils.data.Sampler[List[int]]):
    def __init__(self,N: int, K: int, batch_size: int, drop_last: bool=False) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.N = N
        self.K = K
        self.total_size = self.N * self.K
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.samples = [(torch.randperm(self.N) + self.N * i).tolist() for i in range(self.K)]

    def __iter__(self) -> Iterator[List[int]]:
        while len(self.samples) != 0:
            k = torch.randint(len(self.samples),(1,)).item()
            if self.drop_last:
                yield self.samples[k][:self.batch_size]
                del self.samples[k][:self.batch_size]
                if len(self.samples[k]) < self.batch_size:
                    del self.samples[k]
            else:
                if len(self.samples[k]) <= self.batch_size:
                    yield self.samples[k]
                    del self.samples[k]
                else:
                    yield self.samples[k][:self.batch_size]
                    del self.samples[k][:self.batch_size]

    def __len__(self) -> int:
        if self.drop_last:
            return (self.N // self.batch_size) * self.K
        else:
            return ((self.N + self.batch_size - 1) // self.batch_size) * self.K