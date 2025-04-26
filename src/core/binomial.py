from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class BinomialTree:
    tree: NDArray[np.float64]
    up_probs: NDArray[np.float64]
    periods: int
    dt: float
    node_count: int = 0

    def __post_init__(self) -> None:
        self.node_count = int((self.periods + 1) * (self.periods + 2) / 2)

    @staticmethod
    def to_idx(t: int, n: int) -> int:
        if n > t or n < 0:
            raise ValueError("n must be between 0 and t")
        return t * (t + 1) // 2 + n

    @staticmethod
    def from_idx(m: int) -> Tuple[int, int]:
        if m < 0:
            raise ValueError("m must be non-negative")
        t = int((np.sqrt(8 * m + 1) - 1) // 2)
        n = m - (t * (t + 1) // 2)
        if not (0 <= n <= t):
            raise ValueError(f"No valid (t,n) for m={m}")
        return t, n

    @staticmethod
    def parent(t: int, n: int) -> int:
        if t == 0:
            raise ValueError("t must be greater than 0")
        if n > t or n < 0:
            raise ValueError("n must be between 0 and t")
        if n == t:
            return BinomialTree.to_idx(t - 1, n - 1)
        else:
            return BinomialTree.to_idx(t - 1, n)

    @staticmethod
    def children(m: int) -> Tuple[int, int]:
        t, n = BinomialTree.from_idx(m)
        return BinomialTree.to_idx(t + 1, n), BinomialTree.to_idx(t + 1, n + 1)

    # Calculate the price using backward induction. Starting at the penultimate
    # period and working our way back to time 0.
    def price(self, T: float) -> float:
        periods = int(T / self.dt)
        if periods > self.periods:
            raise ValueError(
                "periods must be less than or equal to the number of periods in the tree"
            )
        node_count = int((periods + 1) * (periods + 2) / 2)
        tree = np.ones(node_count)
        # k-1 is the index of the uppermost node in the penultimate period.
        k = int(periods * (periods + 1) / 2)
        for i in range(k - 1, -1, -1):
            (d, u) = BinomialTree.children(i)
            p_up = self.up_probs[i]
            p_dn = 1 - p_up
            tree[i] = (
                tree[d] * np.exp(-self.tree[d] * self.dt) * p_dn
                + tree[u] * np.exp(-self.tree[u] * self.dt) * p_up
            )
        return float(tree[0])
