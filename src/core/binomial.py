from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class BinomialTree:
    """A class representing a binomial tree for interest rate modeling.

    The binomial tree is represented as a flattened array where each node is indexed
    by a single number m. The tree has a triangular structure where each time step t
    has t+1 nodes. The nodes are ordered from bottom to top, left to right.

    Attributes:
        tree: Array of interest rates at each node
        up_probs: Array of probabilities of moving up at each node
        periods: Number of time periods in the tree
        dt: Time step size
        node_count: Total number of nodes in the tree (automatically calculated)
    """

    tree: NDArray[np.float64]
    up_probs: NDArray[np.float64]
    periods: int
    dt: float
    node_count: int = 0

    def __post_init__(self) -> None:
        """Calculate the total number of nodes in the tree after initialization."""
        self.node_count = int((self.periods + 1) * (self.periods + 2) / 2)

    @staticmethod
    def to_idx(t: int, n: int) -> int:
        """Convert time step t and node number n to a single index m.

        Args:
            t: Time step (0-based)
            n: Node number at time t (0-based, from bottom to top)

        Returns:
            int: The single index m representing the node

        Raises:
            ValueError: If n is not between 0 and t
        """
        if n > t or n < 0:
            raise ValueError("n must be between 0 and t")
        return t * (t + 1) // 2 + n

    @staticmethod
    def from_idx(m: int) -> Tuple[int, int]:
        """Convert a single index m to time step t and node number n.

        Args:
            m: The single index representing a node

        Returns:
            Tuple[int, int]: The time step t and node number n

        Raises:
            ValueError: If m is negative or doesn't correspond to a valid node
        """
        if m < 0:
            raise ValueError("m must be non-negative")
        t = int((np.sqrt(8 * m + 1) - 1) // 2)
        n = m - (t * (t + 1) // 2)
        if not (0 <= n <= t):
            raise ValueError(f"No valid (t,n) for m={m}")
        return t, n

    @staticmethod
    def parent(t: int, n: int) -> int:
        """Get the index of the parent node.

        Args:
            t: Time step of the current node
            n: Node number at time t

        Returns:
            int: Index of the parent node

        Raises:
            ValueError: If t is 0 or n is not between 0 and t
        """
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
        """Get the indices of the two child nodes.

        Args:
            m: Index of the current node

        Returns:
            Tuple[int, int]: Indices of the down and up child nodes
        """
        t, n = BinomialTree.from_idx(m)
        return BinomialTree.to_idx(t + 1, n), BinomialTree.to_idx(t + 1, n + 1)

    def price(self, T: float) -> float:
        """Calculate the price of a zero-coupon bond using backward induction.

        Starting from the final period, recursively calculates the price at each node
        using the risk-neutral pricing formula and working backwards to time 0.

        Args:
            T: Time to maturity in years

        Returns:
            float: The price of the zero-coupon bond

        Raises:
            ValueError: If T/dt is greater than the number of periods in the tree
        """
        periods = int(T / self.dt)
        if periods > self.periods:
            raise ValueError(
                "periods must be less than or equal to the number of periods in the tree"
            )
        node_count = int((periods + 1) * (periods + 2) / 2)
        tree = np.ones(node_count)
        k = int(periods * (periods + 1) / 2)
        for i in range(k - 1, -1, -1):
            (d, u) = BinomialTree.children(i)
            p_up = self.up_probs[i]
            p_dn = 1 - p_up
            tree[i] = (
                tree[d] * np.exp(-self.tree[i] * self.dt) * p_dn
                + tree[u] * np.exp(-self.tree[i] * self.dt) * p_up
            )
        return float(tree[0])
