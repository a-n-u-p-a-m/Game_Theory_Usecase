"""
CalvanoModel -- n-firm Bertrand game with logit demand.

Generalises the original 2-firm model from Calvano et al. (2020) to
arbitrary n, with optional asymmetric quality (a) and cost (c) vectors.
Default parameters reproduce the paper exactly for n=2.
"""

import numpy as np
from itertools import product as iterprod
from scipy.optimize import fsolve
from dataclasses import dataclass, field
from typing import Union
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class CalvanoModel:
    """
    Bertrand oligopoly with logit demand.

    Parameters
    ----------
    n       : number of firms (default 2)
    c       : marginal cost -- scalar (same for all) or array of length n
    a       : product quality -- scalar (same for all) or array of length n
    a0      : outside-option quality
    mu      : logit differentiation parameter
    alpha   : Q-learning step size
    beta    : exploration decay rate;  eps(t) = exp(-beta * t)
    delta   : discount factor
    k       : number of price levels in the action grid
    tstable : consecutive stable periods to declare convergence
    tmax    : maximum number of learning periods
    """
    # -- Economic parameters --
    n:       int   = 2
    c:       Union[float, np.ndarray] = 1.0
    a:       Union[float, np.ndarray] = 2.0
    a0:      float = 0.0
    mu:      float = 0.25

    # -- Learning parameters --
    alpha:   float = 0.15
    beta:    float = 4e-6
    delta:   float = 0.95

    # -- Grid / convergence --
    k:       int   = 15
    tstable: int   = int(1e5)
    tmax:    int   = int(1e7)

    # -- Derived (set in __post_init__) --
    a_arr:   np.ndarray = field(default=None, repr=False)
    c_arr:   np.ndarray = field(default=None, repr=False)
    A:       np.ndarray = field(default=None, repr=False)
    PI:      np.ndarray = field(default=None, repr=False)
    Q:       np.ndarray = field(default=None, repr=False)
    p_nash:  np.ndarray = field(default=None, repr=False)
    p_mono:  np.ndarray = field(default=None, repr=False)
    pi_nash: float      = field(default=None, repr=False)
    pi_mono: float      = field(default=None, repr=False)

    def __post_init__(self):
        # Broadcast scalar a / c to arrays
        self.a_arr = (np.full(self.n, self.a) if np.isscalar(self.a)
                      else np.asarray(self.a, dtype=float))
        self.c_arr = (np.full(self.n, self.c) if np.isscalar(self.c)
                      else np.asarray(self.c, dtype=float))
        assert len(self.a_arr) == self.n and len(self.c_arr) == self.n

        self.p_nash, self.p_mono = self._compute_equilibrium_prices()
        self.A      = self._build_price_grid()
        self.PI     = self._build_profit_tensor()
        self.pi_nash = float(np.mean(self._compute_profits(self.p_nash)))
        self.pi_mono = float(np.mean(self._compute_profits(self.p_mono)))
        self.Q       = self._init_Q()

    # ---- demand ---------------------------------------------------------
    def demand(self, p: np.ndarray) -> np.ndarray:
        """Logit demand: q_i = exp((a_i-p_i)/mu) / D."""
        e = np.exp((self.a_arr - p) / self.mu)
        denom = np.sum(e) + np.exp(self.a0 / self.mu)
        return e / denom

    # ---- first-order conditions -----------------------------------------
    def _foc_nash(self, p: np.ndarray) -> np.ndarray:
        """FOC for Nash: each firm maximises own profit independently.
        Works for arbitrary n and asymmetric (a, c)."""
        d = self.demand(p)
        return 1.0 - (p - self.c_arr) * (1.0 - d) / self.mu

    def _foc_monopoly(self, p: np.ndarray) -> np.ndarray:
        """FOC for joint-profit maximisation (cartel / monopoly).
        General n-firm formulation:
          dPi_joint/dp_i = q_i * [1 - (p_i-c_i)(1-q_i)/mu
                                    + sum_{j!=i} (p_j-c_j)*q_j / mu] = 0
        """
        d = self.demand(p)
        res = np.zeros(self.n)
        for i in range(self.n):
            cross = sum((p[j] - self.c_arr[j]) * d[j]
                        for j in range(self.n) if j != i)
            res[i] = 1.0 - (p[i] - self.c_arr[i]) * (1.0 - d[i]) / self.mu \
                     + cross / self.mu
        return res

    def _compute_equilibrium_prices(self):
        p0 = np.ones(self.n) * 3.0 * np.mean(self.c_arr)
        p_nash = fsolve(self._foc_nash, p0)
        p_mono = fsolve(self._foc_monopoly, p0)
        return p_nash, p_mono

    # ---- price grid -----------------------------------------------------
    def _build_price_grid(self) -> np.ndarray:
        inner = np.linspace(np.min(self.p_nash), np.max(self.p_mono),
                            self.k - 2)
        d = inner[1] - inner[0]
        return np.linspace(inner[0] - d, inner[-1] + d, self.k)

    # ---- profit tensor --------------------------------------------------
    def _compute_profits(self, p: np.ndarray) -> np.ndarray:
        return (p - self.c_arr) * self.demand(p)

    def _build_profit_tensor(self) -> np.ndarray:
        """PI[a1, a2, ..., an, i] = profit of firm i at joint action."""
        dims = tuple([self.k] * self.n)
        PI = np.zeros(dims + (self.n,))
        for idx in iterprod(*[range(self.k)] * self.n):
            prices = self.A[np.array(idx)]
            PI[idx] = self._compute_profits(prices)
        return PI

    # ---- Q-table init (FIX B1: works for arbitrary n) -------------------
    def _init_Q(self) -> np.ndarray:
        """
        Q[n_idx, s1, ..., sn, a] = avg profit of firm n_idx playing
        action a, averaged over all opponent action combinations,
        discounted at 1/(1-delta).
        """
        sdim = tuple([self.k] * self.n)
        Q = np.zeros((self.n,) + sdim + (self.k,))
        for n_idx in range(self.n):
            # Average over all opponent action dimensions
            axes_to_avg = tuple(j for j in range(self.n) if j != n_idx)
            pi_avg = np.mean(self.PI[..., n_idx], axis=axes_to_avg)
            # pi_avg has shape (k,) -- broadcast to all states
            Q[n_idx] = np.broadcast_to(
                pi_avg / (1.0 - self.delta), sdim + (self.k,)
            ).copy()
        return Q

    def reset_Q(self):
        self.Q = self._init_Q()

    # ---- utility --------------------------------------------------------
    def profit_index(self, prices: np.ndarray) -> float:
        """Delta in [0,1]: (pi - pi_Nash) / (pi_Mono - pi_Nash)."""
        pi = float(np.mean(self._compute_profits(prices)))
        gap = self.pi_mono - self.pi_nash
        return (pi - self.pi_nash) / gap if abs(gap) > 1e-12 else 0.0

    def summary(self) -> str:
        a_str = (f"{self.a_arr[0]:.2f}" if np.all(self.a_arr == self.a_arr[0])
                 else str(np.round(self.a_arr, 2)))
        c_str = (f"{self.c_arr[0]:.2f}" if np.all(self.c_arr == self.c_arr[0])
                 else str(np.round(self.c_arr, 2)))
        lines = [
            "=" * 60,
            "  Calvano et al. (2020) -- Model Summary",
            "=" * 60,
            f"  Firms (n)          : {self.n}",
            f"  Marginal cost (c)  : {c_str}",
            f"  Quality (a)        : {a_str}",
            f"  Outside opt (a0)   : {self.a0}",
            f"  Differentiation (mu): {self.mu}",
            f"  Discount (delta)   : {self.delta}",
            f"  Learning rate (alpha): {self.alpha}",
            f"  Exploration (beta) : {self.beta}",
            f"  Price levels (k)   : {self.k}",
            f"  Max periods        : {self.tmax:,.0f}",
            f"  Convergence window : {self.tstable:,.0f}",
            "-" * 60,
            f"  Nash prices        : {np.round(self.p_nash, 4)}",
            f"  Monopoly prices    : {np.round(self.p_mono, 4)}",
            f"  Price grid         : [{self.A[0]:.4f}, ..., {self.A[-1]:.4f}]",
            f"  Nash profit (mean) : {self.pi_nash:.4f}",
            f"  Monopoly profit    : {self.pi_mono:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)
