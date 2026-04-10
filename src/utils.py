"""
Shared utility helpers used across all modules.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import CalvanoModel


def find_nash_action_index(game: "CalvanoModel") -> int:
    """Index of the price level in game.A closest to the mean Nash price."""
    return int(np.argmin(np.abs(game.A - np.mean(game.p_nash))))


def find_greedy_equilibrium_state(game: "CalvanoModel") -> np.ndarray:
    """Converged state under fully greedy (eps=0) play."""
    state = np.array([game.k // 2] * game.n, dtype=int)
    for _ in range(500):
        actions = np.array([
            np.argmax(game.Q[(n,) + tuple(state)])
            for n in range(game.n)
        ])
        if np.array_equal(actions, state):
            break
        state = actions
    return state


def get_equilibrium_profits(game: "CalvanoModel"):
    """Return (pi_c_actual, pi_n_actual, eq_prices) from converged Q-tables."""
    eq_state = find_greedy_equilibrium_state(game)
    eq_actions = np.array([
        np.argmax(game.Q[(n,) + tuple(eq_state)])
        for n in range(game.n)
    ])
    eq_prices = game.A[eq_actions]
    pi_c_actual = float(np.mean(game._compute_profits(eq_prices)))
    pi_n_actual = game.pi_nash
    return pi_c_actual, pi_n_actual, eq_prices


def compute_profits_with_entrant(game: "CalvanoModel",
                                  firm_prices: np.ndarray,
                                  entrant_price: float) -> np.ndarray:
    """
    Per-firm profits when a synthetic competitive entrant prices at
    *entrant_price*.  The entrant is added to the logit denominator but
    does not learn and is not part of the state space.
    """
    e_firms   = np.exp((game.a_arr - firm_prices) / game.mu)
    e_entrant = np.exp((np.mean(game.a_arr) - entrant_price) / game.mu)
    e_outside = np.exp(game.a0 / game.mu)
    denom     = np.sum(e_firms) + e_entrant + e_outside
    return (firm_prices - game.c_arr) * (e_firms / denom)


def compute_consumer_surplus(game: "CalvanoModel",
                              prices: np.ndarray) -> float:
    """
    Consumer surplus under logit demand (log-sum formula):
        CS = mu * ln( sum_j exp((a_j - p_j)/mu) + exp(a0/mu) )
    """
    utilities = np.append(
        (game.a_arr - prices) / game.mu,
        game.a0 / game.mu
    )
    return game.mu * np.log(np.sum(np.exp(utilities)))


def compute_welfare(game: "CalvanoModel", prices: np.ndarray) -> dict:
    """Total welfare = consumer surplus + producer surplus."""
    cs = compute_consumer_surplus(game, prices)
    ps = float(np.sum(game._compute_profits(prices)))
    return {"consumer_surplus": cs, "producer_surplus": ps,
            "total_welfare": cs + ps}


def compute_hhi(game: "CalvanoModel", prices: np.ndarray) -> float:
    """Herfindahl-Hirschman Index from market shares under logit demand."""
    shares = game.demand(prices)
    # Include outside option share
    outside = 1.0 - np.sum(shares)
    all_shares = np.append(shares, outside) * 100  # percentage
    return float(np.sum(all_shares ** 2))
