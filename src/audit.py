"""
M2 -- Audit Protocol: inject competitive entrant, measure p(T).
"""

import numpy as np
from typing import Dict, List, Optional

from .model import CalvanoModel
from .utils import (find_greedy_equilibrium_state,
                    find_nash_action_index,
                    compute_profits_with_entrant)


def _mean_ci95(values: List[float],
               lower: Optional[float] = None,
               upper: Optional[float] = None) -> Dict[str, float]:
    """Return mean/std and two-sided 95% CI for a sample of values."""
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return {
            'mean': mean,
            'std': 0.0,
            'ci_low': mean,
            'ci_high': mean,
        }
    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(arr.size)
    half = 1.96 * sem
    ci_low = mean - half
    ci_high = mean + half
    if lower is not None:
        ci_low = max(lower, ci_low)
        ci_high = max(lower, ci_high)
    if upper is not None:
        ci_low = min(upper, ci_low)
        ci_high = min(upper, ci_high)
    if ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low

    return {
        'mean': mean,
        'std': std,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
    }


def _pava_non_decreasing(values: List[float],
                         weights: Optional[List[float]] = None) -> List[float]:
    """Weighted isotonic regression (PAVA) enforcing non-decreasing sequence."""
    y = np.asarray(values, dtype=float)
    if y.size <= 1:
        return y.tolist()

    w = (np.ones_like(y)
         if weights is None
         else np.asarray(weights, dtype=float))
    if w.shape != y.shape:
        raise ValueError("weights must have same shape as values")

    lvl = y.copy()
    wt = w.copy()
    starts = np.arange(y.size)

    i = 0
    while i < lvl.size - 1:
        if lvl[i] <= lvl[i + 1] + 1e-12:
            i += 1
            continue

        merged = (wt[i] * lvl[i] + wt[i + 1] * lvl[i + 1]) / (wt[i] + wt[i + 1])
        lvl[i] = merged
        wt[i] = wt[i] + wt[i + 1]
        lvl = np.delete(lvl, i + 1)
        wt = np.delete(wt, i + 1)
        starts = np.delete(starts, i + 1)
        if i > 0:
            i -= 1

    out = np.empty_like(y)
    ends = list(starts[1:]) + [y.size]
    for block_level, begin, end in zip(lvl, starts, ends):
        out[begin:end] = block_level

    return np.clip(out, 0.0, 1.0).tolist()


def _enforce_monotone_curve(p_t: Dict[int, float]) -> Dict[int, float]:
    """Return non-decreasing p(T) curve using isotonic projection."""
    Ts = sorted(int(T) for T in p_t.keys())
    vals = [float(p_t[T]) for T in Ts]
    vals_iso = _pava_non_decreasing(vals)
    return {int(T): float(vals_iso[i]) for i, T in enumerate(Ts)}


def _price_gap(firm_prices: np.ndarray,
               nash_mean: float,
               detection_metric: str = 'max') -> float:
    """Compute audit signal gap above Nash using the requested metric."""
    metric = str(detection_metric).strip().lower()
    if metric == 'max':
        return float(np.max(firm_prices) - nash_mean)
    if metric == 'mean':
        return float(np.mean(firm_prices) - nash_mean)
    raise ValueError(f"Unsupported detection_metric='{detection_metric}'")


def calibrate_audit_tau(game: CalvanoModel,
                        detection_metric: str = 'mean',
                        n_samples: int = 1200,
                        competitive_noise: float = 0.05,
                        collusive_state_jitter: int = 2,
                        collusive_epsilon: float = 0.01,
                        competitive_quantile: float = 0.99,
                        collusive_quantile: float = 0.20,
                        min_margin: float = 0.01,
                        min_tau: float = 0.02,
                        max_tau: float = 0.50) -> Dict[str, float]:
    """
    Calibrate audit detection threshold from held-out signal distributions.

    The threshold is anchored by a high quantile of non-collusive (Nash-like)
    signal gaps and a low quantile of collusive signal gaps.
    """
    nash_mean = float(np.mean(game.p_nash))
    eq_state = find_greedy_equilibrium_state(game)
    nash_idx = find_nash_action_index(game)

    comp_gaps = []
    coll_gaps = []

    for _ in range(int(n_samples)):
        # Competitive proxy: Nash action with small random implementation noise.
        noise = np.round(np.random.normal(0.0, competitive_noise * game.k,
                                          size=game.n)).astype(int)
        comp_actions = np.clip(nash_idx + noise, 0, game.k - 1)
        comp_prices = game.A[comp_actions]
        comp_gap = max(0.0, _price_gap(comp_prices, nash_mean,
                                       detection_metric=detection_metric))
        comp_gaps.append(comp_gap)

        # Collusive proxy: greedy actions near converged collusive state.
        jitter = np.random.randint(-int(collusive_state_jitter),
                                   int(collusive_state_jitter) + 1,
                                   size=game.n)
        state = np.clip(eq_state + jitter, 0, game.k - 1)
        coll_actions = np.array([
            np.random.randint(0, game.k)
            if (collusive_epsilon > 0 and np.random.rand() < collusive_epsilon)
            else np.argmax(game.Q[(n,) + tuple(state)])
            for n in range(game.n)
        ])
        coll_prices = game.A[coll_actions]
        coll_gap = max(0.0, _price_gap(coll_prices, nash_mean,
                                       detection_metric=detection_metric))
        coll_gaps.append(coll_gap)

    q_comp = float(np.quantile(comp_gaps, competitive_quantile))
    q_coll = float(np.quantile(coll_gaps, collusive_quantile))

    # Keep tau above competitive tail while not drifting too close to collusive tail.
    tau_raw = max(q_comp + float(min_margin), 0.5 * (q_comp + q_coll))
    tau = float(np.clip(tau_raw, min_tau, max_tau))

    return {
        'tau': tau,
        'tau_raw': float(tau_raw),
        'detection_metric': str(detection_metric),
        'n_samples': int(n_samples),
        'competitive_noise': float(competitive_noise),
        'competitive_quantile': float(competitive_quantile),
        'collusive_quantile': float(collusive_quantile),
        'q_competitive': q_comp,
        'q_collusive': q_coll,
        'competitive_gap_mean': float(np.mean(comp_gaps)),
        'collusive_gap_mean': float(np.mean(coll_gaps)),
        'competitive_gap_std': float(np.std(comp_gaps, ddof=1)) if len(comp_gaps) > 1 else 0.0,
        'collusive_gap_std': float(np.std(coll_gaps, ddof=1)) if len(coll_gaps) > 1 else 0.0,
    }


def _update_q_with_entrant_feedback(game: CalvanoModel,
                                    Q_table: np.ndarray,
                                    state: np.ndarray,
                                    actions: np.ndarray,
                                    next_state: np.ndarray,
                                    profits: np.ndarray) -> None:
    """One-step Q-learning update under entrant-adjusted profits."""
    for n in range(game.n):
        idx = (n,) + tuple(state) + (actions[n],)
        old_q = Q_table[idx]
        max_q_next = np.max(Q_table[(n,) + tuple(next_state)])
        td_target = profits[n] + game.delta * max_q_next
        Q_table[idx] = (1.0 - game.alpha) * old_q + game.alpha * td_target


def run_single_audit(game: CalvanoModel, T_audit: int,
                     tau_threshold: float = 0.05,
                     start_from_equilibrium: bool = True,
                     entrant_price: float = None,
                     audit_epsilon: float = 0.0,
                     update_learning: bool = True,
                     detection_metric: str = 'max',
                     min_consecutive_hits: int = 1) -> dict:
    """
    One audit trial.  Profits are computed via the 3-firm logit model
    (entrant included in denominator).
    """
    if entrant_price is None:
        entrant_price = np.mean(game.c_arr)

    p_nash = np.mean(game.p_nash)
    state  = (find_greedy_equilibrium_state(game)
              if start_from_equilibrium
              else np.array([game.k // 2] * game.n, dtype=int))
    Q_run = game.Q.copy()

    detected     = False
    max_gap      = 0.0
    hits_needed = max(1, min(int(min_consecutive_hits), int(T_audit)))
    consecutive_hits = 0
    price_trace  = np.zeros((T_audit, game.n))
    profit_trace = np.zeros((T_audit, game.n))

    for t in range(T_audit):
        actions = np.array([
            np.random.randint(0, game.k)
            if (audit_epsilon > 0 and np.random.rand() < audit_epsilon)
            else np.argmax(Q_run[(n,) + tuple(state)])
            for n in range(game.n)
        ])
        firm_prices     = game.A[actions]
        price_trace[t]  = firm_prices
        profits = compute_profits_with_entrant(game, firm_prices,
                                               entrant_price)
        profit_trace[t] = profits
        gap = _price_gap(firm_prices, p_nash, detection_metric)
        max_gap = max(max_gap, gap)
        if gap > tau_threshold:
            consecutive_hits += 1
            if consecutive_hits >= hits_needed:
                detected = True
        else:
            consecutive_hits = 0
        next_state = actions.copy()
        if update_learning:
            _update_q_with_entrant_feedback(
                game, Q_run, state, actions, next_state, profits
            )
        state = next_state

    return {'detected': detected, 'max_price_gap': max_gap,
            'price_trace': price_trace, 'profit_trace': profit_trace}


def measure_detection_probability(game: CalvanoModel, T_audit: int,
                                   tau_threshold: float = 0.05,
                                   n_runs: int = 100,
                                   noise_std: float = 1.0,
                                   entrant_price: float = None,
                                   audit_epsilon: float = 0.05,
                                   detection_metric: str = 'max',
                                   min_consecutive_hits: int = 1) -> float:
    """Empirical p(T): fraction of audit runs detecting collusion."""
    if entrant_price is None:
        entrant_price = np.mean(game.c_arr)

    detections = 0
    hits_needed = max(1, min(int(min_consecutive_hits), int(T_audit)))
    eq_state = find_greedy_equilibrium_state(game)
    nash_mean = np.mean(game.p_nash)

    for _ in range(n_runs):
        # Keep training effects local to each audit run.
        Q_run = game.Q.copy()

        if noise_std > 0:
            radius = max(1, int(round(2.0 * float(noise_std))))
            perturb = np.random.randint(-radius, radius + 1, size=game.n)
            state = np.clip(eq_state + perturb, 0, game.k - 1)
        else:
            state = eq_state.copy()

        detected = False
        consecutive_hits = 0
        for _ in range(T_audit):
            actions = np.array([
                np.random.randint(0, game.k)
                if (audit_epsilon > 0 and np.random.rand() < audit_epsilon)
                else np.argmax(Q_run[(n,) + tuple(state)])
                for n in range(game.n)
            ])

            firm_prices = game.A[actions]
            gap = _price_gap(firm_prices, nash_mean, detection_metric)
            if gap > tau_threshold:
                consecutive_hits += 1
                if consecutive_hits >= hits_needed:
                    detected = True
            else:
                consecutive_hits = 0

            profits = compute_profits_with_entrant(
                game, firm_prices, entrant_price
            )
            next_state = actions.copy()
            _update_q_with_entrant_feedback(
                game, Q_run, state, actions, next_state, profits
            )
            state = next_state

            if detected:
                break

        if detected:
            detections += 1

    return detections / n_runs


def measure_p_T_curve(game: CalvanoModel,
                       T_values: List[int] = None,
                       tau: float = 0.05,
                       n_runs: int = 100,
                       entrant_price: float = None,
                       audit_epsilon: float = 0.05,
                       detection_metric: str = 'max',
                       min_consecutive_hits: int = 1,
                       enforce_monotone: bool = True,
                       verbose: bool = True) -> Dict[int, float]:
    """Measure p(T) for a list of audit durations T."""
    if T_values is None:
        T_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    if entrant_price is None:
        entrant_price = np.mean(game.c_arr)

    p_T = {}
    for T in T_values:
        p_T[T] = measure_detection_probability(
            game, T, tau_threshold=tau, n_runs=n_runs,
            noise_std=1.0, entrant_price=entrant_price,
            audit_epsilon=audit_epsilon,
            detection_metric=detection_metric,
            min_consecutive_hits=min_consecutive_hits,
        )
        if verbose:
            print(f"  T={T:>4d}:  p(T) = {p_T[T]:.3f}")

    if enforce_monotone:
        p_T = _enforce_monotone_curve(p_T)
    return p_T


def measure_p_T_curve_with_ci(game: CalvanoModel,
                              T_values: List[int] = None,
                              tau: float = 0.05,
                              n_runs_per_seed: int = 40,
                              seeds: Optional[List[int]] = None,
                              entrant_price: float = None,
                              audit_epsilon: float = 0.05,
                              detection_metric: str = 'max',
                              min_consecutive_hits: int = 1,
                              enforce_monotone: bool = True,
                              verbose: bool = True) -> Dict:
    """
    Measure p(T) over multiple random seeds and return mean + 95% CI.

    CI is computed across seed-level estimates, where each seed estimate
    uses n_runs_per_seed Monte Carlo audit trials.
    """
    if T_values is None:
        T_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    if seeds is None:
        seeds = [101, 202, 303, 404, 505]
    if entrant_price is None:
        entrant_price = np.mean(game.c_arr)

    curves_by_seed = {}
    for seed in seeds:
        np.random.seed(seed)
        curve = measure_p_T_curve(
            game,
            T_values=T_values,
            tau=tau,
            n_runs=n_runs_per_seed,
            entrant_price=entrant_price,
            audit_epsilon=audit_epsilon,
            detection_metric=detection_metric,
            min_consecutive_hits=min_consecutive_hits,
            enforce_monotone=enforce_monotone,
            verbose=False,
        )
        curves_by_seed[int(seed)] = {int(T): float(p) for T, p in curve.items()}

    summary = {}
    for T in T_values:
        vals = [curves_by_seed[int(seed)][int(T)] for seed in seeds]
        stats = _mean_ci95(vals, lower=0.0, upper=1.0)
        summary[int(T)] = {
            'mean': stats['mean'],
            'std': stats['std'],
            'ci_low': stats['ci_low'],
            'ci_high': stats['ci_high'],
            'seed_values': [float(v) for v in vals],
        }
        if verbose:
            print(f"  T={T:>4d}:  p(T)={stats['mean']:.3f}  "
                  f"CI95=[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]  "
                  f"(seeds={len(seeds)} x runs={n_runs_per_seed})")

    if enforce_monotone and summary:
        Ts = [int(T) for T in T_values]
        means = [summary[T]['mean'] for T in Ts]
        lows = [summary[T]['ci_low'] for T in Ts]
        highs = [summary[T]['ci_high'] for T in Ts]
        means_iso = _pava_non_decreasing(means)
        lows_iso = _pava_non_decreasing(lows)
        highs_iso = _pava_non_decreasing(highs)

        for i, T in enumerate(Ts):
            mean_v = float(np.clip(means_iso[i], 0.0, 1.0))
            low_v = float(np.clip(min(lows_iso[i], mean_v), 0.0, 1.0))
            high_v = float(np.clip(max(highs_iso[i], mean_v), 0.0, 1.0))
            summary[T]['mean'] = mean_v
            summary[T]['ci_low'] = low_v
            summary[T]['ci_high'] = high_v

    return {
        'T_values': [int(T) for T in T_values],
        'seeds': [int(s) for s in seeds],
        'n_runs_per_seed': int(n_runs_per_seed),
        'tau': float(tau),
        'audit_epsilon': float(audit_epsilon),
        'detection_metric': str(detection_metric),
        'min_consecutive_hits': int(min_consecutive_hits),
        'enforce_monotone': bool(enforce_monotone),
        'curves_by_seed': curves_by_seed,
        'summary': summary,
    }


def empirical_T_star_with_ci(game: CalvanoModel,
                             p_T_curves_by_seed: Dict[int, Dict[int, float]],
                             F_values: Optional[List[float]] = None,
                             pi_c: float = None,
                             pi_n: float = None,
                             t_search_max: Optional[int] = None,
                             extrapolation: str = 'hold_last',
                             verbose: bool = True) -> Dict:
    """
    Compute empirical T* distribution across seed-specific p(T) curves.

    Returns mean/std/95% CI of T* for each fine level F.
    """
    if F_values is None:
        F_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    from .t_star import compute_T_star_empirical
    from .utils import get_equilibrium_profits

    if pi_c is None or pi_n is None:
        _pi_c, _pi_n, _ = get_equilibrium_profits(game)
        pi_c = pi_c or _pi_c
        pi_n = pi_n or _pi_n

    seed_ids = sorted(int(s) for s in p_T_curves_by_seed.keys())
    if t_search_max is None:
        max_measured_T = max(
            max(int(T) for T in p_T_curves_by_seed[seed].keys())
            for seed in seed_ids
        )
        t_search_max = max(1000, max_measured_T + 200)

    tstar_by_F = {}
    for F in F_values:
        vals = []
        for seed in seed_ids:
            p_T_seed = p_T_curves_by_seed[seed]
            vals.append(float(compute_T_star_empirical(
                game, F, p_T_seed, pi_c=pi_c, pi_n=pi_n,
                T_max=int(t_search_max), extrapolation=extrapolation,
            )))

        stats = _mean_ci95(vals)
        tstar_by_F[float(F)] = {
            'mean': stats['mean'],
            'std': stats['std'],
            'ci_low': stats['ci_low'],
            'ci_high': stats['ci_high'],
            'seed_values': [float(v) for v in vals],
        }
        if verbose:
            print(f"  F={F:>4.2f}:  T*_emp={stats['mean']:.2f}  "
                  f"CI95=[{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]")

    return {
        'F_values': [float(F) for F in F_values],
        'seed_ids': seed_ids,
        't_search_max': int(t_search_max),
        'extrapolation': str(extrapolation),
        'summary': tstar_by_F,
    }
