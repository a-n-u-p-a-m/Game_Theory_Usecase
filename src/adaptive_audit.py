"""
M6 -- Adaptive Audit Mechanism.

Instead of a fixed audit duration T, the regulator observes prices
sequentially and uses Bayesian updating to decide when to stop.

The regulator maintains a posterior P(collusive | prices seen so far)
and stops when confidence exceeds a threshold.
"""

import numpy as np
from typing import Dict, List, Optional

from .model import CalvanoModel
from .utils import find_nash_action_index, find_greedy_equilibrium_state


def _wilson_ci(successes: int, trials: int,
               z: float = 1.96) -> Dict[str, float]:
    """Wilson score interval for Bernoulli proportion."""
    n = int(max(0, trials))
    k = int(max(0, min(successes, n)))
    if n == 0:
        return {'low': 0.0, 'high': 1.0}

    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    spread = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z ** 2) / (4.0 * n ** 2))
    return {
        'low': float(max(0.0, center - spread)),
        'high': float(min(1.0, center + spread)),
    }


def estimate_signal_likelihoods(game: CalvanoModel,
                                tau_threshold: float = 0.05,
                                n_samples: int = 300,
                                audit_epsilon: float = 0.05,
                                competitive_noise: float = 0.01,
                                min_separation: float = 0.02) -> Dict[str, float]:
    """
    Estimate one-step signal likelihoods from the trained policy.

        Returns calibrated likelihoods and 95% Wilson intervals:
            p_c = P(signal | collusive policy)
            p_n = P(signal | competitive/Nash policy)
    """
    eq_state = find_greedy_equilibrium_state(game)
    nash_idx = find_nash_action_index(game)
    nash_mean = np.mean(game.p_nash)

    collusive_signals = 0
    competitive_signals = 0

    for _ in range(n_samples):
        perturb = np.random.randint(-2, 3, size=game.n)
        state = np.clip(eq_state + perturb, 0, game.k - 1)
        collusive_actions = np.array([
            np.random.randint(0, game.k)
            if (audit_epsilon > 0 and np.random.rand() < audit_epsilon)
            else np.argmax(game.Q[(n,) + tuple(state)])
            for n in range(game.n)
        ])
        collusive_prices = game.A[collusive_actions]
        collusive_signals += int(
            np.max(collusive_prices) - nash_mean > tau_threshold
        )

        noise = np.round(
            np.random.normal(0, competitive_noise * game.k, game.n)
        ).astype(int)
        competitive_actions = np.clip(nash_idx + noise, 0, game.k - 1)
        competitive_prices = game.A[competitive_actions]
        competitive_signals += int(
            np.max(competitive_prices) - nash_mean > tau_threshold
        )

    p_c_raw = float(np.clip(collusive_signals / n_samples, 1e-4, 0.9999))
    p_n_raw = float(np.clip(competitive_signals / n_samples, 1e-4, 0.9999))

    # Operational calibration keeps collusive likelihood above competitive one.
    p_n = p_n_raw
    p_c = max(p_c_raw, p_n_raw + float(min_separation))
    p_c = float(np.clip(p_c, 1e-4, 0.9999))

    ci_c = _wilson_ci(collusive_signals, n_samples)
    ci_n = _wilson_ci(competitive_signals, n_samples)

    return {
        'p_c': p_c,
        'p_n': p_n,
        'p_c_raw': p_c_raw,
        'p_n_raw': p_n_raw,
        'p_c_ci_low': ci_c['low'],
        'p_c_ci_high': ci_c['high'],
        'p_n_ci_low': ci_n['low'],
        'p_n_ci_high': ci_n['high'],
        'collusive_signal_hits': int(collusive_signals),
        'competitive_signal_hits': int(competitive_signals),
        'n_samples': int(n_samples),
        'tau_threshold': float(tau_threshold),
        'audit_epsilon': float(audit_epsilon),
        'competitive_noise': float(competitive_noise),
        'min_separation': float(min_separation),
    }


def bayesian_audit(game: CalvanoModel,
                   tau_threshold: float = 0.05,
                   confidence_threshold: float = 0.95,
                   max_T: int = 500,
                   prior_collusive: float = 0.5,
                   entrant_price: float = None,
                   likelihoods: Optional[Dict[str, float]] = None,
                   audit_epsilon: float = 0.05) -> Dict:
    """
    Adaptive Bayesian audit.

    The regulator observes one period at a time.  After each period,
    she updates the posterior probability that the firm is collusive:

        P(C | data) = P(data | C) * P(C) / P(data)

    Likelihood model:
        P(price > p_nash + tau | C) = p_c  (high, ~0.8-0.95)
        P(price > p_nash + tau | N) = p_n  (low,  ~0.01-0.05)

    Stops when P(C | data) >= confidence_threshold  (declare collusive)
    or P(C | data) <= 1 - confidence_threshold      (declare competitive)
    or t >= max_T                                    (inconclusive).

    Returns dict with decision, posterior, stopping_time.
    """
    if entrant_price is None:
        entrant_price = np.mean(game.c_arr)

    nash_mean = np.mean(game.p_nash)

    if likelihoods is None:
        likelihoods = estimate_signal_likelihoods(
            game,
            tau_threshold=tau_threshold,
            n_samples=300,
            audit_epsilon=audit_epsilon,
        )
    p_c = likelihoods['p_c']
    p_n = likelihoods['p_n']

    eq_state = find_greedy_equilibrium_state(game)
    state = eq_state.copy()
    posterior = prior_collusive
    price_trace = []

    for t in range(1, max_T + 1):
        # Greedy actions
        actions = np.array([
            np.random.randint(0, game.k)
            if (audit_epsilon > 0 and np.random.rand() < audit_epsilon)
            else np.argmax(game.Q[(n,) + tuple(state)])
            for n in range(game.n)
        ])
        prices = game.A[actions]
        price_trace.append(prices.copy())

        # Observation: is any price above Nash + tau?
        signal = int(np.max(prices) - nash_mean > tau_threshold)

        # Bayesian update
        if signal:
            likelihood_C = p_c
            likelihood_N = p_n
        else:
            likelihood_C = 1 - p_c
            likelihood_N = 1 - p_n

        numerator = likelihood_C * posterior
        denominator = numerator + likelihood_N * (1 - posterior)
        posterior = numerator / denominator if denominator > 1e-15 else 0.5

        # Stopping rule
        if posterior >= confidence_threshold:
            return {
                'decision': 'COLLUSIVE',
                'posterior': posterior,
                'stopping_time': t,
                'prices': np.array(price_trace),
                'confident': True,
            }
        elif posterior <= 1.0 - confidence_threshold:
            return {
                'decision': 'COMPETITIVE',
                'posterior': posterior,
                'stopping_time': t,
                'prices': np.array(price_trace),
                'confident': True,
            }

        state = actions.copy()

    return {
        'decision': 'INCONCLUSIVE',
        'posterior': posterior,
        'stopping_time': max_T,
        'prices': np.array(price_trace),
        'confident': False,
    }


def bayesian_audit_with_mimicry(game: CalvanoModel,
                                 tau_threshold: float = 0.05,
                                 confidence_threshold: float = 0.95,
                                 max_T: int = 500,
                                 prior_collusive: float = 0.5,
                                 noise_level: float = 0.0,
                                 likelihoods: Optional[Dict[str, float]] = None) -> Dict:
    """
    Adaptive Bayesian audit against a mimicking (Nash-playing) firm.

    If noise_level > 0, the mimicry is imperfect (noisy Nash actions).
    """
    nash_idx = find_nash_action_index(game)
    nash_mean = np.mean(game.p_nash)

    if likelihoods is None:
        likelihoods = estimate_signal_likelihoods(
            game,
            tau_threshold=tau_threshold,
            n_samples=300,
            audit_epsilon=0.05,
        )
    p_c = likelihoods['p_c']
    p_n = likelihoods['p_n']

    posterior = prior_collusive
    price_trace = []

    for t in range(1, max_T + 1):
        # Mimicry: play Nash + noise
        noise = np.round(
            np.random.normal(0, noise_level * game.k, game.n)
        ).astype(int)
        actions = np.clip(nash_idx + noise, 0, game.k - 1)
        prices = game.A[actions]
        price_trace.append(prices.copy())

        signal = int(np.max(prices) - nash_mean > tau_threshold)

        if signal:
            likelihood_C = p_c
            likelihood_N = p_n
        else:
            likelihood_C = 1 - p_c
            likelihood_N = 1 - p_n

        numerator = likelihood_C * posterior
        denominator = numerator + likelihood_N * (1 - posterior)
        posterior = numerator / denominator if denominator > 1e-15 else 0.5

        if posterior >= confidence_threshold:
            return {
                'decision': 'COLLUSIVE',
                'posterior': posterior,
                'stopping_time': t,
                'prices': np.array(price_trace),
                'confident': True,
                'mimicry': True,
                'noise_level': noise_level,
            }
        elif posterior <= 1.0 - confidence_threshold:
            return {
                'decision': 'COMPETITIVE',
                'posterior': posterior,
                'stopping_time': t,
                'prices': np.array(price_trace),
                'confident': True,
                'mimicry': True,
                'noise_level': noise_level,
            }

    return {
        'decision': 'INCONCLUSIVE',
        'posterior': posterior,
        'stopping_time': max_T,
        'prices': np.array(price_trace),
        'confident': False,
        'mimicry': True,
        'noise_level': noise_level,
    }


def compare_adaptive_vs_fixed(game: CalvanoModel,
                               fixed_T_values: List[int] = None,
                               confidence_threshold: float = 0.95,
                               tau_threshold: float = 0.05,
                               audit_epsilon: float = 0.05,
                               n_runs: int = 50,
                               verbose: bool = True) -> Dict:
    """
    Compare adaptive (Bayesian) audit vs fixed-T audit.

    For each run:
      - Adaptive audit reports stopping_time and decision.
      - Fixed audits at each T report detection yes/no.

    Returns comparison of average stopping time and accuracy.
    """
    if fixed_T_values is None:
        fixed_T_values = [1, 5, 10, 20, 50, 100]

    likelihoods = estimate_signal_likelihoods(
        game,
        tau_threshold=tau_threshold,
        n_samples=max(300, n_runs * 5),
        audit_epsilon=audit_epsilon,
    )

    # Adaptive audit
    adaptive_times = []
    adaptive_correct = 0

    for _ in range(n_runs):
        result = bayesian_audit(
            game,
            tau_threshold=tau_threshold,
            confidence_threshold=confidence_threshold,
            likelihoods=likelihoods,
            audit_epsilon=audit_epsilon,
        )
        adaptive_times.append(result['stopping_time'])
        if result['decision'] == 'COLLUSIVE':
            adaptive_correct += 1

    adaptive_acc = adaptive_correct / n_runs
    adaptive_mean_T = np.mean(adaptive_times)
    adaptive_std_T = np.std(adaptive_times)

    # Adaptive with mimicry
    adaptive_mimic_decisions = []
    for _ in range(n_runs):
        result = bayesian_audit_with_mimicry(
            game,
            tau_threshold=tau_threshold,
            confidence_threshold=confidence_threshold,
            noise_level=0.0,
            likelihoods=likelihoods,
        )
        adaptive_mimic_decisions.append(result['decision'])

    mimic_detected = sum(
        1 for d in adaptive_mimic_decisions if d == 'COLLUSIVE') / n_runs
    acc_ci = _wilson_ci(adaptive_correct, n_runs)
    mimic_ci = _wilson_ci(
        sum(1 for d in adaptive_mimic_decisions if d == 'COLLUSIVE'),
        n_runs,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("  ADAPTIVE vs FIXED AUDIT COMPARISON")
        print("=" * 60)
        print(f"\n  Adaptive (Bayesian, conf={confidence_threshold}):")
        print(f"    Calibrated likelihoods          : "
              f"p_c={likelihoods['p_c']:.3f} "
              f"[{likelihoods['p_c_ci_low']:.3f},{likelihoods['p_c_ci_high']:.3f}], "
              f"p_n={likelihoods['p_n']:.3f} "
              f"[{likelihoods['p_n_ci_low']:.3f},{likelihoods['p_n_ci_high']:.3f}]")
        print(f"    Accuracy (collusion detected)  : {adaptive_acc:.1%} "
              f"(CI95 [{acc_ci['low']:.1%}, {acc_ci['high']:.1%}])")
        print(f"    Mean stopping time             : {adaptive_mean_T:.1f} "
              f"+/- {adaptive_std_T:.1f}")
        print(f"    Detection under perfect mimicry: {mimic_detected:.1%} "
              f"(CI95 [{mimic_ci['low']:.1%}, {mimic_ci['high']:.1%}])")

    return {
        'adaptive_accuracy': adaptive_acc,
        'adaptive_accuracy_ci_low': acc_ci['low'],
        'adaptive_accuracy_ci_high': acc_ci['high'],
        'adaptive_mean_T': adaptive_mean_T,
        'adaptive_std_T': adaptive_std_T,
        'adaptive_mimicry_detection': mimic_detected,
        'adaptive_mimicry_ci_low': mimic_ci['low'],
        'adaptive_mimicry_ci_high': mimic_ci['high'],
        'likelihoods': likelihoods,
        'tau_threshold': float(tau_threshold),
        'audit_epsilon': float(audit_epsilon),
        'fixed_T_values': fixed_T_values,
        'n_runs': n_runs,
    }
