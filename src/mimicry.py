"""
M3 -- Mimicry Agent: conditional pricing during audit, cost measurement.
"""

import numpy as np
from typing import Dict, List

from .model import CalvanoModel
from .utils import find_nash_action_index, find_greedy_equilibrium_state


def _clone_model_with_q(game: CalvanoModel) -> CalvanoModel:
    """Create an independent model copy with identical parameters and Q-table."""
    cloned = CalvanoModel(
        n=game.n,
        c=game.c_arr.copy(),
        a=game.a_arr.copy(),
        a0=game.a0,
        mu=game.mu,
        alpha=game.alpha,
        beta=game.beta,
        delta=game.delta,
        k=game.k,
        tstable=game.tstable,
        tmax=game.tmax,
    )
    cloned.Q = game.Q.copy()
    return cloned


def run_audit_with_mimicry(game: CalvanoModel, T_audit: int,
                            tau_threshold: float = 0.05,
                            n_runs: int = 100,
                            entrant_price: float = None) -> float:
    """
    Detection probability when all agents adopt perfect mimicry (play Nash).
    Result: 0.0 always -- Nash pricing gives zero gap above threshold.
    """
    nash_idx   = find_nash_action_index(game)
    detections = 0

    for _ in range(n_runs):
        detected = False
        for _ in range(T_audit):
            prices = game.A[np.full(game.n, nash_idx, dtype=int)]
            if np.max(prices) - np.mean(game.p_nash) > tau_threshold:
                detected = True
                break
        if detected:
            detections += 1

    return detections / n_runs


def compute_mimicry_cost_theoretical(pi_c: float, pi_n: float,
                                      delta: float, T: int) -> float:
    """C_mimic(T) = (pi_c - pi_n) * delta * (1 - delta^T) / (1 - delta)."""
    return (pi_c - pi_n) * delta * (1 - delta**T) / (1 - delta)


def compute_mimicry_cost_empirical(game: CalvanoModel, T: int,
                                    n_runs: int = 50) -> float:
    """
    Discounted profit gap between collusive play and Nash mimicry,
    both in the TWO-FIRM market (no entrant).
    """
    nash_idx = find_nash_action_index(game)
    eq_state = find_greedy_equilibrium_state(game)

    baseline_profits = []
    mimicry_profits  = []

    for _ in range(n_runs):
        # Baseline: greedy (collusive) play
        state      = eq_state.copy()
        total_base = 0.0
        for t in range(T):
            actions = np.array([
                np.argmax(game.Q[(n,) + tuple(state)])
                for n in range(game.n)
            ])
            total_base += (game.delta ** (t + 1)) * float(
                np.mean(game.PI[tuple(actions)])
            )
            state = actions.copy()
        baseline_profits.append(total_base)

        # Mimicry: Nash play
        nash_actions = np.full(game.n, nash_idx, dtype=int)
        pi_nash_period = float(np.mean(game.PI[tuple(nash_actions)]))
        total_mimic = pi_nash_period * (
            game.delta * (1 - game.delta**T) / (1 - game.delta)
        )
        mimicry_profits.append(total_mimic)

    return float(np.mean(baseline_profits) - np.mean(mimicry_profits))


def measure_mimicry_costs(game: CalvanoModel,
                           T_values: List[int] = None,
                           n_runs: int = 50,
                           pi_c: float = None,
                           pi_n: float = None,
                           verbose: bool = True) -> Dict:
    """Empirical and theoretical mimicry costs for a range of T."""
    if T_values is None:
        T_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    if pi_c is None:
        from .utils import get_equilibrium_profits
        pi_c, pi_n_auto, _ = get_equilibrium_profits(game)
        if pi_n is None:
            pi_n = pi_n_auto
    if pi_n is None:
        pi_n = game.pi_nash

    empirical_costs   = []
    theoretical_costs = []

    for T in T_values:
        emp  = compute_mimicry_cost_empirical(game, T, n_runs=n_runs)
        theo = compute_mimicry_cost_theoretical(pi_c, pi_n,
                                                 game.delta, T)
        empirical_costs.append(emp)
        theoretical_costs.append(theo)
        if verbose:
            ratio = f"  ratio={emp/theo:.3f}" if theo > 1e-12 else ""
            print(f"  T={T:>4d}:  C_emp={emp:.4f}  "
                  f"C_theo={theo:.4f}{ratio}")

    return {'T_values': T_values, 'empirical': empirical_costs,
            'theoretical': theoretical_costs,
            'pi_c': pi_c, 'pi_n': pi_n}


def run_audit_with_noisy_mimicry(game: CalvanoModel, T_audit: int,
                                  noise_level: float = 0.05,
                                  tau_threshold: float = 0.05,
                                  n_runs: int = 100) -> float:
    """
    Detection probability with imperfect mimicry.

    Instead of playing exact Nash, the mimicking firm plays
    Nash + Gaussian noise on the action index. This models
    real RL agents that cannot perfectly switch policies.

    noise_level: std of noise as fraction of k (0=perfect, 0.2=noisy).
    """
    nash_idx = find_nash_action_index(game)
    detections = 0

    for _ in range(n_runs):
        detected = False
        for _ in range(T_audit):
            noise = np.round(
                np.random.normal(0, noise_level * game.k, game.n)
            ).astype(int)
            noisy_actions = np.clip(nash_idx + noise, 0, game.k - 1)
            prices = game.A[noisy_actions]
            if np.max(prices) - np.mean(game.p_nash) > tau_threshold:
                detected = True
                break
        if detected:
            detections += 1

    return detections / n_runs


def noise_sweep(game: CalvanoModel,
                noise_levels: List[float] = None,
                T_audit: int = 20,
                tau_threshold: float = 0.05,
                n_runs: int = 100,
                verbose: bool = True) -> Dict:
    """
    Measure detection probability across noise levels.

    Returns dict with noise_levels and detection rates.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    detections = []
    for nl in noise_levels:
        p_det = run_audit_with_noisy_mimicry(
            game, T_audit, noise_level=nl,
            tau_threshold=tau_threshold, n_runs=n_runs)
        detections.append(p_det)
        if verbose:
            print(f"  noise={nl:.0%}  T={T_audit}  -> "
                  f"p(detection) = {p_det:.3f}")

    return {'noise_levels': noise_levels,
            'detection_rates': detections,
            'T_audit': T_audit,
            'tau_threshold': tau_threshold}


# ======================================================================
#  Theorem 3: RL Disruption Cost
# ======================================================================

def measure_rl_disruption(game: CalvanoModel, T_mimic: int,
                           T_recovery_max: int = 500,
                           convergence_threshold: float = 0.95,
                           n_runs: int = 10,
                           verbose: bool = True) -> Dict:
    """
    Theorem 3 test: measure Q-table disruption during mimicry.

    Protocol:
      1. Save converged Q-table.
      2. Force Nash play for T_mimic periods WHILE Q-learning updates.
         (The agent learns during mimicry, distorting its Q-values.)
      3. Resume greedy play and measure how many periods until
         prices recover to within `convergence_threshold` of the
         original collusive price.

    If disruption is real:
      - Recovery time > 0 (Q-table was damaged)
      - True RL mimicry cost > theoretical cost
      - -> T*(RL) < T*(rational)

    Returns:
      recovery_periods: mean periods to re-converge after mimicry
      disruption_cost: additional discounted profit lost during recovery
      total_rl_cost: theoretical_cost + disruption_cost
      ratio: total_rl_cost / theoretical_cost (>1 means disruption exists)
    """
    from .utils import get_equilibrium_profits
    nash_idx = find_nash_action_index(game)
    eq_state = find_greedy_equilibrium_state(game)

    pi_c, pi_n, eq_prices = get_equilibrium_profits(game)
    collusive_price = float(np.mean(eq_prices))
    recovery_threshold = np.mean(game.p_nash) + convergence_threshold * (
        collusive_price - np.mean(game.p_nash))

    Q_original = game.Q.copy()
    recovery_times = []
    disruption_costs = []

    for run in range(n_runs):
        # Restore original Q-table
        game.Q = Q_original.copy()
        state = eq_state.copy()

        # Phase 1: Force Nash play for T_mimic periods WITH Q-learning
        for t in range(T_mimic):
            nash_actions = np.full(game.n, nash_idx, dtype=int)
            profits = game.PI[tuple(nash_actions)]
            next_state = nash_actions.copy()

            # Q-learning update DURING mimicry (this disrupts Q-tables)
            for ni in range(game.n):
                idx = (ni,) + tuple(state) + (nash_actions[ni],)
                old_v = game.Q[idx]
                max_q = np.max(game.Q[(ni,) + tuple(next_state)])
                new_v = profits[ni] + game.delta * max_q
                game.Q[idx] = (1 - game.alpha) * old_v + game.alpha * new_v

            state = next_state

        # Phase 2: Resume greedy play, measure recovery
        recovery_t = T_recovery_max
        disruption_profit_loss = 0.0

        for t in range(T_recovery_max):
            actions = np.array([
                np.argmax(game.Q[(n,) + tuple(state)])
                for n in range(game.n)
            ])
            prices = game.A[actions]
            avg_price = float(np.mean(prices))

            # Track profit loss compared to ideal collusion
            actual_profit = float(np.mean(game.PI[tuple(actions)]))
            disruption_profit_loss += (game.delta ** (t + 1)) * (
                pi_c - actual_profit)

            # Q-learning update during recovery
            profits = game.PI[tuple(actions)]
            next_state = actions.copy()
            for ni in range(game.n):
                idx = (ni,) + tuple(state) + (actions[ni],)
                old_v = game.Q[idx]
                max_q = np.max(game.Q[(ni,) + tuple(next_state)])
                new_v = profits[ni] + game.delta * max_q
                game.Q[idx] = (1 - game.alpha) * old_v + game.alpha * new_v

            state = next_state

            if avg_price >= recovery_threshold:
                recovery_t = t + 1
                break

        recovery_times.append(recovery_t)
        disruption_costs.append(disruption_profit_loss)

    # Restore original Q-table
    game.Q = Q_original.copy()

    # Theoretical cost (no disruption)
    C_theo = compute_mimicry_cost_theoretical(pi_c, pi_n, game.delta, T_mimic)

    mean_recovery = float(np.mean(recovery_times))
    mean_disruption = float(np.mean(disruption_costs))
    total_rl_cost = C_theo + mean_disruption
    ratio = total_rl_cost / C_theo if C_theo > 1e-12 else 1.0

    if verbose:
        print(f"\n  --- RL Disruption (Theorem 3) for T_mimic={T_mimic} ---")
        print(f"  C_mimic(theoretical)  = {C_theo:.4f}")
        print(f"  Disruption cost       = {mean_disruption:.4f}")
        print(f"  Total RL cost         = {total_rl_cost:.4f}")
        print(f"  Ratio (RL/rational)   = {ratio:.3f}")
        print(f"  Recovery time         = {mean_recovery:.1f} periods "
              f"(+/- {np.std(recovery_times):.1f})")
        if ratio > 1.01:
            print(f"  -> Theorem 3 CONFIRMED: RL mimicry costs {ratio:.1f}x more")
        else:
            print(f"  -> Theorem 3 NOT confirmed at T={T_mimic} "
                  f"(disruption negligible)")

    return {
        'T_mimic': T_mimic,
        'C_theoretical': C_theo,
        'disruption_cost': mean_disruption,
        'total_rl_cost': total_rl_cost,
        'ratio': ratio,
        'mean_recovery_time': mean_recovery,
        'std_recovery_time': float(np.std(recovery_times)),
        'recovery_times': recovery_times,
    }


def theorem3_sweep(game: CalvanoModel,
                    T_values: List[int] = None,
                    n_runs: int = 10,
                    verbose: bool = True) -> Dict:
    """
    Run RL disruption analysis for multiple T values.
    Tests Theorem 3: T*(RL) <= T*(rational).
    """
    if T_values is None:
        T_values = [1, 5, 10, 20, 50, 100]

    results = []
    for T in T_values:
        r = measure_rl_disruption(game, T, n_runs=n_runs, verbose=verbose)
        results.append(r)

    # Compute T*(RL) and T*(rational) for a range of F values
    from .utils import get_equilibrium_profits
    from .t_star import compute_T_star_analytical, DEFAULT_TAU
    pi_c, pi_n, _ = get_equilibrium_profits(game)

    if verbose:
        print(f"\n  {'='*70}")
        print(f"  T*(RL) vs T*(rational) Comparison")
        print(f"  {'='*70}")
        print(f"  {'F':>8}  {'T*_rational':>12}  {'T*_RL':>8}  "
              f"{'Ratio':>8}  {'Theorem 3':>12}")
        print(f"  {'-'*70}")

    F_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    t_star_comparisons = []

    for F in F_values:
        # T*(rational): standard analytical
        T_rational = compute_T_star_analytical(
            pi_c, pi_n, game.delta, F, tau_scale=DEFAULT_TAU)

        # T*(RL): find T where total_rl_cost >= p(T)*F
        T_rl = T_rational  # default to same
        for T in range(1, 1001):
            # Interpolate disruption ratio from measured values
            ratio_at_T = 1.0
            for r in results:
                if r['T_mimic'] <= T:
                    ratio_at_T = max(ratio_at_T, r['ratio'])
            C_rl = (pi_c - pi_n) * game.delta * (
                1 - game.delta**T) / (1 - game.delta) * ratio_at_T
            from .t_star import p_T_parametric
            pT = p_T_parametric(T, DEFAULT_TAU)
            if C_rl >= pT * F:
                T_rl = T
                break

        t3_result = "CONFIRMED" if T_rl < T_rational else (
            "EQUAL" if T_rl == T_rational else "NOT CONFIRMED")
        t_star_comparisons.append({
            'F': F, 'T_rational': T_rational, 'T_rl': T_rl,
            'ratio': T_rl / T_rational if T_rational > 0 else 1.0,
            'theorem3': t3_result,
        })

        if verbose:
            print(f"  {F:>8.2f}  {T_rational:>12}  {T_rl:>8}  "
                  f"{T_rl/T_rational if T_rational > 0 else 1.0:>8.2f}  "
                  f"{t3_result:>12}")

    if verbose:
        print(f"  {'='*70}")

    return {
        'disruption_results': results,
        'T_star_comparisons': t_star_comparisons,
    }


def measure_rl_disruption_sarsa(game: CalvanoModel, T_mimic: int,
                                 T_recovery_max: int = 500,
                                 convergence_threshold: float = 0.95,
                                 n_runs: int = 10,
                                 verbose: bool = True) -> Dict:
    """
    Same as measure_rl_disruption() but uses SARSA (on-policy) updates
    during both mimicry and recovery phases.

    SARSA should show MORE disruption than Q-learning because on-policy
    updates during Nash play directly overwrite collusive Q-values.
    """
    from .utils import get_equilibrium_profits
    from .simulation import update_sarsa, pick_strategies
    nash_idx = find_nash_action_index(game)
    eq_state = find_greedy_equilibrium_state(game)

    pi_c, pi_n, eq_prices = get_equilibrium_profits(game)
    collusive_price = float(np.mean(eq_prices))
    recovery_threshold = np.mean(game.p_nash) + convergence_threshold * (
        collusive_price - np.mean(game.p_nash))

    Q_original = game.Q.copy()
    recovery_times = []
    disruption_costs = []

    for run in range(n_runs):
        game.Q = Q_original.copy()
        state = eq_state.copy()

        # Phase 1: Nash mimicry with SARSA updates
        actions = np.full(game.n, nash_idx, dtype=int)
        for t in range(T_mimic):
            profits = game.PI[tuple(actions)]
            next_state = actions.copy()
            next_actions = np.full(game.n, nash_idx, dtype=int)
            # SARSA update
            for ni in range(game.n):
                idx = (ni,) + tuple(state) + (actions[ni],)
                old_v = game.Q[idx]
                next_idx = (ni,) + tuple(next_state) + (next_actions[ni],)
                next_q = game.Q[next_idx]
                new_v = profits[ni] + game.delta * next_q
                game.Q[idx] = (1 - game.alpha) * old_v + game.alpha * new_v
            state = next_state
            actions = next_actions

        # Phase 2: Recovery with SARSA
        recovery_t = T_recovery_max
        disruption_profit_loss = 0.0
        actions = np.array([
            np.argmax(game.Q[(n,) + tuple(state)])
            for n in range(game.n)
        ])

        for t in range(T_recovery_max):
            prices = game.A[actions]
            avg_price = float(np.mean(prices))
            actual_profit = float(np.mean(game.PI[tuple(actions)]))
            disruption_profit_loss += (game.delta ** (t + 1)) * (
                pi_c - actual_profit)

            profits = game.PI[tuple(actions)]
            next_state = actions.copy()
            next_actions = np.array([
                np.argmax(game.Q[(n,) + tuple(next_state)])
                for n in range(game.n)
            ])
            # SARSA update
            for ni in range(game.n):
                idx = (ni,) + tuple(state) + (actions[ni],)
                old_v = game.Q[idx]
                next_idx = (ni,) + tuple(next_state) + (next_actions[ni],)
                next_q = game.Q[next_idx]
                new_v = profits[ni] + game.delta * next_q
                game.Q[idx] = (1 - game.alpha) * old_v + game.alpha * new_v

            state = next_state
            actions = next_actions

            if avg_price >= recovery_threshold:
                recovery_t = t + 1
                break

        recovery_times.append(recovery_t)
        disruption_costs.append(disruption_profit_loss)

    game.Q = Q_original.copy()
    C_theo = compute_mimicry_cost_theoretical(pi_c, pi_n, game.delta, T_mimic)

    mean_recovery = float(np.mean(recovery_times))
    mean_disruption = float(np.mean(disruption_costs))
    total_cost = C_theo + mean_disruption
    ratio = total_cost / C_theo if C_theo > 1e-12 else 1.0

    if verbose:
        print(f"\n  --- SARSA Disruption for T_mimic={T_mimic} ---")
        print(f"  C_mimic(theoretical)  = {C_theo:.4f}")
        print(f"  Disruption cost       = {mean_disruption:.4f}")
        print(f"  Total SARSA cost      = {total_cost:.4f}")
        print(f"  Ratio (SARSA/rational)= {ratio:.3f}")
        print(f"  Recovery time         = {mean_recovery:.1f} periods "
              f"(+/- {np.std(recovery_times):.1f})")

    return {
        'T_mimic': T_mimic, 'algorithm': 'SARSA',
        'C_theoretical': C_theo, 'disruption_cost': mean_disruption,
        'total_rl_cost': total_cost, 'ratio': ratio,
        'mean_recovery_time': mean_recovery,
        'std_recovery_time': float(np.std(recovery_times)),
    }


def theorem3_comparison(game_q: CalvanoModel,
                        game_sarsa: CalvanoModel = None,
                        T_values: List[int] = None,
                        n_runs: int = 10,
                        verbose: bool = True) -> Dict:
    """
    Compare Theorem 3 disruption: Q-learning vs SARSA.
    """
    if T_values is None:
        T_values = [1, 5, 10, 20, 50]

    if game_sarsa is None:
        game_sarsa = _clone_model_with_q(game_q)

    q_results = []
    sarsa_results = []
    for T in T_values:
        qr = measure_rl_disruption(game_q, T, n_runs=n_runs, verbose=False)
        sr = measure_rl_disruption_sarsa(game_sarsa, T, n_runs=n_runs,
                                         verbose=False)
        q_results.append(qr)
        sarsa_results.append(sr)

    if verbose:
        print(f"\n  {'='*72}")
        print(f"  Theorem 3: Q-Learning vs SARSA Disruption")
        print(f"  {'='*72}")
        print(f"  {'T':>5}  {'C_theo':>8}  {'Q-Disrupt':>10}  "
              f"{'Q-Ratio':>8}  {'S-Disrupt':>10}  {'S-Ratio':>8}  "
              f"{'Q-Recov':>8}  {'S-Recov':>8}")
        print(f"  {'-'*72}")
        for i, T in enumerate(T_values):
            qr, sr = q_results[i], sarsa_results[i]
            print(f"  {T:>5}  {qr['C_theoretical']:>8.4f}  "
                  f"{qr['disruption_cost']:>10.4f}  "
                  f"{qr['ratio']:>8.3f}  "
                  f"{sr['disruption_cost']:>10.4f}  "
                  f"{sr['ratio']:>8.3f}  "
                  f"{qr['mean_recovery_time']:>8.1f}  "
                  f"{sr['mean_recovery_time']:>8.1f}")
        print(f"  {'='*72}")

    return {
        'T_values': T_values,
        'q_learning': q_results,
        'sarsa': sarsa_results,
    }
