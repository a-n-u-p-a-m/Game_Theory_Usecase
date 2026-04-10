"""
Q-learning simulation engine, impulse-response, and multi-seed experiments.
Generalised for n firms (FIX B3, B4).
"""

import numpy as np
import sys
import time
from typing import Dict, List

from .model import CalvanoModel


# -------------------------------------------------------------------
#  Action selection
# -------------------------------------------------------------------

def pick_strategies(game: CalvanoModel, state: np.ndarray,
                    t: int) -> np.ndarray:
    """Epsilon-greedy with exponential decay: eps(t) = exp(-beta*t)."""
    actions = np.zeros(game.n, dtype=int)
    pr_explore = np.exp(-t * game.beta)
    explore_flags = pr_explore > np.random.rand(game.n)
    for n_idx in range(game.n):
        if explore_flags[n_idx]:
            actions[n_idx] = np.random.randint(0, game.k)
        else:
            actions[n_idx] = np.argmax(
                game.Q[(n_idx,) + tuple(state)]
            )
    return actions


# -------------------------------------------------------------------
#  Q-learning update
# -------------------------------------------------------------------

def update_q(game: CalvanoModel, state: np.ndarray,
             actions: np.ndarray, next_state: np.ndarray,
             profits: np.ndarray, stable: int):
    """
    Standard Q-learning update + stability tracking.
    Returns (stable, changed).
    """
    changed = False
    for n_idx in range(game.n):
        idx = (n_idx,) + tuple(state) + (actions[n_idx],)
        old_val = game.Q[idx]
        max_q   = np.max(game.Q[(n_idx,) + tuple(next_state)])
        new_val = profits[n_idx] + game.delta * max_q

        old_argmax = np.argmax(game.Q[(n_idx,) + tuple(state)])
        game.Q[idx] = (1 - game.alpha) * old_val + game.alpha * new_val
        new_argmax = np.argmax(game.Q[(n_idx,) + tuple(state)])

        same = int(old_argmax == new_argmax)
        stable = (stable + same) * same
        if not same:
            changed = True
    return stable, changed


# -------------------------------------------------------------------
#  Simulation
# -------------------------------------------------------------------

def simulate_game(game: CalvanoModel, seed: int = 42,
                  verbose: bool = True) -> Dict:
    """
    Run one full Q-learning simulation until convergence or tmax.
    Works for arbitrary n.
    """
    np.random.seed(seed)
    state = np.zeros(game.n, dtype=int)
    stable = 0
    converged = False
    convergence_t = game.tmax

    price_history = []
    profit_index_history = []
    t_start = time.time()

    for t in range(game.tmax):
        actions = pick_strategies(game, state, t)
        profits = game.PI[tuple(actions)]
        next_state = actions.copy()
        stable, _ = update_q(game, state, actions, next_state,
                             profits, stable)
        state = next_state

        if t % 10_000 == 0:
            greedy_prices = game.A[np.array([
                np.argmax(game.Q[(n,) + tuple(state)])
                for n in range(game.n)
            ])]
            price_history.append(greedy_prices.copy())
            profit_index_history.append(
                game.profit_index(greedy_prices))

            if verbose and t % 100_000 == 0:
                elapsed = time.time() - t_start
                eps_now = np.exp(-t * game.beta)
                p_str = ",".join(f"{p:.3f}" for p in greedy_prices)
                sys.stdout.write(
                    f"\r  t={t:>10,}  eps={eps_now:.4f}  "
                    f"p=[{p_str}]  "
                    f"Delta={profit_index_history[-1]:.3f}  "
                    f"stable={stable:>8,}  [{elapsed:.0f}s]"
                )
                sys.stdout.flush()

        if stable > game.tstable:
            converged = True
            convergence_t = t
            if verbose:
                print(f"\n  [OK] Converged at t = {t:,}")
            break

    if not converged and verbose:
        print(f"\n  [!!] Did NOT converge after {game.tmax:,} periods")

    final_prices = game.A[np.array([
        np.argmax(game.Q[(n,) + tuple(state)])
        for n in range(game.n)
    ])]

    return {
        'converged': converged,
        'convergence_t': convergence_t,
        'price_history': price_history,
        'profit_index_history': profit_index_history,
        'final_Q': game.Q.copy(),
        'final_prices': final_prices,
        'profit_index': game.profit_index(final_prices),
        'seed': seed,
    }


# -------------------------------------------------------------------
#  Impulse response
# -------------------------------------------------------------------

def impulse_response(game: CalvanoModel, Q_converged: np.ndarray,
                     state0: np.ndarray = None,
                     deviating_firm: int = 0,
                     deviation_action: int = 0,
                     T_before: int = 5, T_after: int = 50) -> Dict:
    """
    Force one firm to deviate for one period, then resume greedy play.
    """
    game.Q = Q_converged.copy()
    if state0 is None:
        state0 = np.zeros(game.n, dtype=int)
        for _ in range(100):
            acts = np.array([np.argmax(game.Q[(n,) + tuple(state0)])
                             for n in range(game.n)])
            if np.array_equal(acts, state0):
                break
            state0 = acts

    T_total = T_before + 1 + T_after
    prices = np.zeros((T_total, game.n))
    state = state0.copy()

    for t in range(T_total):
        actions = np.array([np.argmax(game.Q[(n,) + tuple(state)])
                            for n in range(game.n)])
        if t == T_before:
            actions[deviating_firm] = deviation_action
        prices[t] = game.A[actions]
        state = actions.copy()

    return {'prices': prices, 'deviation_t': T_before,
            'T_before': T_before, 'T_after': T_after}


# -------------------------------------------------------------------
#  Multi-seed & parameter sweep
# -------------------------------------------------------------------

def run_multi_seed(n_seeds: int = 5, verbose: bool = True,
                   **model_kwargs) -> List[Dict]:
    results = []
    for i in range(n_seeds):
        seed = 42 + i * 7
        if verbose:
            print(f"\n{'-'*60}")
            print(f"  Seed {i+1}/{n_seeds}  (seed={seed})")
            print(f"{'-'*60}")
        game = CalvanoModel(**model_kwargs)
        result = simulate_game(game, seed=seed, verbose=verbose)
        result['game'] = game
        results.append(result)
    return results


def parameter_sweep(param_name: str, param_values: List[float],
                    n_seeds: int = 3, verbose: bool = False,
                    **base_kwargs) -> Dict:
    mean_deltas, std_deltas, all_deltas = [], [], []
    for val in param_values:
        kwargs = {**base_kwargs, param_name: val}
        results = run_multi_seed(n_seeds=n_seeds, verbose=verbose,
                                 **kwargs)
        deltas = [r['profit_index'] for r in results]
        mean_deltas.append(np.mean(deltas))
        std_deltas.append(np.std(deltas))
        all_deltas.append(deltas)
        print(f"  {param_name}={val:.4f}  ->  "
              f"Delta = {np.mean(deltas):.3f} +/- {np.std(deltas):.3f}")
    return {'param_name': param_name, 'param_values': param_values,
            'mean_delta': mean_deltas, 'std_delta': std_deltas,
            'all_delta': all_deltas}


def print_results_table(results: List[Dict], game: CalvanoModel):
    """Print summary table -- works for arbitrary n. (FIX B4)"""
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    hdr = f"  {'Seed':>6}  {'Conv?':>6}  {'Conv. Period':>14}  "
    for i in range(game.n):
        hdr += f"{'p_'+str(i+1):>8}  "
    hdr += f"{'Delta':>8}"
    print(hdr)
    print("-" * 72)
    for r in results:
        row = f"  {r['seed']:>6}  "
        row += f"{'YES' if r['converged'] else 'NO':>6}  "
        row += (f"{r['convergence_t']:>14,}" if r['converged']
                else f"{'N/A':>14}") + "  "
        for i in range(game.n):
            row += f"{r['final_prices'][i]:>8.4f}  "
        row += f"{r['profit_index']:>8.3f}"
        print(row)
    print("-" * 72)
    deltas = [r['profit_index'] for r in results]
    print(f"  Mean Delta = {np.mean(deltas):.4f}  |  "
          f"Std = {np.std(deltas):.4f}  |  "
          f"Converged: {sum(r['converged'] for r in results)}"
          f"/{len(results)}")
    print("=" * 72)


# -------------------------------------------------------------------
#  SARSA update (alternative to Q-learning)
# -------------------------------------------------------------------

def update_sarsa(game: CalvanoModel, state: np.ndarray,
                 actions: np.ndarray, next_state: np.ndarray,
                 next_actions: np.ndarray,
                 profits: np.ndarray, stable: int):
    """
    SARSA update: uses the actual next action instead of max.
    TD target = r + delta * Q(s', a'_actual) instead of max_a' Q(s', a').

    Returns (stable, changed).
    """
    changed = False
    for n_idx in range(game.n):
        idx = (n_idx,) + tuple(state) + (actions[n_idx],)
        old_val = game.Q[idx]
        next_idx = (n_idx,) + tuple(next_state) + (next_actions[n_idx],)
        next_q = game.Q[next_idx]
        new_val = profits[n_idx] + game.delta * next_q

        old_argmax = np.argmax(game.Q[(n_idx,) + tuple(state)])
        game.Q[idx] = (1 - game.alpha) * old_val + game.alpha * new_val
        new_argmax = np.argmax(game.Q[(n_idx,) + tuple(state)])

        same = int(old_argmax == new_argmax)
        stable = (stable + same) * same
        if not same:
            changed = True
    return stable, changed


def simulate_game_sarsa(game: CalvanoModel, seed: int = 42,
                         verbose: bool = True) -> Dict:
    """
    SARSA simulation -- on-policy alternative to Q-learning.
    Uses the ACTUAL next action in the TD target, not the greedy max.
    """
    np.random.seed(seed)
    state = np.zeros(game.n, dtype=int)
    actions = pick_strategies(game, state, 0)
    stable = 0
    converged = False
    convergence_t = game.tmax

    price_history = []
    profit_index_history = []
    t_start = time.time()

    for t in range(game.tmax):
        profits = game.PI[tuple(actions)]
        next_state = actions.copy()
        next_actions = pick_strategies(game, next_state, t + 1)

        stable, _ = update_sarsa(game, state, actions, next_state,
                                  next_actions, profits, stable)
        state = next_state
        actions = next_actions

        if t % 10_000 == 0:
            greedy_prices = game.A[np.array([
                np.argmax(game.Q[(n,) + tuple(state)])
                for n in range(game.n)
            ])]
            price_history.append(greedy_prices.copy())
            profit_index_history.append(
                game.profit_index(greedy_prices))

            if verbose and t % 100_000 == 0:
                elapsed = time.time() - t_start
                eps_now = np.exp(-t * game.beta)
                p_str = ",".join(f"{p:.3f}" for p in greedy_prices)
                sys.stdout.write(
                    f"\r  [SARSA] t={t:>10,}  eps={eps_now:.4f}  "
                    f"p=[{p_str}]  "
                    f"Delta={profit_index_history[-1]:.3f}  "
                    f"stable={stable:>8,}  [{elapsed:.0f}s]"
                )
                sys.stdout.flush()

        if stable > game.tstable:
            converged = True
            convergence_t = t
            if verbose:
                print(f"\n  [SARSA OK] Converged at t = {t:,}")
            break

    if not converged and verbose:
        print(f"\n  [SARSA !!] Did NOT converge after "
              f"{game.tmax:,} periods")

    final_prices = game.A[np.array([
        np.argmax(game.Q[(n,) + tuple(state)])
        for n in range(game.n)
    ])]

    return {
        'converged': converged,
        'convergence_t': convergence_t,
        'price_history': price_history,
        'profit_index_history': profit_index_history,
        'final_Q': game.Q.copy(),
        'final_prices': final_prices,
        'profit_index': game.profit_index(final_prices),
        'seed': seed,
        'algorithm': 'SARSA',
    }


# -------------------------------------------------------------------
#  N-agent comparison
# -------------------------------------------------------------------

def compare_n_agents(n_values: List[int] = None,
                      seeds: int = 1,
                      tmax: int = 5_000_000,
                      verbose: bool = True) -> List[Dict]:
    """
    Run training for different n, compare Delta and F_min.
    Automatically reduces k for n>=3 to keep Q-table manageable.
    """
    if n_values is None:
        n_values = [2, 3]

    from .policy import compute_F_min
    from .utils import get_equilibrium_profits, compute_welfare, compute_hhi

    results = []
    for n in n_values:
        # Reduce k for large n to cap Q-table size
        k = max(15 - 3 * (n - 2), 6)
        if verbose:
            qtable_size = k ** (n + 1) * n
            print(f"\n{'='*60}")
            print(f"  n={n}  k={k}  Q-table entries={qtable_size:,}")
            print(f"{'='*60}")

        game = CalvanoModel(n=n, k=k, tmax=tmax)
        if verbose:
            print(game.summary())

        result = simulate_game(game, seed=42, verbose=verbose)
        game.Q = result['final_Q']

        pi_c, pi_n, eq_prices = get_equilibrium_profits(game)
        welfare = compute_welfare(game, eq_prices)
        hhi = compute_hhi(game, eq_prices)

        entry = {
            'n': n, 'k': k,
            'delta_val': result['profit_index'],
            'pi_c': pi_c, 'pi_n': pi_n,
            'eq_prices': eq_prices,
            'nash_prices': game.p_nash,
            'mono_prices': game.p_mono,
            'F_min': compute_F_min(pi_c, pi_n, game.delta),
            'converged': result['converged'],
            'convergence_t': result['convergence_t'],
            'consumer_surplus': welfare['consumer_surplus'],
            'hhi': hhi,
            'game': game,
            'result': result,
        }
        results.append(entry)

        if verbose:
            print(f"\n  --- n={n} Summary ---")
            print(f"  Delta      = {entry['delta_val']:.4f}")
            print(f"  pi_c       = {pi_c:.4f}")
            print(f"  pi_n       = {pi_n:.4f}")
            print(f"  F_min      = {entry['F_min']:.4f}")
            print(f"  CS         = {welfare['consumer_surplus']:.4f}")
            print(f"  HHI        = {hhi:.0f}")

    # Summary table
    if verbose and len(results) > 1:
        print(f"\n{'='*72}")
        print("  N-AGENT COMPARISON SUMMARY")
        print(f"{'='*72}")
        print(f"  {'n':>3}  {'k':>3}  {'Delta':>8}  {'pi_c':>8}  "
              f"{'pi_n':>8}  {'F_min':>8}  {'HHI':>6}  {'Conv?':>6}")
        print("-" * 72)
        for e in results:
            print(f"  {e['n']:>3}  {e['k']:>3}  "
                  f"{e['delta_val']:>8.4f}  "
                  f"{e['pi_c']:>8.4f}  "
                  f"{e['pi_n']:>8.4f}  "
                  f"{e['F_min']:>8.4f}  "
                  f"{e['hhi']:>6.0f}  "
                  f"{'YES' if e['converged'] else 'NO':>6}")
        print("=" * 72)

    return results
