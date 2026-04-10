"""
M4 -- T* Verification: analytical and empirical minimum audit duration.
"""

import numpy as np
from typing import Dict, List, Callable, Optional
from scipy.optimize import curve_fit

from .model import CalvanoModel
from .utils import find_greedy_equilibrium_state


# Default tau_scale: 1.5 fits the empirical data where p(T)~1 at T~5.
# The original Calvano-era assumption was tau=50 (very slow detection).
DEFAULT_TAU = 1.5
ORIGINAL_TAU = 50.0


def p_T_parametric(T: float, tau: float = DEFAULT_TAU) -> float:
    """Parametric detection model: p(T) = 1 - exp(-T/tau)."""
    return 1.0 - np.exp(-T / tau)


def fit_p_T_model(p_T_dict: Dict[int, float]) -> dict:
    """
    Fit p(T) = 1 - exp(-T/tau) to empirical data.
    Returns {'tau': fitted_tau, 'r_squared': R2}.
    """
    Ts = np.array(sorted(p_T_dict.keys()), dtype=float)
    ps = np.array([p_T_dict[int(T)] for T in Ts])

    def model(T, tau):
        return 1.0 - np.exp(-T / tau)

    try:
        popt, _ = curve_fit(model, Ts, ps, p0=[5.0], maxfev=5000)
        fitted_tau = popt[0]
        ps_pred = model(Ts, fitted_tau)
        ss_res = np.sum((ps - ps_pred) ** 2)
        ss_tot = np.sum((ps - np.mean(ps)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return {'tau': fitted_tau, 'r_squared': r2}
    except Exception:
        return {'tau': DEFAULT_TAU, 'r_squared': 0.0}


def compare_p_T_models(p_T_dict: Dict[int, float],
                       verbose: bool = True) -> dict:
    """
    Compare empirical p(T) against parametric models
    (original tau=50 vs fitted tau).
    """
    fit = fit_p_T_model(p_T_dict)
    Ts = sorted(p_T_dict.keys())

    original_mse, fitted_mse = 0.0, 0.0
    if verbose:
        print("\n  p(T) Model Comparison:")
        fitted_hdr = f"Fitted(tau={fit['tau']:.2f})"
        print(f"  {'T':>5}  {'Empirical':>10}  {'Original(tau=50)':>16}  "
              f"{fitted_hdr:>18}")
        print("  " + "-" * 60)

    for T in Ts:
        emp = p_T_dict[T]
        orig = p_T_parametric(T, ORIGINAL_TAU)
        fitted = p_T_parametric(T, fit['tau'])
        original_mse += (emp - orig) ** 2
        fitted_mse += (emp - fitted) ** 2
        if verbose:
            print(f"  {T:>5}  {emp:>10.3f}  {orig:>16.3f}  {fitted:>18.3f}")

    n = len(Ts)
    original_mse /= n
    fitted_mse /= n
    if verbose:
        print(f"\n  MSE (original tau=50): {original_mse:.6f}")
        print(f"  MSE (fitted  tau={fit['tau']:.2f}): {fitted_mse:.6f}")
        print(f"  R^2 of fitted model:  {fit['r_squared']:.4f}")

    return {'fit': fit, 'original_mse': original_mse,
            'fitted_mse': fitted_mse}


def compute_T_star_analytical(pi_c: float, pi_n: float,
                               delta: float, F: float,
                               p_T_func: Optional[Callable] = None,
                               tau_scale: float = DEFAULT_TAU,
                               T_max: int = 1000) -> int:
    """
    T* = min{T : C_mimic(T) >= p(T)*F}.

    FIX B7: removed the shortcut 'if F >= F_max: return 1'.
    For large F the expected fine at T=1 can exceed C_mimic(1),
    making mimicry profitable at short audits.  The loop handles
    all cases; returns T_max if IC never holds (mimicry undeterrable).

    tau_scale: controls p(T) = 1-exp(-T/tau_scale).
        DEFAULT_TAU=1.5 (fitted to empirical data).
        Use ORIGINAL_TAU=50 for the paper's original parametric model.
    """
    for T in range(1, T_max + 1):
        C_mimic = (pi_c - pi_n) * delta * (1 - delta**T) / (1 - delta)
        pT = (p_T_func(T) if p_T_func is not None
               else p_T_parametric(T, tau_scale))
        if C_mimic >= pT * F:
            return T
    return T_max


def compute_T_star_empirical(game: CalvanoModel, F: float,
                              p_T_dict: Dict[int, float],
                              pi_c: float = None,
                              pi_n: float = None,
                              T_max: int = 1000,
                              extrapolation: str = 'hold_last') -> int:
    """
    Empirical T* from measured p(T), evaluated on a dense T grid.

    For T values beyond the largest measured key, the default behavior is
    stepwise hold-last extrapolation, which keeps the empirical crossing rule
    consistent with the measured p(T) source.
    """
    if pi_c is None or pi_n is None:
        from .utils import get_equilibrium_profits
        _pi_c, _pi_n, _ = get_equilibrium_profits(game)
        pi_c = pi_c or _pi_c
        pi_n = pi_n or _pi_n

    if not p_T_dict:
        return int(T_max)

    Ts_sorted = np.array(sorted(int(T) for T in p_T_dict.keys()), dtype=int)
    ps_sorted = np.array([float(p_T_dict[int(T)]) for T in Ts_sorted],
                         dtype=float)
    ps_sorted = np.clip(ps_sorted, 0.0, 1.0)

    search_max = max(int(T_max), int(Ts_sorted[-1]))
    extrap_mode = str(extrapolation).strip().lower()

    for T in range(1, search_max + 1):
        # Right-continuous step function over measured T-grid.
        idx = int(np.searchsorted(Ts_sorted, T, side='right') - 1)
        if idx < 0:
            pT = float(ps_sorted[0])
        elif idx >= len(ps_sorted):
            if extrap_mode == 'hold_last':
                pT = float(ps_sorted[-1])
            else:
                pT = float(ps_sorted[-1])
        else:
            pT = float(ps_sorted[idx])

        C_mimic = ((pi_c - pi_n) * game.delta
                   * (1 - game.delta**T) / (1 - game.delta))
        if C_mimic >= pT * F:
            return T
    return int(search_max)


def T_star_grid_search(game: CalvanoModel,
                        delta_values: List[float] = None,
                        F_values: List[float] = None,
                        pi_c: float = None,
                        pi_n: float = None,
                        verbose: bool = True) -> Dict:
    """Analytical T* over a (delta, F) grid."""
    if delta_values is None:
        delta_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    if F_values is None:
        F_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    if pi_c is None or pi_n is None:
        from .utils import get_equilibrium_profits
        _pi_c, _pi_n, _ = get_equilibrium_profits(game)
        pi_c = pi_c or _pi_c
        pi_n = pi_n or _pi_n

    T_grid = np.zeros((len(delta_values), len(F_values)))
    for i, d in enumerate(delta_values):
        for j, F in enumerate(F_values):
            T_grid[i, j] = compute_T_star_analytical(pi_c, pi_n, d, F)
            if verbose:
                print(f"  delta={d:.2f}, F={F:>6.1f}  ->  "
                      f"T*={int(T_grid[i,j]):>5d}")

    return {'delta_values': delta_values, 'F_values': F_values,
            'T_star_grid': T_grid, 'pi_c': pi_c, 'pi_n': pi_n}


def theorem2_comparative_statics(pi_c: float, pi_n: float,
                                  delta_values: List[float] = None,
                                  F_values: List[float] = None,
                                  verbose: bool = True) -> Dict:
    """
    Theorem 2 verification: comparative statics of T* via finite differences.

    (i)  ∂T*/∂δ ≤ 0  (more patient firms → shorter audit needed)
    (ii) ∂T*/∂F ≥ 0  (higher fine → longer audit needed)

    These follow from the Implicit Function Theorem applied to
    the IC crossing condition C_mimic(T) = p(T)*F.
    """
    if delta_values is None:
        delta_values = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    if F_values is None:
        F_values = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

    # (i) T* as function of δ (holding F fixed)
    delta_results = []
    for F in [0.5, 1.0, 2.0]:
        T_vals = [compute_T_star_analytical(pi_c, pi_n, d, F)
                  for d in delta_values]
        monotone = all(T_vals[i] >= T_vals[i+1]
                       for i in range(len(T_vals)-1))
        delta_results.append({
            'F': F, 'delta_values': delta_values,
            'T_star_values': T_vals,
            'non_increasing': monotone,
        })

    # (ii) T* as function of F (holding δ fixed)
    F_results = []
    for d in [0.80, 0.90, 0.95]:
        T_vals = [compute_T_star_analytical(pi_c, pi_n, d, F)
                  for F in F_values]
        monotone = all(T_vals[i] <= T_vals[i+1]
                       for i in range(len(T_vals)-1))
        F_results.append({
            'delta': d, 'F_values': F_values,
            'T_star_values': T_vals,
            'non_decreasing': monotone,
        })

    # Overall verification
    part_i_ok = all(r['non_increasing'] for r in delta_results)
    part_ii_ok = all(r['non_decreasing'] for r in F_results)

    if verbose:
        print("\n  " + "=" * 65)
        print("  Theorem 2: Comparative Statics of T*")
        print("  " + "=" * 65)

        print("\n  (i) dT*/d_delta <= 0 (T* non-increasing in delta):")
        for r in delta_results:
            status = "OK" if r['non_increasing'] else "FAIL"
            print(f"    F={r['F']:.1f}: T* = {r['T_star_values']}  [{status}]")

        print(f"\n  (ii) dT*/dF >= 0 (T* non-decreasing in F):")
        for r in F_results:
            status = "OK" if r['non_decreasing'] else "FAIL"
            print(f"    d={r['delta']:.2f}: T* = {r['T_star_values']}  [{status}]")

        overall = "VERIFIED" if (part_i_ok and part_ii_ok) else "FAILED"
        print(f"\n  Theorem 2: {overall}")
        print("  " + "=" * 65)

    return {
        'delta_sweeps': delta_results,
        'F_sweeps': F_results,
        'part_i_verified': part_i_ok,
        'part_ii_verified': part_ii_ok,
        'verified': part_i_ok and part_ii_ok,
    }

