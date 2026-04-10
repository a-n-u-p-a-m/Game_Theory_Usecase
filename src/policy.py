"""
M5 -- Policy Analysis: F_min computation with international fine benchmarks.

Jurisdictions:
  EU   : 10% of worldwide annual turnover (cap); base = up to 30% of
         relevant sales x years + 15-25% entry fee.
  USA  : Greater of $100M statutory max, 20% of volume-of-commerce (proxy
         for overcharge), or 2x gross gain/loss.  Plus treble civil damages.
  India: 10% of turnover OR 3x profit per year of cartel, whichever
         is higher.  Base up to 30% of relevant turnover (2024 CCI
         Guidelines).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .model import CalvanoModel
from .utils import find_greedy_equilibrium_state


# ======================================================================
#  F_min (theory)
# ======================================================================

def compute_F_min(pi_c: float, pi_n: float, delta: float) -> float:
    """F_min = (pi_c - pi_n) * delta / (1 - delta).
    PV of all future collusive surplus."""
    return (pi_c - pi_n) * delta / (1 - delta)


def compute_F_critical(pi_c: float, pi_n: float, delta: float,
                        tau_scale: float = 1.5) -> float:
    """
    F_crit: fine threshold where T* transitions from 1 to >1.

    For F <= F_crit, the 1-period mimicry cost already exceeds p(1)*F,
    so T*=1 (any audit duration works, even 1 period).
    For F > F_crit, the firm finds mimicry profitable at short audits,
    so a longer audit (T* > 1) is needed.

    NOTE: F_crit < F_min always. F_crit is NOT the undeterrable threshold;
    that is F_min (the PV bound of mimicry cost).

    F_crit = C_mimic(1) / p(1) = (pi_c-pi_n)*delta / (1-exp(-1/tau))
    """
    C_1 = (pi_c - pi_n) * delta  # mimicry cost for 1 period
    p_1 = 1.0 - np.exp(-1.0 / tau_scale)
    return C_1 / p_1 if p_1 > 1e-12 else float('inf')


def goldilocks_zone(pi_c: float, pi_n: float, delta: float,
                     tau_scale: float = 1.5) -> dict:
    """
    Fine regime analysis for the audit mechanism.

    The regulator has TWO win conditions:
      WIN #1 (Detection):  F < F_min -> firm doesn't mimic -> audit detects
                           collusion -> regulator enforces.
      WIN #2 (Deterrence): F >= F_min -> expected fine exceeds collusive
                           surplus -> firm plays Nash permanently
                           (under random/covert audits).

    Three fine regimes (all can be wins for the regulator):
      1. F <= F_crit        : T*=1   WIN #1 (easy detection)
      2. F_crit < F < F_min : T*>1   WIN #1 if T >= T* (calibrated detection)
                                     DANGER ZONE if T < T* (mimicry + revert)
      3. F >= F_min          : T*=inf WIN #2 (deterrence via permanent mimicry)

    The ONLY failure is F_crit < F < F_min with T < T*: the firm mimics
    during a too-short audit, avoids detection, then reverts to collusion.

    F_crit = (pi_c-pi_n)*delta / (1-exp(-1/tau))   [T*=1 threshold]
    F_min  = (pi_c-pi_n)*delta / (1-delta)          [deterrence threshold]
    """
    f_min = compute_F_min(pi_c, pi_n, delta)
    f_crit = compute_F_critical(pi_c, pi_n, delta, tau_scale)
    return {'F_min': f_min, 'F_crit': f_crit,
            'ratio': f_min / f_crit if f_crit > 1e-12 else float('inf'),
            'calibration_zone_width': f_min - f_crit,
            'easy_detection': f'F <= {f_crit:.4f}  ->  T*=1  (WIN #1: detection)',
            'calibrated_detection': f'{f_crit:.4f} < F < {f_min:.4f}  ->  T*>1  (WIN #1 if T>=T*, else FAIL)',
            'deterrence': f'F >= {f_min:.4f}  ->  permanent Nash  (WIN #2: deterrence)'}


# ======================================================================
#  International fine structures
# ======================================================================

JURISDICTIONS = {
    "EU": {
        "name": "European Union",
        "cap_rate": 0.10,          # 10% worldwide turnover cap
        "gravity_low": 0.15,       # base = gravity% x sales x duration
        "gravity_high": 0.30,
        "entry_fee_low": 0.15,     # 15-25% additional "entry fee"
        "entry_fee_high": 0.25,
        "description": ("10% of worldwide annual turnover (cap).  "
                        "Base fine = 15-30% of relevant sales x years "
                        "+ 15-25% entry fee."),
    },
    "USA": {
        "name": "United States (DOJ)",
        "vol_commerce_rate": 0.20, # 20% of volume-of-commerce (proxy)
        "min_fine_rate": 0.15,     # minimum fine = 15% of VoC
        "treble_multiplier": 3.0,  # civil treble damages on overcharge
        "description": ("Criminal: 20% of volume-of-commerce.  "
                        "Min = 15% VoC.  "
                        "Civil: treble (3x) damages on 20% overcharge.  "
                        "Total = criminal + civil."),
    },
    "India": {
        "name": "India (CCI)",
        "cap_rate": 0.10,          # 10% of turnover per year
        "profit_multiplier": 3.0,  # or 3x profit per year
        "base_rate_high": 0.30,    # base up to 30% relevant turnover
        "description": ("10% of turnover per year OR 3x annual profit, "
                        "whichever is higher (2024 CCI Guidelines).  "
                        "Base up to 30% of relevant turnover."),
    },
}


def compute_fine_eu(pi_c: float, margin: float,
                    duration_years: float = 1.0,
                    gravity: float = 0.25,
                    entry_fee: float = 0.20) -> dict:
    """
    EU cartel fine estimate.

    revenue = pi_c / margin
    base    = gravity * revenue * duration + entry_fee * revenue
    cap     = 0.10 * revenue  (worldwide turnover)
    fine    = min(base, cap)
    """
    revenue   = pi_c / margin
    base_fine = gravity * revenue * duration_years + entry_fee * revenue
    cap       = 0.10 * revenue
    fine      = min(base_fine, cap)
    return {"fine": fine, "base": base_fine, "cap": cap,
            "revenue": revenue, "jurisdiction": "EU"}


def compute_fine_usa(pi_c: float, margin: float,
                     volume_of_commerce: float = None,
                     duration_years: float = 1.0) -> dict:
    """
    US DOJ cartel fine estimate (rate-based, model-scale).

    criminal = max(20%, 15%) of volume_of_commerce
    civil    = 3 x overcharge_estimate  (overcharge ~ 20% VoC)
    total    = criminal + civil
    """
    revenue = pi_c / margin
    if volume_of_commerce is None:
        volume_of_commerce = revenue * duration_years
    c = JURISDICTIONS["USA"]
    criminal_fine = max(c["vol_commerce_rate"] * volume_of_commerce,
                        c["min_fine_rate"] * volume_of_commerce)
    overcharge = c["vol_commerce_rate"] * volume_of_commerce
    civil_damages = c["treble_multiplier"] * overcharge
    total = criminal_fine + civil_damages
    return {"fine": criminal_fine, "civil_damages": civil_damages,
            "total_exposure": total, "volume_of_commerce": volume_of_commerce,
            "jurisdiction": "USA"}


def compute_fine_india(pi_c: float, margin: float,
                       duration_years: float = 1.0,
                       base_gravity: float = 0.20) -> dict:
    """
    India CCI cartel fine estimate.

    turnover_fine = 0.10 * revenue * duration
    profit_fine   = 3 * pi_c * duration
    fine = max(turnover_fine, profit_fine)
    cap  = 0.10 * global_turnover  (per year)
    """
    revenue = pi_c / margin
    c = JURISDICTIONS["India"]
    turnover_fine = c["cap_rate"] * revenue * duration_years
    profit_fine   = c["profit_multiplier"] * pi_c * duration_years
    base_fine     = base_gravity * revenue * duration_years
    fine = max(turnover_fine, profit_fine, base_fine)
    cap  = c["cap_rate"] * revenue * duration_years
    return {"fine": min(fine, cap * 3), "turnover_method": turnover_fine,
            "profit_method": profit_fine, "base_method": base_fine,
            "jurisdiction": "India"}


def compute_fine_by_jurisdiction(jurisdiction: str, pi_c: float,
                                 margin: float,
                                 duration_years: float = 1.0) -> dict:
    """Dispatch to the correct fine calculator."""
    j = jurisdiction.upper()
    if j == "EU":
        return compute_fine_eu(pi_c, margin, duration_years)
    elif j == "USA":
        return compute_fine_usa(pi_c, margin,
                                duration_years=duration_years)
    elif j == "INDIA":
        return compute_fine_india(pi_c, margin, duration_years)
    else:
        raise ValueError(f"Unknown jurisdiction: {jurisdiction}")


# ======================================================================
#  Policy analysis (multi-jurisdiction)
# ======================================================================

def policy_analysis(game: CalvanoModel,
                     delta_values: List[float] = None,
                     profit_margin_range: Tuple[float, float] = (0.15, 0.30),
                     pi_c: float = None, pi_n: float = None,
                     verbose: bool = True) -> Dict:
    """
    F_min vs international fine benchmarks across delta values.
    """
    if delta_values is None:
        delta_values = list(np.arange(0.50, 1.00, 0.05))

    if pi_c is None or pi_n is None:
        from .utils import get_equilibrium_profits
        _c, _n, _ = get_equilibrium_profits(game)
        pi_c = pi_c or _c
        pi_n = pi_n or _n

    m_low, m_high = profit_margin_range
    m_mid = (m_low + m_high) / 2.0

    results = {
        'delta_values': delta_values,
        'F_min': [],
        'eu_fine_low': [], 'eu_fine_high': [],
        'usa_fine_low': [], 'usa_fine_high': [],
        'india_fine_low': [], 'india_fine_high': [],
    }

    for d in delta_values:
        fm = compute_F_min(pi_c, pi_n, d)
        results['F_min'].append(fm)
        # EU
        results['eu_fine_low'].append(
            compute_fine_eu(pi_c, m_low)['fine'])
        results['eu_fine_high'].append(
            compute_fine_eu(pi_c, m_high)['fine'])
        # USA
        results['usa_fine_low'].append(
            compute_fine_usa(pi_c, m_low)['total_exposure'])
        results['usa_fine_high'].append(
            compute_fine_usa(pi_c, m_high)['total_exposure'])
        # India
        results['india_fine_low'].append(
            compute_fine_india(pi_c, m_low)['fine'])
        results['india_fine_high'].append(
            compute_fine_india(pi_c, m_high)['fine'])

    if verbose:
        print("\n" + "=" * 90)
        print("  POLICY ANALYSIS -- F_min vs International Fine Benchmarks")
        print(f"  pi_c={pi_c:.4f}  pi_n={pi_n:.4f}  margins={m_low:.0%}-{m_high:.0%}")
        print("=" * 90)
        print(f"  {'delta':>6}  {'F_min':>8}  {'EU':>10}  "
              f"{'USA':>10}  {'India':>10}  {'Regulator Outcome':>22}")
        print("-" * 96)
        for i, d in enumerate(delta_values):
            fm = results['F_min'][i]
            fc = compute_F_critical(pi_c, pi_n, d)
            eu = results['eu_fine_high'][i]
            us = results['usa_fine_high'][i]
            ind = results['india_fine_high'][i]
            # Determine outcome per jurisdiction
            outcomes = []
            for name, fine in [('EU', eu), ('USA', us), ('India', ind)]:
                if fine <= fc:
                    outcomes.append(f'{name}:Detect(T*=1)')
                elif fine < fm:
                    from .t_star import compute_T_star_analytical
                    t_star = compute_T_star_analytical(pi_c, pi_n, d, fine)
                    outcomes.append(f'{name}:Detect(T*={t_star})')
                else:
                    outcomes.append(f'{name}:Deter')
            out_str = '  '.join(outcomes)
            print(f"  {d:>6.2f}  {fm:>8.4f}  {eu:>10.4f}  "
                  f"{us:>10.4f}  {ind:>10.4f}  {out_str}")
        print("=" * 96)

    return results


def compute_F_min_heatmap(delta_values: List[float] = None,
                           surplus_ratios: List[float] = None) -> Dict:
    """F_min over a 2D grid of (delta, pi_c/pi_n).  Pure theory."""
    if delta_values is None:
        delta_values = list(np.arange(0.50, 1.00, 0.02))
    if surplus_ratios is None:
        surplus_ratios = list(np.arange(1.1, 3.1, 0.1))

    pi_n_base  = 1.0
    F_min_grid = np.zeros((len(delta_values), len(surplus_ratios)))
    for i, d in enumerate(delta_values):
        for j, ratio in enumerate(surplus_ratios):
            F_min_grid[i, j] = compute_F_min(pi_n_base * ratio,
                                              pi_n_base, d)
    return {'delta_values': delta_values,
            'surplus_ratios': surplus_ratios,
            'F_min_grid': F_min_grid}
