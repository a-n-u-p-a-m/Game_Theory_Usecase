"""
All matplotlib plotting functions for CLI / static figure generation.
ASCII-safe for Windows consoles (FIX B10).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit, OptimizeWarning
from typing import Dict, List
import warnings

from .model import CalvanoModel


def set_plot_style():
    plt.rcParams.update({
        'figure.figsize': (10, 6), 'figure.dpi': 120,
        'font.family': 'serif', 'font.size': 12,
        'axes.labelsize': 13, 'axes.titlesize': 14,
        'legend.fontsize': 11, 'xtick.labelsize': 11,
        'ytick.labelsize': 11, 'axes.grid': True, 'grid.alpha': 0.3,
    })


def _pretty_run_label(algo_label: str = "") -> str:
    """Format labels like 'n2_q_learning' -> 'n=2 | Q-Learning'."""
    txt = str(algo_label or "").strip()
    if not txt:
        return "Q-Learning"

    raw = txt.replace('-', '_').lower()
    n_part = None
    algo_part = raw
    if raw.startswith('n') and '_' in raw:
        head, tail = raw.split('_', 1)
        if head[1:].isdigit():
            n_part = int(head[1:])
            algo_part = tail

    if algo_part == 'q_learning':
        algo_txt = 'Q-Learning'
    elif algo_part == 'sarsa':
        algo_txt = 'SARSA'
    else:
        algo_txt = txt

    if n_part is None:
        return algo_txt
    return f'n={n_part} | {algo_txt}'


# ------------------------------------------------------------------
#  M1 plots
# ------------------------------------------------------------------

def plot_convergence(result: Dict, game: CalvanoModel,
                     save_path: str = None,
                     algo_label: str = None):
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ph = np.array(result['price_history'])
    t_vals = np.arange(len(ph)) * 10_000

    for i in range(game.n):
        ax1.plot(t_vals, ph[:, i], label=f'Firm {i+1}', linewidth=1.5)
    ax1.axhline(np.mean(game.p_nash), color='green', ls='--', lw=1,
                label='Nash')
    ax1.axhline(np.mean(game.p_mono), color='red', ls='--', lw=1,
                label='Monopoly')
    ax1.set_xlabel('Period (t)'); ax1.set_ylabel('Price')
    ax1.set_title('Price Convergence'); ax1.legend()

    ax2.plot(t_vals, result['profit_index_history'],
             color='darkorange', linewidth=1.5)
    ax2.axhline(0, color='green', ls='--', lw=1, label='Delta=0 (Nash)')
    ax2.axhline(1, color='red', ls='--', lw=1, label='Delta=1 (Mono)')
    ax2.set_xlabel('Period (t)'); ax2.set_ylabel('Profit Index (Delta)')
    ax2.set_title('Profit Index Over Time'); ax2.set_ylim(-0.1, 1.3)
    ax2.legend()

    fig.suptitle(f'Calvano et al. (2020) -- {_pretty_run_label(algo_label)} Convergence',
                 fontsize=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_impulse_response(ir: Dict, game: CalvanoModel,
                          save_path: str = None,
                          context_label: str = ""):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    prices = ir['prices']
    dev_t = ir['deviation_t']
    t_axis = np.arange(len(prices)) - dev_t
    colors = plt.cm.tab10(np.linspace(0, 1, max(game.n, 2)))
    for i in range(game.n):
        label = 'Deviating' if i == 0 else f'Firm {i+1}'
        ax.plot(t_axis, prices[:, i], label=label, color=colors[i],
                linewidth=2, marker='o', markersize=3)
    ax.axhline(np.mean(game.p_nash), color='green', ls='--', lw=1,
               label='Nash')
    ax.axhline(np.mean(game.p_mono), color='red', ls='--', lw=1,
               label='Monopoly')
    ax.axvline(0, color='gray', ls=':', lw=1, alpha=0.7)
    ax.set_xlabel('Period relative to deviation')
    ax.set_ylabel('Price')
    if context_label:
        ax.set_title(f'Impulse Response -- Punishment & Forgiveness ({_pretty_run_label(context_label)})')
    else:
        ax.set_title('Impulse Response -- Punishment & Forgiveness')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_profit_index_distribution(results: List[Dict],
                                   save_path: str = None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    deltas = [r['profit_index'] for r in results]
    ax.hist(deltas, bins=20, range=(0, 1.2), color='steelblue',
            edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(deltas), color='red', ls='--', lw=2,
               label=f'Mean Delta = {np.mean(deltas):.3f}')
    ax.set_xlabel('Profit Index (Delta)'); ax.set_ylabel('Count')
    ax.set_title('Distribution of Convergence Outcomes'); ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_greedy_policy_heatmap(game: CalvanoModel, agent_idx: int = 0,
                               save_path: str = None,
                               context_label: str = ""):
    """Works only for n=2.  Skips with warning for n>2."""
    if game.n != 2:
        print("  [WARN] Policy heatmap only available for n=2")
        return None
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    policy = np.zeros((game.k, game.k))
    for s1 in range(game.k):
        for s2 in range(game.k):
            state = np.array([s1, s2])
            policy[s1, s2] = game.A[np.argmax(
                game.Q[(agent_idx,) + tuple(state)]
            )]
    im = ax.imshow(policy, origin='lower', cmap='RdYlGn_r',
                   aspect='equal')
    ax.set_xlabel("Firm 2's last price (index)")
    ax.set_ylabel("Firm 1's last price (index)")
    if context_label:
        ax.set_title(f'Greedy Policy ({_pretty_run_label(context_label)}) -- Firm {agent_idx + 1}')
    else:
        ax.set_title(f'Greedy Policy -- Firm {agent_idx + 1}')
    tick_pos = list(range(0, game.k, max(1, game.k // 5)))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f'{game.A[i]:.2f}' for i in tick_pos],
                       rotation=45)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f'{game.A[i]:.2f}' for i in tick_pos])
    plt.colorbar(im, ax=ax, label='Chosen price')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_parameter_sweep(sweep: Dict, save_path: str = None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = sweep['param_values']; y = sweep['mean_delta']
    yerr = sweep['std_delta']
    ax.errorbar(x, y, yerr=yerr, marker='o', capsize=4, linewidth=2,
                color='steelblue', ecolor='gray')
    ax.axhline(0, color='green', ls='--', lw=1, alpha=0.5,
               label='Nash (Delta=0)')
    ax.axhline(1, color='red', ls='--', lw=1, alpha=0.5,
               label='Monopoly (Delta=1)')
    ax.set_xlabel(sweep['param_name'])
    ax.set_ylabel('Profit Index (Delta)')
    ax.set_title(f'Comparative Statics: Delta vs {sweep["param_name"]}')
    ax.legend(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


# ------------------------------------------------------------------
#  M2 plots
# ------------------------------------------------------------------

def plot_p_T_curve(p_T: Dict[int, float], game: CalvanoModel,
                   save_path: str = None,
                   algo_label: str = ""):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    Ts = sorted(p_T.keys()); ps = [p_T[T] for T in Ts]
    ax.plot(Ts, ps, 'o-', color='#2196F3', linewidth=2, markersize=6,
            label='Empirical p(T)')
    T_cont = np.linspace(1, max(Ts), 200)
    try:
        def p_model(T, tau):
            return 1 - np.exp(-T / tau)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            popt, _ = curve_fit(p_model, Ts, ps, p0=[50.0], maxfev=5000)
        ax.plot(T_cont, p_model(T_cont, *popt), '--', color='#FF5722',
                linewidth=1.5, label=f'Fit: 1-exp(-T/{popt[0]:.1f})')
    except Exception:
        pass
    ax.set_xlabel('Audit Duration T')
    ax.set_ylabel('Detection Probability p(T)')
    if algo_label:
        ax.set_title(f'Detection Probability vs Audit Duration ({_pretty_run_label(algo_label)})')
    else:
        ax.set_title('Detection Probability vs Audit Duration')
    ax.set_ylim(-0.05, 1.05); ax.legend(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_p_T_curve_ci(p_t_ci: Dict, algo_label: str = "",
                      save_path: str = None):
    """Plot p(T) mean with 95% CI band across random seeds."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    Ts = p_t_ci['T_values']
    summary = p_t_ci['summary']
    means = [summary[int(T)]['mean'] for T in Ts]
    lows = [summary[int(T)]['ci_low'] for T in Ts]
    highs = [summary[int(T)]['ci_high'] for T in Ts]

    ax.plot(Ts, means, 'o-', color='#1976D2', linewidth=2,
            label='Mean p(T)')
    ax.fill_between(Ts, lows, highs, color='#64B5F6', alpha=0.25,
                    label='95% CI')

    ax.set_xlabel('Audit Duration T')
    ax.set_ylabel('Detection Probability p(T)')
    ax.set_title(f'p(T) with Uncertainty ({_pretty_run_label(algo_label)})')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_empirical_tstar_bands(tstar_ci: Dict, algo_label: str = "",
                               save_path: str = None):
    """Plot empirical T* mean with 95% CI band across seed curves."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    F_vals = tstar_ci['F_values']
    summary = tstar_ci['summary']
    means = [summary[float(F)]['mean'] for F in F_vals]
    lows = [summary[float(F)]['ci_low'] for F in F_vals]
    highs = [summary[float(F)]['ci_high'] for F in F_vals]

    ax.plot(F_vals, means, 'o-', color='#388E3C', linewidth=2,
            label='Empirical T* mean')
    ax.fill_between(F_vals, lows, highs, color='#81C784', alpha=0.25,
                    label='95% CI')

    ax.set_xlabel('Fine F')
    ax.set_ylabel('Empirical T*')
    ax.set_title(f'Empirical T* Uncertainty ({_pretty_run_label(algo_label)})')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


# ------------------------------------------------------------------
#  M3 plots
# ------------------------------------------------------------------

def plot_mimicry_cost_comparison(costs: Dict, game: CalvanoModel,
                                  save_path: str = None,
                                  algo_label: str = ""):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    Ts = costs['T_values']; emp = costs['empirical']
    theo = costs['theoretical']
    pi_c, pi_n = costs['pi_c'], costs['pi_n']
    ax.plot(Ts, theo, 's--', color='#FF5722', linewidth=2, markersize=6,
            label='Theoretical C_mimic(T)')
    ax.plot(Ts, emp, 'o-', color='#2196F3', linewidth=2, markersize=6,
            label='Empirical C_mimic(T)')
    C_inf = (pi_c - pi_n) * game.delta / (1 - game.delta)
    ax.axhline(C_inf, color='gray', ls=':', lw=1.5,
               label=f'PV bound = {C_inf:.4f}')
    ax.set_xlabel('Audit Duration T')
    ax.set_ylabel('Discounted Mimicry Cost')
    if algo_label:
        ax.set_title(f'Mimicry Cost: Empirical vs Theoretical ({_pretty_run_label(algo_label)})')
    else:
        ax.set_title('Mimicry Cost: Empirical vs Theoretical')
    ax.legend(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


# ------------------------------------------------------------------
#  M4 plots
# ------------------------------------------------------------------

def plot_T_star_crossing(game: CalvanoModel, F: float,
                          pi_c: float = None, pi_n: float = None,
                          p_T_dict: Dict[int, float] = None,
                          tau_scale: float = None,
                          save_path: str = None):
    if pi_c is None or pi_n is None:
        from .utils import get_equilibrium_profits
        _c, _n, _ = get_equilibrium_profits(game)
        pi_c = pi_c or _c; pi_n = pi_n or _n

    from .t_star import compute_T_star_analytical, DEFAULT_TAU

    if tau_scale is None:
        tau_scale = DEFAULT_TAU

    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    T_range = np.arange(1, 501)
    C_mimic = [(pi_c - pi_n) * game.delta * (1 - game.delta**T)
               / (1 - game.delta) for T in T_range]
    pT_F = [(1 - np.exp(-T / tau_scale)) * F for T in T_range]

    ax.plot(T_range, C_mimic, color='#2196F3', linewidth=2.5,
            label='C_mimic(T)')
    ax.plot(T_range, pT_F, color='#FF5722', linewidth=2.5,
            label=f'p(T)*F  [F={F:.1f}]')

    if p_T_dict:
        Ts_emp = sorted(p_T_dict.keys())
        ax.plot(Ts_emp, [p_T_dict[T] * F for T in Ts_emp], 'o',
                color='#FF9800', markersize=6,
                label='Empirical p(T)*F', zorder=5)

    T_star = compute_T_star_analytical(
        pi_c, pi_n, game.delta, F, tau_scale=tau_scale
    )
    C_at = (pi_c - pi_n) * game.delta * (1 - game.delta**T_star) \
           / (1 - game.delta)
    ax.axvline(T_star, color='green', ls='--', lw=2, alpha=0.7)
    ax.plot(T_star, C_at, '*', color='green', markersize=15, zorder=10,
            label=f'T* = {T_star}')

    C_inf = (pi_c - pi_n) * game.delta / (1 - game.delta)
    ax.axhline(C_inf, color='#2196F3', ls=':', lw=1, alpha=0.5)
    ax.axhline(F, color='#FF5722', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Audit Duration T'); ax.set_ylabel('Monetary Value')
    ax.set_title(f'Theorem 1 IC Crossing (F={F:.1f}, pi_c={pi_c:.4f}, '
                 f'pi_n={pi_n:.4f}, delta={game.delta}, '
                 f'tau={tau_scale:.2f})')
    ax.legend(loc='center right')
    ax.set_xlim(0, min(500, max(T_star * 3, 50)))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_T_star_heatmap(grid: Dict, save_path: str = None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    T_grid = grid['T_star_grid']
    T_capped = np.clip(T_grid, 1, 500)
    im = ax.imshow(T_capped, origin='lower', cmap='YlOrRd',
                   aspect='auto', interpolation='nearest')
    dv = grid['delta_values']; fv = grid['F_values']
    ax.set_xticks(range(len(fv)))
    ax.set_xticklabels([f'{f:.1f}' for f in fv], rotation=45)
    ax.set_yticks(range(len(dv)))
    ax.set_yticklabels([f'{d:.2f}' for d in dv])
    ax.set_xlabel('Fine (F)'); ax.set_ylabel('Discount Factor (delta)')
    ax.set_title('T* -- Minimum IC Audit Duration')
    for i in range(len(dv)):
        for j in range(len(fv)):
            v = int(T_grid[i, j])
            ax.text(j, i, f'{v}' if v < 1000 else '>1K',
                    ha='center', va='center', fontsize=8,
                    color='white' if v > 200 else 'black')
    plt.colorbar(im, ax=ax, label='T*')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


# ------------------------------------------------------------------
#  M5 plots
# ------------------------------------------------------------------

def plot_policy_analysis(policy: Dict, save_path: str = None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    d = policy['delta_values']; fm = policy['F_min']

    ax.plot(d, fm, 'o-', color='#2196F3', linewidth=2.5, markersize=7,
            label='F_min (IC threshold)')

    # EU
    ax.fill_between(d, policy['eu_fine_low'], policy['eu_fine_high'],
                    alpha=0.15, color='#4CAF50')
    ax.plot(d, policy['eu_fine_high'], '--', color='#4CAF50', lw=1.5,
            label='EU fine range')
    # USA
    ax.fill_between(d, policy['usa_fine_low'], policy['usa_fine_high'],
                    alpha=0.15, color='#FF9800')
    ax.plot(d, policy['usa_fine_high'], '--', color='#FF9800', lw=1.5,
            label='USA fine range')
    # India
    ax.fill_between(d, policy['india_fine_low'],
                    policy['india_fine_high'], alpha=0.15, color='#9C27B0')
    ax.plot(d, policy['india_fine_high'], '--', color='#9C27B0', lw=1.5,
            label='India fine range')

    ax.set_xlabel('Discount Factor (delta)')
    ax.set_ylabel('Fine Level')
    ax.set_title('F_min vs International Cartel Fines')
    ax.legend(loc='upper left'); ax.set_yscale('log')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_F_min_heatmap(heatmap: Dict, eu_fine_level: float = None,
                        save_path: str = None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    F_grid = heatmap['F_min_grid']
    dv = heatmap['delta_values']; sv = heatmap['surplus_ratios']
    im = ax.imshow(F_grid, origin='lower', cmap='inferno',
                   aspect='auto', interpolation='bilinear',
                   norm=LogNorm(vmin=max(0.01, F_grid.min()),
                                vmax=F_grid.max()))
    sx = max(1, len(sv) // 8); sy = max(1, len(dv) // 8)
    ax.set_xticks(range(0, len(sv), sx))
    ax.set_xticklabels([f'{sv[i]:.1f}' for i in range(0, len(sv), sx)],
                       rotation=45)
    ax.set_yticks(range(0, len(dv), sy))
    ax.set_yticklabels([f'{dv[i]:.2f}' for i in range(0, len(dv), sy)])
    ax.set_xlabel('Collusive Surplus Ratio (pi_c / pi_n)')
    ax.set_ylabel('Discount Factor (delta)')
    ax.set_title('F_min -- Minimum Fine for IC')
    if eu_fine_level:
        CS = ax.contour(F_grid, levels=[eu_fine_level],
                        colors=['lime'], linewidths=2)
        ax.clabel(CS, fmt=f'EU={eu_fine_level:.3f}', fontsize=10)
    plt.colorbar(im, ax=ax, label='F_min')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_noisy_mimicry(noise_dict: Dict, algo_label: str = "", save_path: str = None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = noise_dict['noise_levels']
    y = noise_dict['detection_rates']
    
    # Plot as percentages
    ax.plot([n * 100 for n in x], [d * 100 for d in y], 'D-', color='#9C27B0', 
            linewidth=2, markersize=8,
            label=f'Detection Rate ({_pretty_run_label(algo_label)})')
            
    ax.set_xlabel('Mimicry Noise (%)')
    ax.set_ylabel('Detection Probability (%)')
    ax.set_title(f'Vulnerability of Imperfect Mimicry ({_pretty_run_label(algo_label)})')
    ax.set_ylim(-5, 105)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def plot_theorem2_comparative_statics(t2_results: Dict, save_path: str = None):
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # (i) T* vs delta
    for r in t2_results['delta_sweeps']:
        ax1.plot(r['delta_values'], r['T_star_values'], 'o-', lw=2, label=f"F = {r['F']}")
    ax1.set_xlabel('Discount Factor (delta)')
    ax1.set_ylabel('Minimum Audit Duration (T*)')
    ax1.set_title('(i) T* vs delta (fixed F)')
    ax1.legend()

    # (ii) T* vs F
    for r in t2_results['F_sweeps']:
        ax2.plot(r['F_values'], r['T_star_values'], 's-', lw=2, label=f"delta = {r['delta']:.2f}")
    ax2.set_xlabel('Fine (F)')
    ax2.set_ylabel('Minimum Audit Duration (T*)')
    ax2.set_title('(ii) T* vs F (fixed delta)')
    ax2.legend()
    
    fig.suptitle('Theorem 2: Comparative Statics of T*', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def plot_theorem3_comparison(t3c_results: Dict, save_path: str = None):
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    Ts = t3c_results['T_values']
    
    q_r = [r['ratio'] for r in t3c_results['q_learning']]
    s_r = [r['ratio'] for r in t3c_results['sarsa']]
    ax1.plot(Ts, q_r, 'o-', label='Q-Learning', lw=2, markersize=8)
    ax1.plot(Ts, s_r, 's-', label='SARSA', lw=2, markersize=8, alpha=0.7)
    ax1.axhline(1.0, color='gray', ls='--', alpha=0.8, label='Rational Bound')
    ax1.set_xlabel('Audit Duration (T_mimic)')
    ax1.set_ylabel('Ratio (RL total / Rational theoretical)')
    ax1.set_title('Theorem 3: RL Disruption Cost Ratio')
    ax1.legend()

    q_rec = [r['mean_recovery_time'] for r in t3c_results['q_learning']]
    s_rec = [r['mean_recovery_time'] for r in t3c_results['sarsa']]
    
    width = 0.35
    x = np.arange(len(Ts))
    ax2.bar(x - width/2, q_rec, width, label='Q-Learning', color='#2196F3')
    ax2.bar(x + width/2, s_rec, width, label='SARSA', color='#FF9800')
    ax2.set_xticks(x)
    ax2.set_xticklabels(Ts)
    ax2.set_xlabel('Audit Duration (T_mimic)')
    ax2.set_ylabel('Recovery periods')
    ax2.set_title('Post-Audit Re-Convergence Time')
    ax2.legend()
    
    fig.suptitle('RL Disruption Cost: Q-Learning vs SARSA', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig
