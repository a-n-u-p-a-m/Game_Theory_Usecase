"""
=============================================================================
  Comprehensive Experiment Runner
=============================================================================
  Runs ALL experiments across ALL dimensions for the paper.

  Dimensions:
    - Algorithm:  Q-learning, SARSA
    - Firms:      n=2, n=3
    - Audit:      Fixed-T, Adaptive Bayesian
    - Mimicry:    Perfect, Noisy (5%, 10%)
    - Theorems:   T2 (comparative statics), T3 (RL disruption)

  Usage:
    python run_experiments.py               # full (~20 min)
    python run_experiments.py --quick       # fast (~5 min)
=============================================================================
"""

import numpy as np
import json
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from src.model import CalvanoModel
from src.simulation import simulate_game, simulate_game_sarsa, impulse_response
from src.audit import (calibrate_audit_tau,
                       measure_p_T_curve_with_ci,
                       empirical_T_star_with_ci)
from src.mimicry import (run_audit_with_mimicry, measure_mimicry_costs,
                         noise_sweep, theorem3_sweep,
                         theorem3_comparison)
from src.t_star import (compute_T_star_analytical,
                        theorem2_comparative_statics)
from src.policy import (compute_F_min, goldilocks_zone, policy_analysis)
from src.adaptive_audit import compare_adaptive_vs_fixed
from src.utils import (get_equilibrium_profits, compute_welfare,
                       compute_hhi)
from src.plotting import (plot_convergence, plot_p_T_curve, 
                          plot_p_T_curve_ci, plot_empirical_tstar_bands,
                          plot_noisy_mimicry, plot_theorem2_comparative_statics,
                          plot_theorem3_comparison, plot_policy_analysis,
                          plot_T_star_heatmap, plot_impulse_response,
                          plot_greedy_policy_heatmap, plot_mimicry_cost_comparison,
                          plot_T_star_crossing)


def _mean_ci95(values):
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, 0.0, mean, mean
    std = float(np.std(arr, ddof=1))
    half = 1.96 * std / np.sqrt(arr.size)
    return mean, std, float(mean - half), float(mean + half)


def run_all(quick=False, save_plots=True):
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    os.makedirs(results_dir, exist_ok=True)
    if save_plots:
        os.makedirs(figures_dir, exist_ok=True)

    all_results = {}
    t_start = time.time()
    tmax = 5_000_000
    seed_ci_count = 3 if quick else 7
    seed_ci_values = [42 + 17 * i for i in range(seed_ci_count)]

    # ==========================================
    #  1. Training: Q-learning and SARSA, n=2,3
    # ==========================================
    print("\n" + "=" * 72)
    print("  EXPERIMENT 1: Training Agents")
    print("=" * 72)

    n_values = [2, 3] if not quick else [2]

    for n in n_values:
        k = max(15 - 3 * (n - 2), 6)
        for algo in ['q_learning', 'sarsa']:
            label = f"n{n}_{algo}"
            print(f"\n  --- {label} (k={k}) ---")

            game = CalvanoModel(n=n, k=k, tmax=tmax,
                                c=1.0, a=2.0, a0=0.0, mu=0.25,
                                alpha=0.15, beta=4e-6, delta=0.95)

            if algo == 'q_learning':
                result = simulate_game(game, seed=42, verbose=True)
            else:
                result = simulate_game_sarsa(game, seed=42, verbose=True)

            game.Q = result['final_Q']
            pi_c, pi_n, eq_prices = get_equilibrium_profits(game)
            welfare = compute_welfare(game, eq_prices)
            hhi = compute_hhi(game, eq_prices)
            gz = goldilocks_zone(pi_c, pi_n, game.delta)

            entry = {
                'n': n, 'k': k, 'algorithm': algo,
                'delta_val': float(result['profit_index']),
                'pi_c': float(pi_c), 'pi_n': float(pi_n),
                'F_min': float(gz['F_min']),
                'F_crit': float(gz['F_crit']),
                'converged': result['converged'],
                'convergence_t': result['convergence_t'],
                'cs': float(welfare['consumer_surplus']),
                'hhi': float(hhi),
            }
            all_results[label] = entry
            print(f"  Result: Delta={entry['delta_val']:.4f}  "
                  f"Conv={entry['converged']}  "
                  f"HHI={entry['hhi']:.0f}")

            # Save game for later use
            all_results[f"{label}_game"] = game
            all_results[f"{label}_result"] = result

            if save_plots:
                plot_path = os.path.join(figures_dir, f'convergence_{label}.png')
                plot_convergence(result, game, save_path=plot_path,
                                 algo_label=label)

                if n == 2:
                    heatmap_path = os.path.join(figures_dir, f'greedy_policy_{label}.png')
                    plot_greedy_policy_heatmap(
                        game, agent_idx=0, save_path=heatmap_path,
                        context_label=label,
                    )
                
                # Impulse response
                try:
                    ir = impulse_response(game, result['final_Q'],
                                          T_before=5, T_after=15)
                    ir_path = os.path.join(figures_dir, f'impulse_response_{label}.png')
                    plot_impulse_response(ir, game, save_path=ir_path,
                                          context_label=label)
                except Exception as e:
                    print(f"    [WARN] Could not compute impulse response: {e}")

    # =====================================================
    #  1B. Multi-seed CI across algorithm and firm count
    # =====================================================
    print("\n" + "=" * 72)
    print("  EXPERIMENT 1B: Multi-Seed Robustness CI")
    print("=" * 72)

    for n in n_values:
        k = max(15 - 3 * (n - 2), 6)
        for algo in ['q_learning', 'sarsa']:
            label = f"n{n}_{algo}"
            deltas = []
            conv_times = []
            conv_count = 0

            for seed in seed_ci_values:
                game_ms = CalvanoModel(n=n, k=k, tmax=tmax,
                                       c=1.0, a=2.0, a0=0.0, mu=0.25,
                                       alpha=0.15, beta=4e-6, delta=0.95)
                if algo == 'q_learning':
                    r_ms = simulate_game(game_ms, seed=seed, verbose=False)
                else:
                    r_ms = simulate_game_sarsa(game_ms, seed=seed,
                                               verbose=False)

                deltas.append(float(r_ms['profit_index']))
                conv_times.append(float(r_ms['convergence_t']))
                conv_count += int(bool(r_ms['converged']))

            d_mean, d_std, d_low, d_high = _mean_ci95(deltas)
            ct_mean, ct_std, ct_low, ct_high = _mean_ci95(conv_times)
            all_results[f"{label}_multiseed_ci"] = {
                'seed_values': [int(s) for s in seed_ci_values],
                'delta_values': [float(v) for v in deltas],
                'delta_mean': d_mean,
                'delta_std': d_std,
                'delta_ci_low': d_low,
                'delta_ci_high': d_high,
                'convergence_time_mean': ct_mean,
                'convergence_time_std': ct_std,
                'convergence_time_ci_low': ct_low,
                'convergence_time_ci_high': ct_high,
                'converged_fraction': float(conv_count / len(seed_ci_values)),
            }
            print(f"  {label:<16} Delta={d_mean:.3f} "
                  f"CI95=[{d_low:.3f},{d_high:.3f}]  "
                  f"Conv={conv_count}/{len(seed_ci_values)}")

    # ==========================================
    #  2. p(T) curves and mimicry
    # ==========================================
    print("\n" + "=" * 72)
    print("  EXPERIMENT 2: Audit & Mimicry Analysis + Uncertainty")
    print("=" * 72)

    if quick:
        T_vals = [1, 2, 3, 5, 10, 20]
        n_mc = 40
        audit_ci_seeds = [101, 202, 303]
    else:
        T_vals = list(range(1, 21)) + [25, 30, 40, 50, 75, 100]
        n_mc = 350
        audit_ci_seeds = [101, 202, 303, 404, 505, 606, 707]
    n_runs_per_seed = max(10, n_mc // len(audit_ci_seeds))
    audit_detection_metric = 'mean'
    audit_min_consecutive_hits = 2
    audit_epsilon = 0.02 if quick else 0.01
    audit_tau_by_label = {}
    audit_tau_calibration_by_label = {}
    tstar_search_max = 500 if quick else 1200

    for label in [k for k in all_results if k.endswith('_game')]:
        algo_label = label.replace('_game', '')
        game = all_results[label]
        info = all_results[algo_label]
        print(f"\n  --- {algo_label}: p(T) mean + CI + mimicry ---")

        # Robust tau calibration from held-out non-collusive and collusive
        # signal distributions.
        tau_cfg = calibrate_audit_tau(
            game,
            detection_metric=audit_detection_metric,
            n_samples=600 if quick else 1600,
            competitive_noise=0.07 if quick else 0.05,
            collusive_state_jitter=2,
            collusive_epsilon=audit_epsilon,
            competitive_quantile=0.98 if quick else 0.99,
            collusive_quantile=0.20,
            min_margin=0.01,
            min_tau=0.02,
            max_tau=0.50,
        )
        tau_audit = float(tau_cfg['tau'])
        audit_tau_by_label[algo_label] = float(tau_audit)
        audit_tau_calibration_by_label[algo_label] = tau_cfg
        print(f"  [audit cfg] tau={tau_audit:.3f}  "
              f"metric={audit_detection_metric}  "
              f"consecutive_hits={audit_min_consecutive_hits}  "
              f"eps={audit_epsilon:.3f}")

        p_t_ci = measure_p_T_curve_with_ci(
            game,
            T_values=T_vals,
            tau=tau_audit,
            n_runs_per_seed=n_runs_per_seed,
            seeds=audit_ci_seeds,
            audit_epsilon=audit_epsilon,
            detection_metric=audit_detection_metric,
            min_consecutive_hits=audit_min_consecutive_hits,
            enforce_monotone=True,
            verbose=True,
        )
        p_T = {
            int(T): float(p_t_ci['summary'][int(T)]['mean'])
            for T in p_t_ci['T_values']
        }
        all_results[f"{algo_label}_pT"] = p_T
        all_results[f"{algo_label}_pT_ci"] = p_t_ci

        # Empirical T* uncertainty bands from seed-specific p(T) curves
        tstar_ci = empirical_T_star_with_ci(
            game,
            p_t_ci['curves_by_seed'],
            F_values=[0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
            pi_c=info['pi_c'],
            pi_n=info['pi_n'],
            t_search_max=tstar_search_max,
            extrapolation='hold_last',
            verbose=True,
        )
        all_results[f"{algo_label}_tstar_empirical_ci"] = tstar_ci
        all_results[f"{algo_label}_audit_tau_calibration"] = tau_cfg

        # Noisy mimicry
        noise_res = noise_sweep(game, T_audit=20,
                                 tau_threshold=tau_audit,
                                 n_runs=n_mc, verbose=True)
        all_results[f"{algo_label}_noise"] = {
            'noise_levels': [float(x) for x in noise_res['noise_levels']],
            'detection_rates': [float(x) for x in noise_res['detection_rates']],
        }

        if save_plots:
            plot_path_pt = os.path.join(figures_dir, f'p_T_curve_{algo_label}.png')
            plot_p_T_curve(p_T, game, save_path=plot_path_pt,
                           algo_label=algo_label)
            plot_path_pt_ci = os.path.join(figures_dir, f'p_T_curve_ci_{algo_label}.png')
            plot_p_T_curve_ci(p_t_ci, algo_label=algo_label,
                              save_path=plot_path_pt_ci)

            plot_path_tstar_ci = os.path.join(
                figures_dir, f'tstar_empirical_ci_{algo_label}.png'
            )
            plot_empirical_tstar_bands(tstar_ci, algo_label=algo_label,
                                       save_path=plot_path_tstar_ci)

            plot_path_noise = os.path.join(figures_dir, f'noisy_mimicry_{algo_label}.png')
            plot_noisy_mimicry(all_results[f"{algo_label}_noise"], algo_label, save_path=plot_path_noise)
            
            # Mimicry Cost
            costs = measure_mimicry_costs(game, T_values=T_vals, verbose=False)
            cost_path = os.path.join(figures_dir, f'mimicry_cost_{algo_label}.png')
            plot_mimicry_cost_comparison(costs, game, save_path=cost_path,
                                         algo_label=algo_label)

    # ==========================================
    #  3. Theorem 2: Comparative Statics
    # ==========================================
    print("\n" + "=" * 72)
    print("  EXPERIMENT 3: Theorem 2 (Comparative Statics)")
    print("=" * 72)

    game_q2 = all_results.get('n2_q_learning_game')
    if game_q2:
        pi_c, pi_n, _ = get_equilibrium_profits(game_q2)
        t2 = theorem2_comparative_statics(pi_c, pi_n)
        all_results['theorem2'] = {
            'verified': t2['verified'],
            'part_i': t2['part_i_verified'],
            'part_ii': t2['part_ii_verified'],
        }
        
        if save_plots:
            plot_path = os.path.join(figures_dir, 'theorem2_comparative_statics.png')
            plot_theorem2_comparative_statics(t2, save_path=plot_path)
            
            # visual proof of T* crossing
            p_T_dict = all_results.get('n2_q_learning_pT', {1:0.1, 5:0.5, 10:0.8, 20:0.99})
            cross_path = os.path.join(figures_dir, 'theorem1_t_star_crossing.png')
            plot_T_star_crossing(game_q2, F=1.0, pi_c=pi_c, pi_n=pi_n, 
                                 p_T_dict=p_T_dict, save_path=cross_path)

    # ==========================================
    #  4. Theorem 3: Q-learning vs SARSA
    # ==========================================
    print("\n" + "=" * 72)
    print("  EXPERIMENT 4: Theorem 3 (RL Disruption)")
    print("=" * 72)

    if game_q2:
        game_s2 = all_results.get('n2_sarsa_game')
        t3_T = [1, 5, 10, 20] if quick else [1, 5, 10, 20, 50]
        t3c = theorem3_comparison(game_q2, game_s2,
                                  T_values=t3_T, n_runs=10)
        all_results['theorem3_comparison'] = {
            'T_values': t3_T,
            'q_ratios': [float(r['ratio']) for r in t3c['q_learning']],
            'sarsa_ratios': [float(r['ratio']) for r in t3c['sarsa']],
            'q_recovery': [float(r['mean_recovery_time'])
                           for r in t3c['q_learning']],
            'sarsa_recovery': [float(r['mean_recovery_time'])
                               for r in t3c['sarsa']],
        }
        
        if save_plots:
            plot_path = os.path.join(figures_dir, 'theorem3_q_vs_sarsa.png')
            plot_theorem3_comparison(t3c, save_path=plot_path)

    # ==========================================
    #  5. Adaptive vs Fixed Audit
    # ==========================================
    print("\n" + "=" * 72)
    print("  EXPERIMENT 5: Adaptive vs Fixed Audit")
    print("=" * 72)

    if game_q2:
        adapt_tau = float(audit_tau_by_label.get('n2_q_learning', 0.05))
        adapt = compare_adaptive_vs_fixed(
            game_q2,
            tau_threshold=adapt_tau,
            audit_epsilon=audit_epsilon,
            n_runs=n_mc,
        )
        all_results['adaptive_audit'] = {
            'accuracy': float(adapt['adaptive_accuracy']),
            'accuracy_ci_low': float(adapt.get('adaptive_accuracy_ci_low', 0.0)),
            'accuracy_ci_high': float(adapt.get('adaptive_accuracy_ci_high', 1.0)),
            'mean_T': float(adapt['adaptive_mean_T']),
            'std_T': float(adapt['adaptive_std_T']),
            'mimicry_detection': float(adapt['adaptive_mimicry_detection']),
            'mimicry_detection_ci_low': float(adapt.get('adaptive_mimicry_ci_low', 0.0)),
            'mimicry_detection_ci_high': float(adapt.get('adaptive_mimicry_ci_high', 1.0)),
            'tau_threshold': float(adapt.get('tau_threshold', adapt_tau)),
            'audit_epsilon': float(adapt.get('audit_epsilon', audit_epsilon)),
            'likelihoods': adapt.get('likelihoods', {}),
        }

    # ==========================================
    #  6. Policy Analysis
    # ==========================================
    print("\n" + "=" * 72)
    print("  EXPERIMENT 6: Multi-Jurisdiction Policy")
    print("=" * 72)

    if game_q2:
        pi_c, pi_n, _ = get_equilibrium_profits(game_q2)
        policy = policy_analysis(game_q2, pi_c=pi_c, pi_n=pi_n)
        
        if save_plots:
            plot_path_policy = os.path.join(figures_dir, 'policy_Fmin_vs_international.png')
            plot_policy_analysis(policy, save_path=plot_path_policy)
            
            # Since generating a heatmap requires a grid we already computed previously or can quickly:
            from src.t_star import T_star_grid_search
            grid = T_star_grid_search(game_q2, pi_c=pi_c, pi_n=pi_n, verbose=False)
            plot_path_heatmap = os.path.join(figures_dir, 't_star_heatmap.png')
            plot_T_star_heatmap(grid, save_path=plot_path_heatmap)

    # ==========================================
    #  Save results
    # ==========================================
    all_results['_meta'] = {
        'quick': bool(quick),
        'tmax': int(tmax),
        'n_values': n_values,
        'T_values': T_vals,
        'n_mc': int(n_mc),
        'seed_ci_values': seed_ci_values,
        'audit_ci_seeds': audit_ci_seeds,
        'n_runs_per_seed': int(n_runs_per_seed),
        'audit_detection_metric': audit_detection_metric,
        'audit_min_consecutive_hits': int(audit_min_consecutive_hits),
        'audit_epsilon': float(audit_epsilon),
        'audit_tau_by_label': audit_tau_by_label,
        'audit_tau_calibration_by_label': audit_tau_calibration_by_label,
        'tstar_search_max': int(tstar_search_max),
        'tstar_extrapolation': 'hold_last',
        'timestamp_unix': float(time.time()),
    }

    # Clean for JSON
    clean = {}
    for k, v in all_results.items():
        if isinstance(v, (dict, list, str, int, float, bool)):
            if isinstance(v, dict):
                try:
                    json.dumps(v)
                    clean[k] = v
                except (TypeError, ValueError):
                    pass
            else:
                clean[k] = v

    output_path = os.path.join(results_dir, 'all_results.json')
    with open(output_path, 'w') as f:
        json.dump(clean, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"  ALL EXPERIMENTS COMPLETE  ({elapsed:.0f}s)")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*72}")

    # Summary table
    print(f"\n  {'Label':<20} {'Delta':>8} {'pi_c':>8} {'HHI':>6} {'Conv':>5}")
    print(f"  {'-'*50}")
    for k, v in all_results.items():
        if isinstance(v, dict) and 'delta_val' in v:
            print(f"  {k:<20} {v['delta_val']:>8.4f} "
                  f"{v['pi_c']:>8.4f} {v['hhi']:>6.0f} "
                  f"{'Y' if v['converged'] else 'N':>5}")

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    run_all(quick=args.quick, save_plots=not args.no_plots)
