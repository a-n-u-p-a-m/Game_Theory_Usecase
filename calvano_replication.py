"""
=============================================================================
CALVANO ET AL. (2020) -- CLI WRAPPER
=============================================================================
Thin wrapper preserving backward-compatible CLI.
All logic lives in src/.

Usage:
    python calvano_replication.py              # run baseline
    python calvano_replication.py --seeds 10   # 10 seeds
    python calvano_replication.py --help
=============================================================================
"""

import argparse
import numpy as np
import os
import sys

from src.model import CalvanoModel
from src.simulation import (simulate_game, impulse_response,
                            run_multi_seed, parameter_sweep,
                            print_results_table)
from src.plotting import (plot_convergence, plot_impulse_response,
                          plot_profit_index_distribution,
                          plot_greedy_policy_heatmap,
                          plot_parameter_sweep)


def main():
    parser = argparse.ArgumentParser(
        description="Calvano et al. (2020) -- Python Replication",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--n', type=int, default=2,
                        help='Number of firms (default: 2)')
    parser.add_argument('--delta', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--beta', type=float, default=4e-6)
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--mu', type=float, default=0.25)
    parser.add_argument('--tmax', type=int, default=int(1e7))
    parser.add_argument('--impulse', action='store_true')
    parser.add_argument('--sweep', type=str, default=None,
                        choices=['delta', 'alpha', 'mu'])
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()

    model_kwargs = dict(
        n=args.n, delta=args.delta, alpha=args.alpha,
        beta=args.beta, k=args.k, mu=args.mu, tmax=args.tmax,
    )

    game = CalvanoModel(**model_kwargs)
    print(game.summary())

    # -- Parameter sweep mode --
    if args.sweep:
        sweep_ranges = {
            'delta': np.arange(0.5, 1.0, 0.05),
            'alpha': np.arange(0.025, 0.30, 0.025),
            'mu':    np.arange(0.1, 1.0, 0.1),
        }
        print(f"\n  Running parameter sweep over '{args.sweep}'...")
        sweep = parameter_sweep(
            args.sweep, list(sweep_ranges[args.sweep]),
            n_seeds=args.seeds, verbose=False, **model_kwargs,
        )
        if not args.no_plot:
            sp = None
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                sp = os.path.join(args.save_dir, f'sweep_{args.sweep}.png')
            plot_parameter_sweep(sweep, save_path=sp)
        return

    # -- Standard run --
    if args.seeds == 1:
        result = simulate_game(game, seed=42, verbose=True)
        results = [result]
        result['game'] = game
    else:
        results = run_multi_seed(n_seeds=args.seeds, verbose=True,
                                 **model_kwargs)

    print_results_table(results, game)

    # -- Plots --
    if not args.no_plot:
        sd = args.save_dir
        if sd:
            os.makedirs(sd, exist_ok=True)
        plot_convergence(
            results[0], results[0].get('game', game),
            save_path=os.path.join(sd, 'convergence.png') if sd else None,
        )
        g = results[0].get('game', game)
        g.Q = results[0]['final_Q']
        plot_greedy_policy_heatmap(
            g, agent_idx=0,
            save_path=os.path.join(sd, 'policy_firm1.png') if sd else None,
        )
        if len(results) > 1:
            plot_profit_index_distribution(
                results,
                save_path=os.path.join(sd, 'delta_dist.png') if sd else None,
            )

    # -- Impulse response --
    if args.impulse and results[0]['converged']:
        print("\n  Running impulse-response analysis...")
        g = results[0].get('game', game)
        ir = impulse_response(g, results[0]['final_Q'],
                              T_before=10, T_after=50)
        if not args.no_plot:
            plot_impulse_response(
                ir, g,
                save_path=(os.path.join(sd, 'impulse_response.png')
                           if sd else None),
            )
        eq_price = ir['prices'][0, 0]
        min_price = np.min(ir['prices'][ir['deviation_t']:, :])
        recovery_t = None
        for t in range(ir['deviation_t'] + 1, len(ir['prices'])):
            if np.allclose(ir['prices'][t], ir['prices'][0], atol=0.01):
                recovery_t = t - ir['deviation_t']
                break
        print(f"  Equilibrium price : {eq_price:.4f}")
        print(f"  Minimum after dev : {min_price:.4f}")
        print(f"  Recovery period   : "
              f"{recovery_t if recovery_t else '>50'} periods")


if __name__ == '__main__':
    main()
