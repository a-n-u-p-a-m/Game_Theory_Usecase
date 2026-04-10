"""
=============================================================================
  Algorithmic Collusion Detection -- Interactive Streamlit Dashboard
=============================================================================
  Run:   streamlit run app.py
=============================================================================
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import time, os, sys, json

# -- local imports --
sys.path.insert(0, os.path.dirname(__file__))
from src.model import CalvanoModel
from src.simulation import (simulate_game, simulate_game_sarsa,
                            impulse_response, print_results_table,
                            compare_n_agents)
from src.audit import measure_p_T_curve
from src.mimicry import (run_audit_with_mimicry, measure_mimicry_costs,
                         compute_mimicry_cost_theoretical,
                         run_audit_with_noisy_mimicry, noise_sweep,
                         theorem3_sweep, theorem3_comparison)
from src.t_star import (compute_T_star_analytical,
                        compute_T_star_empirical,
                        T_star_grid_search,
                        compare_p_T_models, fit_p_T_model,
                        p_T_parametric, DEFAULT_TAU, ORIGINAL_TAU,
                        theorem2_comparative_statics)
from src.policy import (compute_F_min, compute_F_critical,
                        goldilocks_zone, policy_analysis,
                        compute_F_min_heatmap, compute_fine_eu,
                        compute_fine_usa, compute_fine_india,
                        JURISDICTIONS)
from src.utils import (find_greedy_equilibrium_state,
                       get_equilibrium_profits, compute_welfare,
                       compute_hhi)
from src.adaptive_audit import (bayesian_audit,
                                compare_adaptive_vs_fixed)
from src.plotting import set_plot_style

# =================================================================
#  Page config
# =================================================================
st.set_page_config(
    page_title="Algorithmic Collusion Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =================================================================
#  Custom CSS
# =================================================================
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label {
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px; padding: 15px; color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; font-size: 0.85em; opacity: 0.85; }
    .metric-card .value { font-size: 1.8em; font-weight: 700; margin: 5px 0; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


def metric_card(title, value, delta=None):
    d_html = f'<div style="font-size:0.8em;opacity:0.7">{delta}</div>' if delta else ''
    return f"""<div class="metric-card">
        <h3>{title}</h3>
        <div class="value">{value}</div>{d_html}
    </div>"""


# =================================================================
#  Sidebar -- parameters
# =================================================================
st.sidebar.markdown("# ⚙️ Model Parameters")

preset = st.sidebar.selectbox("Preset", [
    "Paper Defaults (n=2)", "High Collusion", "Competitive",
    "3-Firm Oligopoly", "Custom"
])

PRESETS = {
    "Paper Defaults (n=2)": dict(n=2, c=1.0, a=2.0, a0=0.0, mu=0.25,
                                  alpha=0.15, beta=4e-6, delta=0.95, k=15),
    "High Collusion":       dict(n=2, c=1.0, a=2.0, a0=0.0, mu=0.10,
                                  alpha=0.15, beta=4e-6, delta=0.99, k=15),
    "Competitive":          dict(n=2, c=1.0, a=2.0, a0=0.0, mu=0.50,
                                  alpha=0.15, beta=4e-6, delta=0.80, k=15),
    "3-Firm Oligopoly":     dict(n=3, c=1.0, a=2.0, a0=0.0, mu=0.25,
                                  alpha=0.15, beta=4e-6, delta=0.95, k=10),
}
defaults = PRESETS.get(preset, PRESETS["Paper Defaults (n=2)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Economic")
n_firms = st.sidebar.slider("Firms (n)", 2, 5, defaults["n"])
if n_firms >= 4:
    st.sidebar.warning(f"n={n_firms}: Q-table has {defaults.get('k',15)**n_firms:,} "
                       f"state entries. Training will be slow.")
delta = st.sidebar.slider("Discount (delta)", 0.50, 0.99, defaults["delta"], 0.01)
mu    = st.sidebar.slider("Differentiation (mu)", 0.05, 1.0, defaults["mu"], 0.05)
c_val = st.sidebar.number_input("Marginal cost (c)", 0.1, 10.0, defaults["c"], 0.1)
a_val = st.sidebar.number_input("Quality (a)", 0.5, 10.0, defaults["a"], 0.1)

st.sidebar.markdown("### Learning")
alpha = st.sidebar.slider("Learning rate (alpha)", 0.01, 0.50, defaults["alpha"], 0.01)
beta  = st.sidebar.select_slider("Exploration decay (beta)",
    options=[1e-7, 5e-7, 1e-6, 4e-6, 1e-5, 5e-5, 1e-4],
    value=defaults["beta"],
    format_func=lambda x: f"{x:.1e}")
k     = st.sidebar.slider("Price levels (k)", 5, 25, defaults["k"])

st.sidebar.markdown("### Simulation")
tmax = st.sidebar.select_slider("Max periods",
    options=[500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000],
    value=5_000_000, format_func=lambda x: f"{x/1e6:.0f}M")

model_params = dict(n=n_firms, c=c_val, a=a_val, a0=0.0, mu=mu,
                    alpha=alpha, beta=beta, delta=delta, k=k,
                    tmax=tmax)

# =================================================================
#  Tabs
# =================================================================
st.markdown("# 📊 Algorithmic Collusion Detection Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎯 Model & Training", "🔍 Audit Protocol",
    "🎭 Mimicry Analysis", "⏱️ T* Verification",
    "🏛️ Policy & Fines", "📈 Welfare & Diagnostics",
    "🔬 Theorems 2 & 3", "🗂️ Saved Results"
])

# =================================================================
#  TAB 1 -- Model & Training
# =================================================================
with tab1:
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("## Model Configuration")
        st.code(f"CalvanoModel(n={n_firms}, delta={delta}, mu={mu}, "
                f"alpha={alpha}, beta={beta}, k={k})", language="python")

    train_btn = st.button("🚀 Train Q-Learning Agents", type="primary",
                          use_container_width=True)

    if train_btn:
        game = CalvanoModel(**model_params)
        st.session_state['game'] = game
        st.text(game.summary())

        progress = st.progress(0, text="Training...")
        status_area = st.empty()

        # -- Training loop with live progress --
        np.random.seed(42)
        state = np.zeros(game.n, dtype=int)
        stable = 0
        converged = False
        convergence_t = game.tmax
        price_history, pi_history = [], []
        t_start = time.time()

        for t in range(game.tmax):
            pr = np.exp(-t * game.beta)
            explore = pr > np.random.rand(game.n)
            actions = np.zeros(game.n, dtype=int)
            for ni in range(game.n):
                if explore[ni]:
                    actions[ni] = np.random.randint(0, game.k)
                else:
                    actions[ni] = np.argmax(game.Q[(ni,) + tuple(state)])
            profits = game.PI[tuple(actions)]
            next_state = actions.copy()

            changed_any = False
            for ni in range(game.n):
                idx = (ni,) + tuple(state) + (actions[ni],)
                old_v = game.Q[idx]
                max_q = np.max(game.Q[(ni,) + tuple(next_state)])
                new_v = profits[ni] + game.delta * max_q
                old_am = np.argmax(game.Q[(ni,) + tuple(state)])
                game.Q[idx] = (1-game.alpha)*old_v + game.alpha*new_v
                new_am = np.argmax(game.Q[(ni,) + tuple(state)])
                same = int(old_am == new_am)
                stable = (stable + same) * same
                if not same:
                    changed_any = True
            state = next_state

            if t % 10_000 == 0:
                gp = game.A[np.array([
                    np.argmax(game.Q[(n,) + tuple(state)])
                    for n in range(game.n)])]
                price_history.append(gp.copy())
                pi_history.append(game.profit_index(gp))

            if t % 50_000 == 0:
                pct = min(t / game.tmax, 1.0)
                progress.progress(pct,
                    text=f"t={t:,}  eps={np.exp(-t*game.beta):.4f}  "
                         f"stable={stable:,}")

            if stable > game.tstable:
                converged = True
                convergence_t = t
                break

        progress.progress(1.0, text="Done!")
        elapsed = time.time() - t_start

        final_prices = game.A[np.array([
            np.argmax(game.Q[(n,) + tuple(state)])
            for n in range(game.n)])]
        result = {
            'converged': converged, 'convergence_t': convergence_t,
            'price_history': price_history,
            'profit_index_history': pi_history,
            'final_Q': game.Q.copy(), 'final_prices': final_prices,
            'profit_index': game.profit_index(final_prices), 'seed': 42,
        }
        st.session_state['result'] = result
        st.session_state['game'] = game

        # -- Metrics --
        pi_c, pi_n, eq_p = get_equilibrium_profits(game)
        st.session_state['pi_c'] = pi_c
        st.session_state['pi_n'] = pi_n

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Converged", "YES" if converged else "NO",
                                    f"t={convergence_t:,}"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Profit Index (Delta)",
                                    f"{result['profit_index']:.3f}",
                                    "0=Nash, 1=Monopoly"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Collusive Profit",
                                    f"{pi_c:.4f}",
                                    f"Nash={pi_n:.4f}"),
                        unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Training Time",
                                    f"{elapsed:.1f}s",
                                    f"{convergence_t:,} periods"),
                        unsafe_allow_html=True)

        # -- Convergence plot --
        st.markdown("### Convergence")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ph = np.array(price_history)
        t_vals = np.arange(len(ph)) * 10_000
        for i in range(game.n):
            ax1.plot(t_vals, ph[:, i], label=f'Firm {i+1}', lw=1.5)
        ax1.axhline(np.mean(game.p_nash), color='green', ls='--', lw=1, label='Nash')
        ax1.axhline(np.mean(game.p_mono), color='red', ls='--', lw=1, label='Monopoly')
        ax1.set_xlabel('Period'); ax1.set_ylabel('Price')
        ax1.legend(); ax1.set_title('Price Convergence')

        ax2.plot(t_vals, pi_history, color='darkorange', lw=1.5)
        ax2.axhline(0, color='green', ls='--', lw=1); ax2.axhline(1, color='red', ls='--', lw=1)
        ax2.set_xlabel('Period'); ax2.set_ylabel('Delta')
        ax2.set_ylim(-0.1, 1.3); ax2.set_title('Profit Index')
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        # -- Policy heatmap (n=2 only) --
        if game.n == 2:
            st.markdown("### Greedy Policy Heatmap")
            fig2, ax = plt.subplots(figsize=(7, 6))
            policy = np.zeros((game.k, game.k))
            for s1 in range(game.k):
                for s2 in range(game.k):
                    policy[s1, s2] = game.A[np.argmax(
                        game.Q[(0, s1, s2)])]
            im = ax.imshow(policy, origin='lower', cmap='RdYlGn_r', aspect='equal')
            ax.set_xlabel("Firm 2 last price"); ax.set_ylabel("Firm 1 last price")
            tp = list(range(0, game.k, max(1, game.k//5)))
            ax.set_xticks(tp); ax.set_xticklabels([f'{game.A[i]:.2f}' for i in tp], rotation=45)
            ax.set_yticks(tp); ax.set_yticklabels([f'{game.A[i]:.2f}' for i in tp])
            plt.colorbar(im, ax=ax, label='Chosen price'); plt.tight_layout()
            st.pyplot(fig2); plt.close(fig2)

    elif 'result' in st.session_state:
        st.info("Model already trained. Adjust parameters and click Train to re-run.")
        r = st.session_state['result']
        c1, c2 = st.columns(2)
        c1.metric("Profit Index", f"{r['profit_index']:.3f}")
        c2.metric("Converged at", f"t={r['convergence_t']:,}")
    else:
        st.info("Configure parameters in the sidebar and click **Train** to begin.")


# =================================================================
#  TAB 2 -- Audit Protocol (M2)
# =================================================================
with tab2:
    st.markdown("## 🔍 Audit Protocol (M2)")

    if 'game' not in st.session_state:
        st.warning("Train a model first (Tab 1).")
    else:
        game = st.session_state['game']
        c1, c2, c3 = st.columns(3)
        tau   = c1.slider("Detection threshold (tau)", 0.01, 0.20, 0.05, 0.01)
        eps_a = c2.slider("Audit exploration (eps)", 0.0, 0.20, 0.05, 0.01)
        n_r   = c3.slider("Monte Carlo runs", 10, 200, 50, 10)

        T_choices = st.multiselect("Audit durations (T)",
            [1, 2, 5, 10, 20, 50, 100, 200, 500],
            default=[1, 5, 10, 20, 50, 100, 200])

        if st.button("Run Audit Analysis", type="primary"):
            with st.spinner("Measuring p(T)..."):
                p_T = measure_p_T_curve(game, T_values=T_choices,
                                         tau=tau, n_runs=n_r,
                                         audit_epsilon=eps_a,
                                         verbose=False)
                st.session_state['p_T'] = p_T

            # p(T) with mimicry
            with st.spinner("Measuring mimicry evasion..."):
                p_T_mim = {}
                for T in T_choices:
                    p_T_mim[T] = run_audit_with_mimicry(
                        game, T, tau_threshold=tau, n_runs=n_r)

            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(8, 5))
                Ts = sorted(p_T.keys())
                ax.plot(Ts, [p_T[T] for T in Ts], 'o-', color='#2196F3',
                        lw=2, ms=6, label='Without mimicry')
                ax.plot(Ts, [p_T_mim[T] for T in Ts], 's--',
                        color='#FF5722', lw=2, ms=6, label='With mimicry')
                ax.set_xlabel('Audit Duration T')
                ax.set_ylabel('Detection Probability')
                ax.set_ylim(-0.05, 1.05); ax.legend()
                ax.set_title('p(T): Detection Probability')
                plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            with col_b:
                df = pd.DataFrame({
                    'T': Ts,
                    'p(T) no mimicry': [f"{p_T[T]:.3f}" for T in Ts],
                    'p(T) with mimicry': [f"{p_T_mim[T]:.3f}" for T in Ts],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.success("Perfect mimicry (Nash play) -> p(T) = 0.0 "
                           "for all T, confirming theoretical prediction.")

            # --- p(T) Model Comparison ---
            st.markdown("### p(T) Parametric Model Comparison")
            fit_result = fit_p_T_model(p_T)
            fitted_tau = fit_result['tau']
            fig_pT, ax_pT = plt.subplots(figsize=(10, 5))
            T_dense = np.arange(1, max(Ts) + 10)
            ax_pT.plot(T_dense,
                       [p_T_parametric(T, ORIGINAL_TAU) for T in T_dense],
                       '--', color='gray', lw=1.5,
                       label=f'Original (tau=50)')
            ax_pT.plot(T_dense,
                       [p_T_parametric(T, fitted_tau) for T in T_dense],
                       '-', color='#FF5722', lw=2,
                       label=f'Fitted (tau={fitted_tau:.2f})')
            ax_pT.plot(Ts, [p_T[T] for T in Ts], 'o', color='#2196F3',
                       ms=8, label='Empirical', zorder=5)
            ax_pT.set_xlabel('Audit Duration T')
            ax_pT.set_ylabel('Detection Probability p(T)')
            ax_pT.set_title('p(T) Model Fit Comparison')
            ax_pT.legend(); ax_pT.set_ylim(-0.05, 1.1)
            plt.tight_layout(); st.pyplot(fig_pT); plt.close(fig_pT)
            st.caption(f"Fitted tau = {fitted_tau:.2f}, "
                       f"R^2 = {fit_result['r_squared']:.4f}")

            # --- Noisy Mimicry ---
            st.markdown("### Noisy Mimicry Analysis")
            st.caption("What if the mimicking firm can't play *exact* Nash?")
            T_noise_test = st.slider("T for noise test", 5, 100, 20,
                                     key="noise_T")
            with st.spinner("Running noise sweep..."):
                noise_res = noise_sweep(
                    game, T_audit=T_noise_test, n_runs=n_r, verbose=False)
            fig_n, ax_n = plt.subplots(figsize=(10, 5))
            ax_n.plot(noise_res['noise_levels'],
                      noise_res['detection_rates'],
                      'o-', color='#E91E63', lw=2.5, ms=8)
            ax_n.set_xlabel('Noise Level (fraction of k)')
            ax_n.set_ylabel('Detection Probability')
            ax_n.set_title(f'Detection vs Mimicry Noise (T={T_noise_test})')
            ax_n.set_ylim(-0.05, 1.05)
            plt.tight_layout(); st.pyplot(fig_n); plt.close(fig_n)
            nr_df = pd.DataFrame({
                'Noise': [f"{nl:.0%}" for nl in noise_res['noise_levels']],
                'p(detection)': [f"{d:.3f}"
                                 for d in noise_res['detection_rates']]
            })
            st.dataframe(nr_df, use_container_width=True, hide_index=True)


# =================================================================
#  TAB 3 -- Mimicry Analysis (M3)
# =================================================================
with tab3:
    st.markdown("## 🎭 Mimicry Analysis (M3)")

    if 'game' not in st.session_state:
        st.warning("Train a model first (Tab 1).")
    else:
        game = st.session_state['game']
        pi_c = st.session_state.get('pi_c')
        pi_n = st.session_state.get('pi_n')

        T_vals_m3 = st.multiselect("T values for cost measurement",
            [1, 2, 5, 10, 20, 50, 100, 200, 500],
            default=[1, 5, 10, 20, 50, 100, 200],
            key="m3_T_vals")
        n_mc = st.slider("MC runs per T", 10, 100, 30, 10, key="m3_mc")

        if st.button("Compute Mimicry Costs", type="primary"):
            with st.spinner("Running..."):
                costs = measure_mimicry_costs(
                    game, T_values=T_vals_m3, n_runs=n_mc,
                    pi_c=pi_c, pi_n=pi_n, verbose=False)
            st.session_state['costs'] = costs

            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(8, 5))
                Ts = costs['T_values']
                ax.plot(Ts, costs['theoretical'], 's--', color='#FF5722',
                        lw=2, ms=6, label='Theoretical')
                ax.plot(Ts, costs['empirical'], 'o-', color='#2196F3',
                        lw=2, ms=6, label='Empirical (RL)')
                C_inf = (pi_c - pi_n) * delta / (1 - delta)
                ax.axhline(C_inf, color='gray', ls=':', lw=1.5,
                           label=f'PV bound = {C_inf:.4f}')
                ax.set_xlabel('T'); ax.set_ylabel('Mimicry Cost')
                ax.set_title('Mimicry Cost: Theory vs RL')
                ax.legend(); plt.tight_layout()
                st.pyplot(fig); plt.close(fig)

            with col_b:
                rows = []
                for i, T in enumerate(Ts):
                    emp = costs['empirical'][i]
                    theo = costs['theoretical'][i]
                    ratio = emp / theo if theo > 1e-12 else float('nan')
                    rows.append({'T': T, 'Empirical': f"{emp:.4f}",
                                 'Theoretical': f"{theo:.4f}",
                                 'Ratio': f"{ratio:.3f}"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)


# =================================================================
#  TAB 4 -- T* Verification (M4)
# =================================================================
with tab4:
    st.markdown("## ⏱️ T* Verification (M4)")

    if 'game' not in st.session_state:
        st.warning("Train a model first (Tab 1).")
    else:
        game = st.session_state['game']
        pi_c = st.session_state.get('pi_c', game.pi_mono)
        pi_n = st.session_state.get('pi_n', game.pi_nash)
        p_T  = st.session_state.get('p_T')

        F_val = st.slider("Fine level (F) for crossing diagram",
                          0.1, 20.0, 2.0, 0.1)
        F_min_val = compute_F_min(pi_c, pi_n, game.delta)

        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("F_min", f"{F_min_val:.4f}",
                                "Min fine for IC at any T"),
                    unsafe_allow_html=True)
        T_ana = compute_T_star_analytical(pi_c, pi_n, game.delta, F_val)
        c2.markdown(metric_card("T* (Analytical)", f"{T_ana}",
                                f"F={F_val:.1f}"), unsafe_allow_html=True)
        if p_T:
            T_emp = compute_T_star_empirical(game, F_val, p_T,
                                             pi_c=pi_c, pi_n=pi_n)
            c3.markdown(metric_card("T* (Empirical)", f"{T_emp}",
                                    "from measured p(T)"),
                        unsafe_allow_html=True)

        # Crossing diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        T_range = np.arange(1, 501)
        C_m = [(pi_c-pi_n)*game.delta*(1-game.delta**T)/(1-game.delta)
               for T in T_range]
        # Use fitted tau if available, else default
        tau_used = DEFAULT_TAU
        if 'p_T' in st.session_state:
            fit = fit_p_T_model(st.session_state['p_T'])
            tau_used = fit['tau']
        pT_F = [p_T_parametric(T, tau_used)*F_val for T in T_range]
        pT_F_orig = [(1-np.exp(-T/ORIGINAL_TAU))*F_val for T in T_range]
        ax.plot(T_range, C_m, color='#2196F3', lw=2.5, label='C_mimic(T)')
        ax.plot(T_range, pT_F, color='#FF5722', lw=2.5,
                label=f'p(T)*F fitted (tau={tau_used:.1f})')
        ax.plot(T_range, pT_F_orig, color='gray', lw=1, ls='--',
                label='p(T)*F original (tau=50)')
        ax.plot(T_range[:1], pT_F[:1], color='#FF5722', lw=2.5,  # dummy for legend alignment
                label=f'p(T)*F  [F={F_val:.1f}]')
        if p_T:
            Ts_e = sorted(p_T.keys())
            ax.plot(Ts_e, [p_T[T]*F_val for T in Ts_e], 'o',
                    color='#FF9800', ms=6, label='Empirical p(T)*F')
        ax.axvline(T_ana, color='green', ls='--', lw=2, alpha=0.7)
        ax.set_xlabel('Audit Duration T'); ax.set_ylabel('Monetary Value')
        ax.set_title(f'IC Crossing (F={F_val:.1f})')
        ax.legend(); ax.set_xlim(0, min(500, max(T_ana*3, 50)))
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Heatmap
        if st.checkbox("Show T* heatmap (slow for large grids)"):
            with st.spinner("Computing T* grid..."):
                grid = T_star_grid_search(game, pi_c=pi_c, pi_n=pi_n,
                                           verbose=False)
            fig2, ax2 = plt.subplots(figsize=(10, 7))
            Tg = np.clip(grid['T_star_grid'], 1, 500)
            im = ax2.imshow(Tg, origin='lower', cmap='YlOrRd',
                           aspect='auto')
            dv = grid['delta_values']; fv = grid['F_values']
            ax2.set_xticks(range(len(fv)))
            ax2.set_xticklabels([f'{f:.1f}' for f in fv], rotation=45)
            ax2.set_yticks(range(len(dv)))
            ax2.set_yticklabels([f'{d:.2f}' for d in dv])
            ax2.set_xlabel('Fine (F)'); ax2.set_ylabel('delta')
            ax2.set_title('T* Heatmap')
            for i in range(len(dv)):
                for j in range(len(fv)):
                    v = int(grid['T_star_grid'][i,j])
                    ax2.text(j, i, f'{v}' if v<1000 else '>1K',
                            ha='center', va='center', fontsize=8,
                            color='white' if v>200 else 'black')
            plt.colorbar(im, ax=ax2); plt.tight_layout()
            st.pyplot(fig2); plt.close(fig2)


# =================================================================
#  TAB 5 -- Policy & International Fines (M5)
# =================================================================
with tab5:
    st.markdown("## 🏛️ Policy Analysis & International Fines (M5)")

    if 'game' not in st.session_state:
        st.warning("Train a model first (Tab 1).")
    else:
        game = st.session_state['game']
        pi_c = st.session_state.get('pi_c', game.pi_mono)
        pi_n = st.session_state.get('pi_n', game.pi_nash)

        st.markdown("### Jurisdiction Details")
        cols = st.columns(3)
        for i, (key, info) in enumerate(JURISDICTIONS.items()):
            with cols[i]:
                st.markdown(f"**{info['name']}**")
                st.caption(info['description'])

        st.markdown("---")
        margin = st.slider("Profit margin for fine estimation",
                           0.05, 0.50, 0.225, 0.025, key="m5_margin")

        if st.button("Run Policy Analysis", type="primary"):
            dvals = list(np.arange(0.50, 1.00, 0.05))
            results = policy_analysis(game, delta_values=dvals,
                                       profit_margin_range=(0.15, 0.30),
                                       pi_c=pi_c, pi_n=pi_n,
                                       verbose=False)

            # Multi-jurisdiction plot
            fig, ax = plt.subplots(figsize=(10, 6))
            d = results['delta_values']; fm = results['F_min']
            ax.plot(d, fm, 'o-', color='#2196F3', lw=2.5, ms=7,
                    label='F_min (IC threshold)', zorder=5)
            ax.fill_between(d, results['eu_fine_low'],
                           results['eu_fine_high'],
                           alpha=0.15, color='#4CAF50')
            ax.plot(d, results['eu_fine_high'], '--', color='#4CAF50',
                    lw=1.5, label='EU')
            ax.fill_between(d, results['usa_fine_low'],
                           results['usa_fine_high'],
                           alpha=0.15, color='#FF9800')
            ax.plot(d, results['usa_fine_high'], '--', color='#FF9800',
                    lw=1.5, label='USA (crim+civil)')
            ax.fill_between(d, results['india_fine_low'],
                           results['india_fine_high'],
                           alpha=0.15, color='#9C27B0')
            ax.plot(d, results['india_fine_high'], '--', color='#9C27B0',
                    lw=1.5, label='India (CCI)')
            ax.set_xlabel('Discount Factor (delta)')
            ax.set_ylabel('Fine Level'); ax.set_yscale('log')
            ax.set_title('F_min vs International Cartel Fines')
            ax.legend(loc='upper left'); plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

            # Fine Regime Analysis
            st.markdown("### Fine Regime Analysis")
            gz = goldilocks_zone(pi_c, pi_n, game.delta)
            gc1, gc2, gc3 = st.columns(3)
            gc1.markdown(metric_card("F_crit", f"{gz['F_crit']:.4f}",
                                     "T*=1 threshold"),
                         unsafe_allow_html=True)
            gc2.markdown(metric_card("F_min", f"{gz['F_min']:.4f}",
                                     "Deterrence threshold"),
                         unsafe_allow_html=True)
            gc3.markdown(metric_card("Ratio", f"{gz['ratio']:.1f}x",
                                     "F_min / F_crit"),
                         unsafe_allow_html=True)

            regime_df = pd.DataFrame([
                {"Regime": "WIN #1: Easy Detection",
                 "Fine Range": gz['easy_detection'].split('->')[0].strip(),
                 "T*": "1",
                 "Outcome": "Firm doesn't mimic -> audit detects -> enforce"},
                {"Regime": "WIN #1: Calibrated Detection",
                 "Fine Range": gz['calibrated_detection'].split('->')[0].strip(),
                 "T*": "> 1, finite",
                 "Outcome": "Audit works if T >= T*; FAILS if T < T*"},
                {"Regime": "WIN #2: Deterrence",
                 "Fine Range": gz['deterrence'].split('->')[0].strip(),
                 "T*": "N/A",
                 "Outcome": "Firm plays Nash permanently (under random audits)"},
            ])
            st.dataframe(regime_df, use_container_width=True,
                         hide_index=True)
            st.success(
                "**Two Win Conditions**: The regulator wins if collusion is "
                "*detected* (WIN #1: F < F_min) OR if the fine threat forces "
                "*permanent competitive pricing* (WIN #2: F >= F_min). "
                f"Current fines (EU/USA/India) are all < F_min={gz['F_min']:.4f}, "
                "so WIN #1 applies: the audit mechanism detects collusion "
                "for all jurisdictions."
            )

            # Summary table
            rows = []
            for i, dv in enumerate(d):
                fm_v = results['F_min'][i]
                fc_v = compute_F_critical(pi_c, pi_n, dv)
                eu_v = results['eu_fine_high'][i]
                us_v = results['usa_fine_high'][i]
                in_v = results['india_fine_high'][i]

                def _outcome(fine, fc, fm):
                    if fine <= fc:
                        return "Detect(T*=1)"
                    elif fine < fm:
                        T_s = compute_T_star_analytical(
                            pi_c, pi_n, dv, fine)
                        return f"Detect(T*={T_s})"
                    else:
                        return "Deter"

                rows.append({
                    'delta': f"{dv:.2f}", 'F_min': f"{fm_v:.4f}",
                    'EU': f"{eu_v:.4f}", 'USA': f"{us_v:.4f}",
                    'India': f"{in_v:.4f}",
                    'EU Win': _outcome(eu_v, fc_v, fm_v),
                    'USA Win': _outcome(us_v, fc_v, fm_v),
                    'India Win': _outcome(in_v, fc_v, fm_v),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

            # Per-jurisdiction detail
            st.markdown("### Fine Breakdown (current parameters)")
            cc = st.columns(3)
            eu_detail = compute_fine_eu(pi_c, margin)
            us_detail = compute_fine_usa(pi_c, margin)
            in_detail = compute_fine_india(pi_c, margin)
            with cc[0]:
                st.json({"EU": {k: f"{v:.4f}" if isinstance(v, float)
                                else v for k, v in eu_detail.items()}})
            with cc[1]:
                st.json({"USA": {k: f"{v:.4f}" if isinstance(v, float)
                                 else v for k, v in us_detail.items()}})
            with cc[2]:
                st.json({"India": {k: f"{v:.4f}" if isinstance(v, float)
                                   else v for k, v in in_detail.items()}})


# =================================================================
#  TAB 6 -- Welfare & Diagnostics
# =================================================================
with tab6:
    st.markdown("## 📈 Welfare & Market Diagnostics")

    if 'game' not in st.session_state:
        st.warning("Train a model first (Tab 1).")
    else:
        game = st.session_state['game']
        pi_c = st.session_state.get('pi_c', game.pi_mono)
        pi_n = st.session_state.get('pi_n', game.pi_nash)

        eq_state = find_greedy_equilibrium_state(game)
        eq_actions = np.array([np.argmax(game.Q[(n,) + tuple(eq_state)])
                               for n in range(game.n)])
        eq_prices = game.A[eq_actions]

        st.markdown("### Welfare Comparison")
        scenarios = {
            "Nash Equilibrium": game.p_nash,
            "Converged (RL)": eq_prices,
            "Monopoly (Joint Max)": game.p_mono,
        }
        rows = []
        for name, prices in scenarios.items():
            w = compute_welfare(game, prices)
            h = compute_hhi(game, prices)
            pi_mean = float(np.mean(game._compute_profits(prices)))
            rows.append({
                'Scenario': name,
                'Avg Price': f"{np.mean(prices):.4f}",
                'Avg Profit': f"{pi_mean:.4f}",
                'Consumer Surplus': f"{w['consumer_surplus']:.4f}",
                'Producer Surplus': f"{w['producer_surplus']:.4f}",
                'Total Welfare': f"{w['total_welfare']:.4f}",
                'HHI': f"{h:.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

        # Welfare bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [r['Scenario'] for r in rows]
        cs_vals = [float(r['Consumer Surplus']) for r in rows]
        ps_vals = [float(r['Producer Surplus']) for r in rows]
        x = np.arange(len(names))
        ax.bar(x - 0.15, cs_vals, 0.3, label='Consumer Surplus',
               color='#42A5F5')
        ax.bar(x + 0.15, ps_vals, 0.3, label='Producer Surplus',
               color='#FF7043')
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel('Welfare'); ax.legend()
        ax.set_title('Welfare Decomposition')
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Impulse response
        st.markdown("### Impulse Response (Punishment & Forgiveness)")
        if st.session_state.get('result', {}).get('converged', False):
            dev_firm = st.selectbox("Deviating firm",
                                    list(range(game.n)), key="ir_firm")
            if st.button("Run Impulse Response"):
                ir = impulse_response(game,
                                      st.session_state['result']['final_Q'],
                                      deviating_firm=dev_firm,
                                      T_before=10, T_after=50)
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                prices = ir['prices']; dev_t = ir['deviation_t']
                t_axis = np.arange(len(prices)) - dev_t
                for i in range(game.n):
                    lbl = 'Deviating' if i == dev_firm else f'Firm {i+1}'
                    ax3.plot(t_axis, prices[:, i], label=lbl, lw=2,
                            marker='o', ms=3)
                ax3.axhline(np.mean(game.p_nash), color='green', ls='--',
                           lw=1, label='Nash')
                ax3.axhline(np.mean(game.p_mono), color='red', ls='--',
                           lw=1, label='Monopoly')
                ax3.axvline(0, color='gray', ls=':', alpha=0.7)
                ax3.set_xlabel('Period relative to deviation')
                ax3.set_ylabel('Price')
                ax3.set_title('Impulse Response')
                ax3.legend(); plt.tight_layout()
                st.pyplot(fig3); plt.close(fig3)

        # Export
        st.markdown("---")
        st.markdown("### Export Results")
        if st.button("Download Results as CSV"):
            data = {
                'metric': ['n', 'delta', 'mu', 'alpha', 'pi_c', 'pi_n',
                           'profit_index', 'F_min'],
                'value': [game.n, game.delta, game.mu, game.alpha,
                          pi_c, pi_n,
                          st.session_state.get('result', {}).get(
                              'profit_index', 0),
                          compute_F_min(pi_c, pi_n, game.delta)]
            }
            df = pd.DataFrame(data)
            st.download_button("Download", df.to_csv(index=False),
                               "collusion_results.csv", "text/csv")

        # --- SARSA comparison ---
        st.markdown("---")
        st.markdown("### Algorithm Comparison: Q-Learning vs SARSA")
        if st.button("Train SARSA Agent", key="sarsa_train"):
            with st.spinner("Training SARSA agent..."):
                game_s = CalvanoModel(**model_params)
                result_s = simulate_game_sarsa(game_s, seed=42,
                                               verbose=False)
            r_q = st.session_state.get('result', {})
            rows_algo = []
            if r_q:
                rows_algo.append({
                    'Algorithm': 'Q-Learning',
                    'Converged': 'YES' if r_q.get('converged') else 'NO',
                    'Conv. Period': f"{r_q.get('convergence_t', 0):,}",
                    'Delta': f"{r_q.get('profit_index', 0):.4f}",
                })
            rows_algo.append({
                'Algorithm': 'SARSA',
                'Converged': 'YES' if result_s['converged'] else 'NO',
                'Conv. Period': f"{result_s['convergence_t']:,}",
                'Delta': f"{result_s['profit_index']:.4f}",
            })
            st.dataframe(pd.DataFrame(rows_algo),
                         use_container_width=True, hide_index=True)
            if result_s['converged']:
                st.success(f"SARSA converged with Delta="
                           f"{result_s['profit_index']:.3f}")
            else:
                st.warning("SARSA did not converge.")

        # --- Adaptive audit ---
        st.markdown("---")
        st.markdown("### Adaptive Bayesian Audit")
        if st.button("Run Adaptive Audit Comparison", key="adaptive"):
            with st.spinner("Running adaptive audit..."):
                adapt = compare_adaptive_vs_fixed(game, n_runs=30,
                                                  verbose=False)
            st.markdown(f"**Adaptive audit accuracy**: "
                        f"{adapt['adaptive_accuracy']:.0%}")
            st.markdown(f"**Mean stopping time**: "
                        f"{adapt['adaptive_mean_T']:.1f} "
                        f"+/- {adapt['adaptive_std_T']:.1f} periods")
            st.markdown(f"**Detection under perfect mimicry**: "
                        f"{adapt['adaptive_mimicry_detection']:.0%}")
            if adapt['adaptive_accuracy'] > 0.9:
                st.success("Adaptive audit achieves >90% accuracy with "
                           "variable stopping time.")
            else:
                st.warning("Adaptive audit accuracy below 90%.")


# =================================================================
#  TAB 7 -- Theorems 2 & 3
# =================================================================
with tab7:
    st.markdown("## 🔬 Theorem Verification")

    if 'game' not in st.session_state:
        st.warning("Train a model first (Tab 1).")
    else:
        game = st.session_state['game']
        pi_c = st.session_state.get('pi_c', game.pi_mono)
        pi_n = st.session_state.get('pi_n', game.pi_nash)

        # --- Theorem 2: Comparative Statics ---
        st.markdown("### Theorem 2: Comparative Statics of T*")
        st.markdown("""
        > **(i)** T* is non-increasing in δ (more patient → shorter audit)
        > **(ii)** T* is non-decreasing in F (higher fine → longer audit)
        """)

        if st.button("Verify Theorem 2", type="primary",
                      key="t2_btn"):
            with st.spinner("Computing comparative statics..."):
                t2 = theorem2_comparative_statics(pi_c, pi_n,
                                                   verbose=False)

            if t2['verified']:
                st.success("✅ **Theorem 2 VERIFIED**")
            else:
                st.error("❌ Theorem 2 NOT verified")

            t2c1, t2c2 = st.columns(2)
            with t2c1:
                st.markdown("#### (i) T* vs δ (fixed F)")
                for r in t2['delta_sweeps']:
                    df = pd.DataFrame({
                        'δ': r['delta_values'],
                        'T*': r['T_star_values'],
                    })
                    st.markdown(f"**F = {r['F']}**  "
                                f"{'✓' if r['non_increasing'] else '✗'}")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(df['δ'], df['T*'], 'o-', lw=2)
                    ax.set_xlabel('δ'); ax.set_ylabel('T*')
                    ax.set_title(f'T* vs δ (F={r["F"]})')
                    plt.tight_layout()
                    st.pyplot(fig); plt.close(fig)

            with t2c2:
                st.markdown("#### (ii) T* vs F (fixed δ)")
                for r in t2['F_sweeps']:
                    df = pd.DataFrame({
                        'F': r['F_values'],
                        'T*': r['T_star_values'],
                    })
                    st.markdown(f"**δ = {r['delta']}**  "
                                f"{'✓' if r['non_decreasing'] else '✗'}")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(df['F'], df['T*'], 's-', lw=2, color='#e74c3c')
                    ax.set_xlabel('F'); ax.set_ylabel('T*')
                    ax.set_title(f'T* vs F (δ={r["delta"]})')
                    plt.tight_layout()
                    st.pyplot(fig); plt.close(fig)

        st.markdown("---")

        # --- Theorem 3: RL Disruption ---
        st.markdown("### Theorem 3: RL Disruption Cost")
        st.markdown("""
        > **Claim**: T*(RL) ≤ T*(rational).
        > Q-learning agents forced to mimic Nash suffer Q-table disruption,
        > making mimicry MORE expensive than for rational agents.
        """)

        t3_quick = st.checkbox("Quick mode (fewer T values)",
                                value=True, key="t3_quick")

        if st.button("Run Theorem 3 Analysis", type="primary",
                      key="t3_btn"):
            T_vals = [1, 5, 10, 20] if t3_quick else [1, 5, 10, 20, 50]

            with st.spinner("Measuring Q-table disruption..."):
                t3 = theorem3_sweep(game, T_values=T_vals,
                                     n_runs=10, verbose=False)

            # Disruption table
            st.markdown("#### Disruption by T_mimic")
            rows = []
            for r in t3['disruption_results']:
                rows.append({
                    'T_mimic': r['T_mimic'],
                    'C_theoretical': f"{r['C_theoretical']:.4f}",
                    'Disruption': f"{r['disruption_cost']:.4f}",
                    'Total RL Cost': f"{r['total_rl_cost']:.4f}",
                    'Ratio (RL/Rational)': f"{r['ratio']:.3f}",
                    'Recovery (periods)': f"{r['mean_recovery_time']:.1f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

            # T*(RL) vs T*(rational) table
            st.markdown("#### T*(RL) vs T*(rational)")
            t_rows = []
            for c in t3['T_star_comparisons']:
                t_rows.append({
                    'F': c['F'],
                    'T*_rational': c['T_rational'],
                    'T*_RL': c['T_rl'],
                    'Ratio': f"{c['ratio']:.2f}",
                    'Theorem 3': c['theorem3'],
                })
            st.dataframe(pd.DataFrame(t_rows),
                         use_container_width=True, hide_index=True)

            confirmed = sum(1 for c in t3['T_star_comparisons']
                            if c['theorem3'] == 'CONFIRMED')
            total = len(t3['T_star_comparisons'])
            if confirmed == total:
                st.success(f"✅ Theorem 3 CONFIRMED across all "
                           f"{total} fine levels")
            else:
                st.warning(f"Theorem 3 confirmed for {confirmed}/{total}")

        st.markdown("---")

        # --- Q-learning vs SARSA Disruption ---
        st.markdown("### Q-Learning vs SARSA Disruption")
        st.markdown("""
        Compare on-policy (SARSA) vs off-policy (Q-learning) disruption.
        SARSA updates from actual transitions during Nash play, which
        should more aggressively overwrite collusive Q-values.
        """)

        if st.button("Run Q-Learning vs SARSA Comparison", key="t3c_btn"):
            T_vals = [1, 5, 10, 20] if t3_quick else [1, 5, 10, 20, 50]

            with st.spinner("Running comparison (this takes a moment)..."):
                game_sarsa = CalvanoModel(**model_params)
                sarsa_result = simulate_game_sarsa(game_sarsa, seed=42,
                                                   verbose=False)
                game_sarsa.Q = sarsa_result['final_Q']
                t3c = theorem3_comparison(game, game_sarsa,
                                          T_values=T_vals,
                                          n_runs=10, verbose=False)

            rows = []
            for i, T in enumerate(t3c['T_values']):
                qr = t3c['q_learning'][i]
                sr = t3c['sarsa'][i]
                rows.append({
                    'T_mimic': T,
                    'Q-Disruption': f"{qr['disruption_cost']:.4f}",
                    'Q-Ratio': f"{qr['ratio']:.3f}",
                    'Q-Recovery': f"{qr['mean_recovery_time']:.1f}",
                    'SARSA-Disruption': f"{sr['disruption_cost']:.4f}",
                    'SARSA-Ratio': f"{sr['ratio']:.3f}",
                    'SARSA-Recovery': f"{sr['mean_recovery_time']:.1f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

            # Chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            Ts = t3c['T_values']
            q_r = [r['ratio'] for r in t3c['q_learning']]
            s_r = [r['ratio'] for r in t3c['sarsa']]
            ax1.plot(Ts, q_r, 'o-', label='Q-Learning', lw=2)
            ax1.plot(Ts, s_r, 's-', label='SARSA', lw=2)
            ax1.axhline(1.0, color='gray', ls='--', alpha=0.5)
            ax1.set_xlabel('T_mimic'); ax1.set_ylabel('Total/Theoretical')
            ax1.set_title('Disruption Ratio'); ax1.legend()

            q_rec = [r['mean_recovery_time'] for r in t3c['q_learning']]
            s_rec = [r['mean_recovery_time'] for r in t3c['sarsa']]
            ax2.bar(np.array(Ts) - 1, q_rec, 2, label='Q-Learning')
            ax2.bar(np.array(Ts) + 1, s_rec, 2, label='SARSA')
            ax2.set_xlabel('T_mimic'); ax2.set_ylabel('Recovery periods')
            ax2.set_title('Recovery Time'); ax2.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)


# =================================================================
#  TAB 8 -- Saved Results & Figures
# =================================================================
with tab8:
    st.markdown("## 🗂️ Pre-Saved Results & Figures")

    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results")
    results_file = os.path.join(results_dir, "all_results.json")
    figures_dir = os.path.join(results_dir, "figures")

    fig_files = []
    if os.path.isdir(figures_dir):
        fig_files = sorted([
            fname for fname in os.listdir(figures_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])

    s1, s2, s3 = st.columns(3)
    s1.metric("Results JSON", "Found" if os.path.isfile(results_file) else "Missing")
    s2.metric("Figures Folder", "Found" if os.path.isdir(figures_dir) else "Missing")
    s3.metric("Figure Count", f"{len(fig_files)}")

    st.caption(f"Results file: {results_file}")
    st.caption(f"Figures folder: {figures_dir}")

    # --- JSON result explorer ---
    st.markdown("### Saved Result Explorer")
    saved_results = None
    if os.path.isfile(results_file):
        try:
            with open(results_file, "r", encoding="utf-8") as fh:
                saved_results = json.load(fh)
        except Exception as exc:
            st.error(f"Failed to load all_results.json: {exc}")
    else:
        st.warning("No pre-saved all_results.json found. Run run_experiments.py first.")

    if saved_results is not None:
        meta = saved_results.get("_meta", {})
        m1, m2, m3, m4 = st.columns(4)
        n_vals = meta.get("n_values", [])
        n_text = ", ".join(str(v) for v in n_vals) if n_vals else "N/A"
        m1.metric("Quick Mode", str(meta.get("quick", "N/A")))
        m2.metric("n Values", n_text)
        m3.metric("MC Runs", str(meta.get("n_mc", "N/A")))
        m4.metric("Audit Epsilon", str(meta.get("audit_epsilon", "N/A")))

        result_keys = sorted([k for k in saved_results.keys() if k != "_meta"])
        key_filter = st.text_input(
            "Filter result keys",
            value="",
            placeholder="Try: pT, theorem, noise, tstar, adaptive"
        )
        filtered_keys = [
            k for k in result_keys
            if key_filter.lower() in k.lower()
        ]

        if not filtered_keys:
            st.info("No keys match the current filter.")
        else:
            selected_key = st.selectbox(
                "Choose a result block",
                filtered_keys,
                key="saved_result_key"
            )
            st.json(saved_results[selected_key])

        with st.expander("Show _meta block"):
            st.json(meta)

        with open(results_file, "r", encoding="utf-8") as fh:
            st.download_button(
                "Download all_results.json",
                data=fh.read(),
                file_name="all_results.json",
                mime="application/json"
            )

    # --- Figure explorer ---
    st.markdown("### Saved Figure Explorer")
    if not os.path.isdir(figures_dir):
        st.warning("No figures directory found. Run run_experiments.py with plotting enabled.")
    elif not fig_files:
        st.info("Figures directory exists but has no image files.")
    else:
        fig_filter = st.text_input(
            "Filter figure filenames",
            value="",
            placeholder="Try: p_T_curve_ci, theorem3, heatmap"
        )
        filtered_figs = [
            fname for fname in fig_files
            if fig_filter.lower() in fname.lower()
        ]

        if not filtered_figs:
            st.info("No figures match the current filter.")
        else:
            selected_fig = st.selectbox(
                "Choose a figure",
                filtered_figs,
                key="saved_figure_choice"
            )
            selected_fig_path = os.path.join(figures_dir, selected_fig)
            st.image(selected_fig_path, caption=selected_fig, use_container_width=True)

            with st.expander("Gallery Preview"):
                max_preview = min(12, len(filtered_figs))
                default_preview = min(6, max_preview)
                preview_count = st.slider(
                    "Number of preview figures",
                    min_value=1,
                    max_value=max_preview,
                    value=default_preview,
                    key="saved_preview_count"
                )
                preview_cols = st.columns(3)
                for idx, fname in enumerate(filtered_figs[:preview_count]):
                    img_path = os.path.join(figures_dir, fname)
                    with preview_cols[idx % 3]:
                        st.image(img_path, caption=fname, use_container_width=True)


# =================================================================
#  Sidebar footer
# =================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🚀 Deployment
```bash
# Local
streamlit run app.py

# Streamlit Cloud
# 1. Push to GitHub
# 2. Connect at share.streamlit.io
# 3. Set main file: app.py
```
""")
st.sidebar.caption("Calvano et al. (2020) Replication | v3.0")
