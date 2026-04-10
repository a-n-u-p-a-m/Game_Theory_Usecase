# Game Your Brains Out- Game Theory Usecase

## Algorithmic Collusion Detection: Designing Incentive-Compatible Audit Mechanisms

### Overview
Building upon the framework of [Calvano et al. (2020)](https://www.aeaweb.org/articles?id=10.1257/aer.20190623), this project investigates how independent reinforcement learning pricing agents (using Q-learning and SARSA) independently discover and sustain collusive equilibria in Bertrand markets with no explicit communication channels. 

Crucially, the repository reframes collusion detection as a mechanism design problem. Rather than simply looking for suspicious patterns, the regulator (designer) creates an audit protocol $\mathcal{M} = (T, \tau, F)$ that makes it incentive-compatible for algorithmic agents to reveal their true nature, where maintaining a collusive pricing strategy during an active audit becomes the weakly dominant strategy.

### Repository Structure

The repository focuses heavily on simulation execution and visual reporting, structured as follows:

```text
/
├── app.py                     # Streamlit dashboard for interactive exploration
├── audit_mechanism.py         # Logic for threshold tau and optimal audit duration T*
├── calvano_replication.py     # Script to replicate the foundational Calvano et al. findings
├── run_experiments.py         # Main execution script for all training and audit benchmarks
├── requirements.txt           # Python dependencies
├── src/                       # Core python modules
│   ├── adaptive_audit.py      # Adaptive auditing mechanics
│   ├── audit.py               # Audit injection and tracking parameters
│   ├── mimicry.py             # Logic for competitive behavior evasion tactics
│   ├── model.py               # Bertrand duopoly environment definitions
│   ├── plotting.py            # Matplotlib/Seaborn visualization scripts
│   ├── policy.py              # Epsilon-greedy and exploration mechanisms
│   ├── simulation.py          # Training loops for Q-Learning and SARSA agents
│   └── t_star.py              # Theoretical derivation implementations for minimal T
└── results/
    ├── all_results.json       # Compiled data points from simulation runs
    └── figures/               # Output graphics (convergence_n2_q_learning.png, etc.)
```

### Methodology

1. **Strategic Pricing Context:** Firms use Q-Learning/SARSA in a simulated repeated Bertrand oligopoly. Through long-run market responses, the algorithms settle on supracompetitive prices, confirming the viability of purely algorithmic collusion.
2. **Audit Intervention:** The mechanism injects a synthetic competitive entrant pricing at marginal cost ($p = c$) during an audit phase lasting $T$ rounds.
3. **Detection via Mechanism Design:** The regulator observes the price responses $\{p_i(t)\}_{t=1}^T$ of the existing firm $i$. The firm is flagged if its deviation from competitive pricing exceeds a threshold $\tau$. An expected fine $F$ is imposed.
4. **Theoretical Minimum $T^*$:** At a calculated audit duration threshold $T^*$, the cost to the collusive firm of mimicking a competitive entity (deliberately dropping its prices to evade the audit) outweighs the expected fine. Hence, the system guarantees an incentive-compatible truth revelation from the algorithm itself.

### Results

1. **Algorithmic Convergence:** The `results/figures/` directory visualizes agents consistently reaching joint-monopoly-level pricing levels independently, replicating the core problem.
2. **Effective Auditing:** The experiments confirm the theoretical presence of $T^*$. For audits extending beyond $T^*$, algorithmic firms mathematically cannot mimic competitive behavior profitably, forcing them into a strict dilemma where they either reveal their supracompetitive logic and are fined, or indefinitely maintain low competitive pricing—which is exactly what the regulator wants.
3. Detailed structural results output to `results/all_results.json` and graphical representations (like $p_T$ curves mapping defection strategies) are generated directly into the `figures` subfolder.

### How to Run the Experiments

Experiments will train the agents, apply the audit mechanisms, and generate the plots automatically.

1. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the experiment suite to generate data arrays and visualizations:
   ```bash
   python run_experiments.py
   ```

*(You can also use `python calvano_replication.py` if you specifically want to run the baseline Q-Learning convergence independently).*

### How to Run the App

The project includes an interactive web dashboard to explore the collision dynamics and audit responses dynamically, rather than statically via the raw scripts.

1. Ensure the dependencies from `requirements.txt` are installed.
2. Download the repository and launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```
3. A local server will start, mapping to your browser (default: `http://localhost:8501`). Use the UI pane provided to dynamically tweak learning rates, discount factors, the fine size ($F$), and the audit rounds ($T$).
