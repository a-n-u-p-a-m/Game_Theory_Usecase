"""
Algorithmic Collusion Detection -- modular source package.

Submodules:
    model           CalvanoModel (n-agent Bertrand with logit demand)
    simulation      Q-learning & SARSA engines, multi-seed, n-agent comparison
    audit           M2: audit protocol, p(T) measurement
    mimicry         M3: mimicry agent, cost measurement, noisy mimicry
    t_star          M4: T* computation, p(T) model fitting & comparison
    policy          M5: F_min, Goldilocks zone, EU/USA/India fines
    adaptive_audit  M6: Bayesian sequential audit
    plotting        All matplotlib figures
    utils           Shared helpers (equilibrium, welfare, HHI)
"""
__version__ = "3.0.0"

