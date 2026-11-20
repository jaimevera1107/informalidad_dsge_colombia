"""M2: estimated two-sector DSGE for the informal (hidden) economy of Colombia.

Standard real business-cycle core with a formal and an informal sector,
a stochastic (unit-root) labor-augmenting trend and six structural shocks.
Estimated with the Kalman filter and random-walk Metropolis-Hastings.
"""

__all__ = ["data", "model", "statespace", "priors", "estimation", "diagnostics", "informal"]
