"""MCMC convergence diagnostics: rank-normalized split R-hat and ESS.

Implements Vehtari, Gelman, Simpson, Carpenter and Buerkner (2021),
"Rank-normalization, folding, and localization", Bayesian Analysis.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def _split(chains: np.ndarray) -> np.ndarray:
    n = chains.shape[1] // 2
    return np.vstack((chains[:, :n], chains[:, n:2 * n]))


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    ranks = rankdata(chains, method="average").reshape(chains.shape)
    return norm.ppf((ranks - 0.375) / (chains.size + 0.25))


def _rhat(chains: np.ndarray) -> float:
    m, n = chains.shape
    means = chains.mean(axis=1)
    w = chains.var(axis=1, ddof=1).mean()
    b = n * means.var(ddof=1)
    if w <= 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def _autocov(x: np.ndarray) -> np.ndarray:
    n = len(x)
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x - x.mean(), size)
    acov = np.fft.irfft(f * np.conjugate(f))[:n] / n
    return acov


def _ess(chains: np.ndarray) -> float:
    m, n = chains.shape
    acov = np.array([_autocov(c) for c in chains])
    chain_var = acov[:, 0] * n / (n - 1)
    mean_var = chain_var.mean()
    var_plus = mean_var * (n - 1) / n + (chains.mean(axis=1).var(ddof=1) if m > 1 else 0.0)
    if var_plus <= 0:
        return float("nan")
    rho = 1.0 - (mean_var - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer initial monotone sequence on pair sums
    tau = -1.0
    prev = np.inf
    for t in range(0, n - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev)
        prev = pair
        tau += 2 * pair
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def summarize(chains: np.ndarray) -> dict[str, float]:
    """chains: array (n_chains, n_draws) for one parameter."""
    split = _split(chains)
    z = _rank_normalize(split)
    folded = _rank_normalize(np.abs(split - np.median(split)))
    rhat = max(_rhat(z), _rhat(folded))
    lo, hi = np.quantile(split, [0.05, 0.95])
    tail = min(_ess((split <= lo).astype(float)), _ess((split <= hi).astype(float)))
    return {"rhat": rhat, "ess_bulk": _ess(z), "ess_tail": tail}
