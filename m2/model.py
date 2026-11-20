"""Two-sector real DSGE with a formal and an informal sector.

Household (log utility, separable disutility of each type of work):
    max E sum beta^t [ log C_t - psi_t ( chi_f Nf^(1+phi) + chi_i Ni^(1+phi) ) / (1+phi) ]
Formal labor income bears the wedge tau_t = 1 - (1 - tau_bar) exp(-omega_t)
(taxes, contributions, regulation); informal income does not.

Firms in both sectors use the same Cobb-Douglas technology with freely
mobile capital; the informal sector has relative labor productivity rho_tilde:
    Y = A K^alpha (X_t L_t)^(1-alpha),  L_t = Nf_t + rho_tilde Ni_t,
so the informal output share equals theta_t = rho_tilde Ni / L
(and equals the informal share of labor income).

X_t is a unit-root labor-augmenting trend with growth gamma + mu_t, mu AR(1).
Capital: K_{t+1} = (1 - delta) K_t + nu_t I_t (marginal efficiency of investment).
Resources: Y_t (1 + xbar + zx_t) = C_t + I_t + G_t, where xbar absorbs the
chain-volume non-additivity and net imports and zx_t is an absorption shock.

All variables are detrended by X_t (capital by X_{t-1}); the model is solved
by first-order perturbation with a generalized Schur decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import ordqz

VARIABLES: tuple[str, ...] = ("lc", "ly", "li", "lnf", "lni", "lk",
                              "mu", "zg", "lv", "lpsi", "om", "zx", "zr")
STATE_INDEX: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12)
JUMP_INDEX: tuple[int, ...] = (0, 1, 2, 3, 4)
SHOCKS: tuple[str, ...] = ("mu", "g", "v", "psi", "om", "x", "zr")
SHOCK_STATE: tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12)


@dataclass(frozen=True)
class Calibration:
    """Parameters fixed before estimation (with their source in the report)."""

    alpha: float = 0.33          # capital share (of formal output in the labor-only variant)
    beta: float = 0.985          # quarterly discount factor
    delta: float = 0.025         # depreciation (used only by the labor-only variant)
    phi: float = 1.0             # inverse Frisch elasticity
    tau_bar: float = 0.25        # steady-state wedge on formal labor income
    n_total: float = 0.30        # normalization of total work
    gamma: float = 0.009         # mean quarterly trend growth (data)
    s_bar: float = 0.62          # mean labor-informality rate (data)
    c_y: float = 0.70            # mean C/Y (data)
    i_y: float = 0.20            # mean I/Y (data)
    g_y: float = 0.15            # mean G/Y (data)


@dataclass(frozen=True)
class Structural:
    """Estimated parameters."""

    theta: float = 0.35          # steady-state informal output share
    eta: float = 0.7             # labor elasticity of informal production (labor-only variant)
    rho_mu: float = 0.4
    rho_g: float = 0.7
    rho_v: float = 0.7
    rho_psi: float = 0.7
    rho_om: float = 0.9
    rho_x: float = 0.7
    sig_mu: float = 0.01
    sig_g: float = 0.04
    sig_v: float = 0.05
    sig_psi: float = 0.02
    sig_om: float = 0.02
    sig_x: float = 0.02
    rho_zr: float = 0.9          # relative informal productivity (active only with earnings data)
    sig_zr: float = 0.0

    def persistences(self) -> np.ndarray:
        return np.array([self.rho_mu, self.rho_g, self.rho_v, self.rho_psi, self.rho_om, self.rho_x,
                         self.rho_zr])

    def volatilities(self) -> np.ndarray:
        return np.array([self.sig_mu, self.sig_g, self.sig_v, self.sig_psi, self.sig_om, self.sig_x,
                         self.sig_zr])


@dataclass(frozen=True)
class SteadyState:
    """Deterministic steady state in detrended units (all derived, none free)."""

    delta: float
    rk: float
    y: float
    c: float
    i: float
    g: float
    k: float
    nf: float
    ni: float
    rho_tilde: float
    chi_f: float
    chi_i: float
    xbar: float

    def vector(self) -> np.ndarray:
        return np.array([np.log(self.c), np.log(self.y), np.log(self.i), np.log(self.nf),
                         np.log(self.ni), np.log(self.k), 0., 0., 0., 0., 0., 0., 0.])


class ModelError(ValueError):
    """Raised when parameters imply no valid steady state or solution."""


class TwoSectorModel:
    """Equilibrium conditions, steady state and first-order solution."""

    def __init__(self, calibration: Calibration, structural: Structural) -> None:
        self.cal = calibration
        self.par = structural
        self.ss = self._steady_state()

    def with_structural(self, structural: Structural) -> "TwoSectorModel":
        return TwoSectorModel(self.cal, structural)

    # ------------------------------------------------------------------ steady state
    def _steady_state(self) -> SteadyState:
        cal, par = self.cal, self.par
        if not 0.0 < par.theta < 1.0:
            raise ModelError("theta must lie in (0, 1)")
        growth = np.exp(cal.gamma)
        ratio = cal.i_y / cal.alpha
        delta = (growth - 1.0 - ratio * (growth / cal.beta - 1.0)) / (ratio - 1.0)
        if not 0.0 < delta < 0.1:
            raise ModelError(f"Implied depreciation outside (0, 0.1): {delta:.4f}")
        rk = growth / cal.beta - 1.0 + delta
        nf = cal.n_total * (1.0 - cal.s_bar)
        ni = cal.n_total * cal.s_bar
        rho_tilde = par.theta * nf / ((1.0 - par.theta) * ni)
        labor = nf + rho_tilde * ni
        y = (cal.alpha / rk) ** (cal.alpha / (1.0 - cal.alpha)) * labor
        k_tilde = cal.alpha * y / rk
        k = k_tilde * growth
        i = k - (1.0 - delta) * k_tilde
        g = cal.g_y * y
        c = cal.c_y * y
        xbar = (c + i + g) / y - 1.0
        wage_f = (1.0 - cal.alpha) * y / labor
        chi_f = (1.0 - cal.tau_bar) * wage_f / (nf ** cal.phi * c)
        chi_i = rho_tilde * wage_f / (ni ** cal.phi * c)
        return SteadyState(delta=delta, rk=rk, y=y, c=c, i=i, g=g, k=k, nf=nf, ni=ni,
                           rho_tilde=rho_tilde, chi_f=chi_f, chi_i=chi_i, xbar=xbar)

    # ------------------------------------------------------------------ equilibrium
    def residuals(self, now: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        """Equilibrium conditions f(Y_t, E_t Y_{t+1}) = 0 in log deviations form."""
        cal, par, ss = self.cal, self.par, self.ss
        lc, ly, li, lnf, lni, lk, mu, zg, lv, lpsi, om, zx, zr = now
        lc1, ly1, _, _, _, lk1, mu1, zg1, lv1, lpsi1, om1, zx1, zr1 = nxt
        a = cal.alpha
        nf, ni = np.exp(lnf), np.exp(lni)
        labor = nf + ss.rho_tilde * np.exp(zr) * ni
        lk_tilde = lk - cal.gamma - mu
        lk_tilde1 = lk1 - cal.gamma - mu1
        y, c, inv = np.exp(ly), np.exp(lc), np.exp(li)
        g = ss.g * np.exp(zg)
        log_scale = np.log(ss.y) - a * np.log(ss.k / np.exp(cal.gamma)) - (1 - a) * np.log(ss.nf + ss.rho_tilde * ss.ni)
        rho = self.par.persistences()
        return np.array([
            ly - log_scale - a * lk_tilde - (1 - a) * np.log(labor),
            lpsi + np.log(ss.chi_f) + cal.phi * lnf + lc
            - (np.log(1 - cal.tau_bar) - om + np.log(1 - a) + ly - np.log(labor)),
            lpsi + np.log(ss.chi_i) + cal.phi * lni + lc
            - (np.log(ss.rho_tilde) + zr + np.log(1 - a) + ly - np.log(labor)),
            np.log(y * (1 + ss.xbar + zx)) - np.log(c + inv + g),
            lk1 - np.log((1 - ss.delta) * np.exp(lk_tilde) + np.exp(lv) * inv),
            (-lc - lv) - np.log(cal.beta) + cal.gamma + mu1 + lc1
            - np.log(a * np.exp(ly1 - lk_tilde1) + (1 - ss.delta) * np.exp(-lv1)),
            mu1 - rho[0] * mu,
            zg1 - rho[1] * zg,
            lv1 - rho[2] * lv,
            lpsi1 - rho[3] * lpsi,
            om1 - rho[4] * om,
            zx1 - rho[5] * zx,
            zr1 - rho[6] * zr,
        ])

    # ------------------------------------------------------------------ earnings and shares
    def earnings_level(self) -> float:
        """Steady-state log of informal earnings over the gross formal wage."""
        return float(np.log(self.ss.rho_tilde))

    def earnings_loadings(self) -> dict[str, float]:
        """Loadings of the log earnings ratio on the model variables (log deviations)."""
        return {"zr": 1.0}

    def output_share(self, ratio: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Informal output share from the earnings ratio and the informality rate."""
        return output_share_common(ratio, s)

    # ------------------------------------------------------------------ solution
    def jacobians(self, step: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        x0 = self.ss.vector()
        n = len(x0)
        d_now, d_next = np.zeros((n, n)), np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = step
            d_now[:, j] = (self.residuals(x0 + e, x0) - self.residuals(x0 - e, x0)) / (2 * step)
            d_next[:, j] = (self.residuals(x0, x0 + e) - self.residuals(x0, x0 - e)) / (2 * step)
        return d_now, d_next

    def solve(self) -> "LinearSolution":
        resid = np.abs(self.residuals(self.ss.vector(), self.ss.vector())).max()
        if resid > 1e-9:
            raise ModelError(f"Steady state does not satisfy the equilibrium: {resid:.2e}")
        d_now, d_next = self.jacobians()
        # d_next * dY' = -d_now * dY  ->  generalized eigenproblem
        aa, bb, alpha, beta, _, z = ordqz(-d_now, d_next, sort="iuc", output="complex")
        stable = int(np.sum(np.abs(alpha) < np.abs(beta) * (1 - 1e-9)))
        ns = len(STATE_INDEX)
        if stable != ns:
            raise ModelError(f"Blanchard-Kahn violated: {stable} stable roots, need {ns}")
        z11 = z[np.ix_(STATE_INDEX, range(ns))]
        z21 = z[np.ix_(JUMP_INDEX, range(ns))]
        if np.linalg.cond(z11) > 1e10:
            raise ModelError("Rank condition fails (z11 singular)")
        z11_inv = np.linalg.inv(z11)
        transition = np.real(z11 @ np.linalg.solve(bb[:ns, :ns], aa[:ns, :ns]) @ z11_inv)
        policy = np.real(z21 @ z11_inv)
        impact = np.zeros((ns, len(SHOCKS)))
        for j, idx in enumerate(SHOCK_STATE):
            impact[STATE_INDEX.index(idx), j] = 1.0
        return LinearSolution(transition=transition, policy=policy, impact=impact, ss=self.ss)


@dataclass(frozen=True)
class LinearSolution:
    """s_{t+1} = T s_t + R eps_{t+1};  jumps_t = P s_t  (s = deviations of states)."""

    transition: np.ndarray
    policy: np.ndarray
    impact: np.ndarray
    ss: SteadyState

    def full_map(self) -> np.ndarray:
        """Matrix M with Y_t = M s_t for the complete variable vector."""
        n_var = len(VARIABLES)
        ns = len(STATE_INDEX)
        mapping = np.zeros((n_var, ns))
        for j, idx in enumerate(STATE_INDEX):
            mapping[idx, j] = 1.0
        for j, idx in enumerate(JUMP_INDEX):
            mapping[idx, :] = self.policy[j]
        return mapping


def output_share_common(ratio: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Equal technologies: output share = labor-income share."""
    return ratio * s / ((1.0 - s) + ratio * s)


def output_share_labor_only(ratio: np.ndarray, s: np.ndarray, alpha: float) -> np.ndarray:
    """Labor-only informal sector: informal earnings ~ informal value added, while the
    gross formal wage is the share (1 - alpha) of formal value added per worker."""
    x = (1.0 - alpha) * ratio * s / (1.0 - s)
    return x / (1.0 + x)


def with_values(structural: Structural, **values: float) -> Structural:
    return replace(structural, **values)


class LaborOnlyInformalModel(TwoSectorModel):
    """Variant with the standard informal technology of the shadow-economy DGE literature.

    Formal:   Yf = K^alpha (X Nf)^(1-alpha)        (all capital is formal)
    Informal: Yi = Ai X Ni^eta,  0 < eta < 1        (labor only, decreasing returns)
    Aggregate shocks now change the relative return of formal and informal work,
    so informality responds to the cycle and its size is tied to observed co-movements.
    Depreciation and beta are fixed; the model I/Y is implied and the chain-volume
    residual xbar absorbs the gap with the data ratio.
    """

    def _steady_state(self) -> SteadyState:
        cal, par = self.cal, self.par
        if not (0.0 < par.theta < 1.0 and 0.0 < par.eta < 1.0):
            raise ModelError("theta and eta must lie in (0, 1)")
        growth = np.exp(cal.gamma)
        delta = cal.delta
        rk = growth / cal.beta - 1.0 + delta
        nf = cal.n_total * (1.0 - cal.s_bar)
        ni = cal.n_total * cal.s_bar
        k_tilde_per_y = cal.alpha / rk                     # K~ / Yf
        yf = k_tilde_per_y ** (cal.alpha / (1.0 - cal.alpha)) * nf
        y = yf / (1.0 - par.theta)
        yi = y - yf
        k_tilde = k_tilde_per_y * yf
        k = k_tilde * growth
        i = k - (1.0 - delta) * k_tilde
        g = cal.g_y * y
        c = cal.c_y * y
        xbar = (c + i + g) / y - 1.0
        chi_f = (1.0 - cal.tau_bar) * (1.0 - cal.alpha) * yf / nf / (nf ** cal.phi * c)
        chi_i = par.eta * yi / ni / (ni ** cal.phi * c)
        a_i = yi / ni ** par.eta
        return SteadyState(delta=delta, rk=rk, y=y, c=c, i=i, g=g, k=k, nf=nf, ni=ni,
                           rho_tilde=a_i, chi_f=chi_f, chi_i=chi_i, xbar=xbar)

    def residuals(self, now: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        cal, par, ss = self.cal, self.par, self.ss
        lc, ly, li, lnf, lni, lk, mu, zg, lv, lpsi, om, zx, zr = now
        lc1, ly1, _, lnf1, _, lk1, mu1, zg1, lv1, lpsi1, om1, zx1, zr1 = nxt
        a = cal.alpha
        lk_tilde = lk - cal.gamma - mu
        lk_tilde1 = lk1 - cal.gamma - mu1
        yf = np.exp(a * lk_tilde + (1 - a) * lnf)
        yf1 = np.exp(a * lk_tilde1 + (1 - a) * lnf1)
        yi = ss.rho_tilde * np.exp(zr + par.eta * lni)
        y, c, inv = np.exp(ly), np.exp(lc), np.exp(li)
        g = ss.g * np.exp(zg)
        rho = par.persistences()
        return np.array([
            ly - np.log(yf + yi),
            lpsi + np.log(ss.chi_f) + cal.phi * lnf + lc
            - (np.log(1 - cal.tau_bar) - om + np.log(1 - a) + np.log(yf) - lnf),
            lpsi + np.log(ss.chi_i) + cal.phi * lni + lc - (np.log(par.eta) + np.log(yi) - lni),
            np.log(y * (1 + ss.xbar + zx)) - np.log(c + inv + g),
            lk1 - np.log((1 - ss.delta) * np.exp(lk_tilde) + np.exp(lv) * inv),
            (-lc - lv) - np.log(cal.beta) + cal.gamma + mu1 + lc1
            - np.log(a * yf1 / np.exp(lk_tilde1) + (1 - ss.delta) * np.exp(-lv1)),
            mu1 - rho[0] * mu,
            zg1 - rho[1] * zg,
            lv1 - rho[2] * lv,
            lpsi1 - rho[3] * lpsi,
            om1 - rho[4] * om,
            zx1 - rho[5] * zx,
            zr1 - rho[6] * zr,
        ])

    def with_structural(self, structural: Structural) -> "LaborOnlyInformalModel":
        return LaborOnlyInformalModel(self.cal, structural)

    # Survey income of informal workers is mixed income, profit or the wage paid out of
    # them, so it measures informal value added per worker (Yi / Ni). Formal survey
    # wages measure (1 - alpha) Yf / Nf before the wedge.
    def earnings_level(self) -> float:
        ss, a = self.ss, self.cal.alpha
        yi, yf = ss.y * self.par.theta, ss.y * (1.0 - self.par.theta)
        return float(np.log((yi / ss.ni) / ((1.0 - a) * yf / ss.nf)))

    def earnings_loadings(self) -> dict[str, float]:
        a = self.cal.alpha
        # log(Yi/Ni) = log Ai + zr + (eta - 1) lni;  log(Yf/Nf) = a (lk - gamma - mu - lnf)
        return {"zr": 1.0, "lni": self.par.eta - 1.0, "lk": -a, "mu": a, "lnf": a}

    def output_share(self, ratio: np.ndarray, s: np.ndarray) -> np.ndarray:
        return output_share_labor_only(ratio, s, self.cal.alpha)
