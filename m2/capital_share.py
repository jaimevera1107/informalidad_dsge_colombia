"""Capital share of a Cobb-Douglas production function, estimated from the data.

Two estimates are produced from annual national accounts and employment:

aggregate  ln(Y/N)   = c + alpha ln(K/N)   + g t + e
formal     ln(Yf/Nf) = c + alpha ln(K/Nf)  + g t + e

The formal equation is the one the labor-only informal model needs: all capital
is formal, Yf = Y (1 - theta_t) and Nf = N (1 - s_t). The informal output share
theta_t itself depends on alpha through the value-added accounting

    theta_t = (1 - alpha) x_t / ((1 - alpha) x_t + 1),   x_t = R_t s_t / (1 - s_t),

where R_t is informal value added per worker over the gross formal wage. The
formal estimate is therefore a fixed point in alpha.

Capital is built by the perpetual-inventory method from gross fixed capital
formation in constant prices. Because the series are integrated, the regression
is estimated by dynamic OLS (Stock and Watson 1993) with leads and lags of the
differenced regressor and Newey-West standard errors. Constant returns are
imposed; the unrestricted regression is reported as a check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CapitalConfig:
    """Choices of the perpetual-inventory capital stock and the regression."""

    depreciation: float = 0.06        # annual depreciation rate
    initial_growth_years: int = 10    # years used for the initial steady-state stock
    dols_leads_lags: int = 1
    newey_west_lags: int = 2
    trend: bool = True
    start_year: int = 1993
    end_year: int | None = None        # last complete year by default
    max_iterations: int = 200
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 < self.depreciation < 1.0:
            raise ValueError("depreciation must lie in (0, 1)")
        if self.dols_leads_lags < 0 or self.newey_west_lags < 0:
            raise ValueError("lags must be non-negative")


@dataclass
class RegressionResult:
    """Coefficient of interest with its standard error and fit statistics."""

    name: str
    alpha: float
    se: float
    n_obs: int
    r2: float
    extra: dict[str, float] = field(default_factory=dict)

    def row(self) -> dict[str, float | str | int]:
        lo, hi = self.alpha - 1.645 * self.se, self.alpha + 1.645 * self.se
        return {"specification": self.name, "alpha": self.alpha, "se": self.se,
                "ci90_low": lo, "ci90_high": hi, "n_obs": self.n_obs, "r2": self.r2, **self.extra}


class AnnualData:
    """Annual aggregates from the quarterly estimation dataset (levels)."""

    REQUIRED = ("quarter", "pib", "i", "ocupados", "informalidad_ponderada_ocupados")

    def __init__(self, quarterly: pd.DataFrame, config: CapitalConfig) -> None:
        missing = [c for c in self.REQUIRED if c not in quarterly]
        if missing:
            raise KeyError(f"columns missing in the dataset: {missing}")
        self.config = config
        q = quarterly.copy()
        q["year"] = q["quarter"].astype(str).str[:4].astype(int)
        counts = q.groupby("year")["pib"].count()
        complete = counts.index[counts.eq(4)]
        q = q.loc[q.year.isin(complete)]
        annual = q.groupby("year").agg(y=("pib", "sum"), inv=("i", "sum"), n=("ocupados", "mean"),
                                       s=("informalidad_ponderada_ocupados", "mean"))
        end = config.end_year or int(annual.index.max())
        self.frame = annual.loc[config.start_year:end].copy()
        if len(self.frame) < 15:
            raise ValueError("fewer than 15 complete years: the regression is not informative")
        self.frame["k"] = self._capital(self.frame["inv"])

    def _capital(self, inv: pd.Series) -> pd.Series:
        """Perpetual inventory with a steady-state initial stock K0 = I0 / (g + delta)."""
        d = self.config.depreciation
        years = min(self.config.initial_growth_years, len(inv) - 1)
        growth = float(np.log(inv.iloc[years] / inv.iloc[0]) / years)
        k = np.empty(len(inv))
        k[0] = inv.iloc[0] / (max(growth, 0.0) + d)
        for t in range(1, len(inv)):
            k[t] = (1.0 - d) * k[t - 1] + inv.iloc[t - 1]
        return pd.Series(k, index=inv.index)


class DynamicOLS:
    """Dynamic OLS for one cointegrating coefficient with Newey-West errors."""

    def __init__(self, leads_lags: int, nw_lags: int, trend: bool) -> None:
        self.p, self.nw, self.trend = leads_lags, nw_lags, trend

    def fit(self, y: pd.Series, x: pd.Series, name: str) -> RegressionResult:
        frame = pd.DataFrame({"y": y, "x": x})
        dx = frame.x.diff()
        for j in range(-self.p, self.p + 1):
            frame[f"dx_{j}"] = dx.shift(-j)
        frame = frame.dropna()
        cols = [np.ones(len(frame)), frame.x.to_numpy()]
        if self.trend:
            cols.append(np.arange(len(frame), dtype=float))
        cols += [frame[f"dx_{j}"].to_numpy() for j in range(-self.p, self.p + 1)]
        X = np.column_stack(cols)
        yv = frame.y.to_numpy()
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        cov = self._newey_west(X, resid)
        r2 = 1.0 - resid.var() / yv.var()
        extra = {"trend_coef": float(beta[2])} if self.trend else {}
        return RegressionResult(name, float(beta[1]), float(np.sqrt(cov[1, 1])), len(frame), float(r2), extra)

    def _newey_west(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        n = len(u)
        xu = X * u[:, None]
        s = xu.T @ xu
        for lag in range(1, self.nw + 1):
            w = 1.0 - lag / (self.nw + 1.0)
            g = xu[lag:].T @ xu[:-lag]
            s += w * (g + g.T)
        bread = np.linalg.inv(X.T @ X)
        return bread @ s @ bread * n / (n - X.shape[1])


class CapitalShareEstimator:
    """Aggregate and formal-sector capital shares, plus a returns-to-scale check."""

    def __init__(self, data: AnnualData, value_added_ratio: pd.Series | None = None) -> None:
        self.data = data
        self.config = data.config
        self.ratio = value_added_ratio            # annual x_t = R_t s_t / (1 - s_t)
        self.ols = DynamicOLS(self.config.dols_leads_lags, self.config.newey_west_lags, self.config.trend)

    def aggregate(self) -> RegressionResult:
        f = self.data.frame
        return self.ols.fit(np.log(f.y / f.n), np.log(f.k / f.n), "aggregate, CRS")

    def unrestricted(self) -> RegressionResult:
        """ln Y on ln K and ln N by OLS; reports the sum of elasticities."""
        f = self.data.frame
        X = np.column_stack([np.ones(len(f)), np.log(f.k), np.log(f.n), np.arange(len(f), dtype=float)])
        yv = np.log(f.y).to_numpy()
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        r2 = 1.0 - resid.var() / yv.var()
        return RegressionResult("aggregate, unrestricted OLS", float(beta[1]), float("nan"), len(f), float(r2),
                                {"labor_elasticity": float(beta[2]), "returns_to_scale": float(beta[1] + beta[2])})

    def formal(self, start_alpha: float = 0.33) -> RegressionResult:
        """Fixed point alpha = DOLS(Yf(alpha)) for the labor-only informal model."""
        if self.ratio is None:
            raise ValueError("the formal estimate needs the value-added ratio x_t")
        f = self.data.frame
        x = self.ratio.reindex(f.index)
        if x.isna().any():
            raise ValueError(f"value-added ratio missing for years {list(x.index[x.isna()])}")
        nf = f.n * (1.0 - f.s)
        alpha, result = start_alpha, None
        for iteration in range(1, self.config.max_iterations + 1):
            theta = (1.0 - alpha) * x / ((1.0 - alpha) * x + 1.0)
            yf = f.y * (1.0 - theta)
            result = self.ols.fit(np.log(yf / nf), np.log(f.k / nf), "formal sector, labor-only informal, CRS")
            if abs(result.alpha - alpha) < self.config.tolerance:
                break
            alpha = result.alpha
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"fixed point left (0, 1) at iteration {iteration}: alpha={alpha:.3f}")
        else:
            raise RuntimeError("fixed point for the formal capital share did not converge")
        theta = (1.0 - result.alpha) * x / ((1.0 - result.alpha) * x + 1.0)
        result.extra.update({"iterations": iteration, "theta_first": float(theta.iloc[0]),
                             "theta_last": float(theta.iloc[-1])})
        return result

    def table(self) -> pd.DataFrame:
        rows = [self.aggregate().row(), self.unrestricted().row()]
        if self.ratio is not None:
            rows.append(self.formal().row())
        out = pd.DataFrame(rows)
        out.insert(1, "depreciation", self.config.depreciation)
        return out
