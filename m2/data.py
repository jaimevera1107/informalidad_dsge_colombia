"""Observable construction for the M2 estimation.

Observables are quarterly log growth rates of real GDP, private consumption,
gross fixed capital formation, government consumption and total employment,
plus the change in the logit of the labor-informality rate. Deterministic
seasonality and mean growth are removed with quarter-of-year means computed
on a fixed training window (DANE original, non seasonally adjusted series).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

OBSERVABLES: tuple[str, ...] = ("obs_y", "obs_c", "obs_i", "obs_g", "obs_n", "obs_s")
SOURCE_COLUMNS: dict[str, str] = {
    "obs_y": "pib",
    "obs_c": "c",
    "obs_i": "i",
    "obs_g": "g",
    "obs_n": "ocupados",
}
SHARE_COLUMN = "informalidad_ponderada_ocupados"


@dataclass(frozen=True)
class DataConfig:
    """Sample and transformation choices (all explicit, none hidden)."""

    start: str = "2005Q1"
    end: str | None = None
    seasonal_train: tuple[str, str] = ("2005Q1", "2018Q4")
    missing_quarters: tuple[str, ...] = ("2020Q2", "2020Q3", "2020Q4")
    # optional sensitivity: exclude ICML employment and informality before this date from
    # the likelihood (1993-2000 are ICML reconstructions). None (default) uses every quarter.
    informality_observed_from: str | None = None
    # observe the logit informality rate in LEVELS (deviation from logit of its sample mean,
    # seasonal pattern removed) instead of in changes. This pins the level of the wedge state,
    # which the labor-only model needs: there informal earnings depend on informal employment.
    informality_in_levels: bool = False
    # observe the capital-output ratio in LEVELS. Capital is built by perpetual inventory from
    # quarterly investment; capital_initial is the stock at the start of the first quarter of the
    # sample (e.g. from the annual long series); without it a steady-state stock is used.
    capital_in_levels: bool = False
    capital_depreciation: float = 0.0102        # quarterly
    capital_initial: float | None = None


@dataclass
class ObservableSet:
    """Transformed observables plus the levels needed to rebuild shares."""

    quarters: pd.PeriodIndex
    values: np.ndarray              # (T, 6), NaN marks quarters excluded from the likelihood
    informality_rate: np.ndarray    # level of the labor-informality rate s_t, aligned with values
    ratios: dict[str, float]        # sample means of C/Y, I/Y, G/Y
    mean_growth: float              # mean quarterly growth of real GDP in the training window
    informality_mean: float
    seasonal_means: dict[str, dict[int, float]] = field(default_factory=dict)
    names: tuple[str, ...] = OBSERVABLES
    level_observables: tuple[str, ...] = ()

    def with_relative_earnings(self, series: pd.Series) -> "ObservableSet":
        """Append obs_w = log(informal / formal mean labor earnings), in LEVELS.

        `series` is indexed by quarter strings (e.g. '2008Q1'); quarters without
        data stay NaN and are skipped by the Kalman filter.
        """
        lookup = {str(k): float(v) for k, v in series.items()}
        column = np.array([np.log(lookup[q]) if q in lookup and lookup[q] > 0 else np.nan
                           for q in self.quarters.astype(str)])
        values = np.column_stack((self.values, column))
        for t in np.where(np.all(np.isnan(self.values), axis=1))[0]:
            values[t, -1] = np.nan
        return ObservableSet(self.quarters, values, self.informality_rate, self.ratios, self.mean_growth,
                             self.informality_mean, self.seasonal_means, self.names + ("obs_w",),
                             self.level_observables + ("obs_w",))

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.values, columns=list(self.names))
        frame.insert(0, "quarter", self.quarters.astype(str))
        frame["informality_rate"] = self.informality_rate
        return frame


class ObservableBuilder:
    """Loads the consolidated DANE/ICML base and builds M2 observables."""

    def __init__(self, config: DataConfig | None = None) -> None:
        self.config = config or DataConfig()

    def load(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = set(SOURCE_COLUMNS.values()) | {"quarter", SHARE_COLUMN}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        frame = frame.sort_values("quarter").reset_index(drop=True)
        periods = pd.PeriodIndex(frame["quarter"], freq="Q")
        expected = pd.period_range(periods.min(), periods.max(), freq="Q")
        if not periods.equals(expected):
            raise ValueError("Quarters are duplicated or have gaps")
        frame["period"] = periods
        return frame

    def build(self, path: Path) -> ObservableSet:
        cfg = self.config
        frame = self.load(path)
        mask = frame["period"] >= pd.Period(cfg.start, freq="Q")
        if cfg.end is not None:
            mask &= frame["period"] <= pd.Period(cfg.end, freq="Q")
        frame = frame.loc[mask].reset_index(drop=True)
        if len(frame) < 48:
            raise ValueError("Sample shorter than 12 years")

        share = frame[SHARE_COLUMN].to_numpy(float)
        if np.any((share <= 0) | (share >= 1)):
            raise ValueError("Informality rate outside (0, 1)")
        levels = np.column_stack(
            [np.log(frame[col].to_numpy(float)) for col in SOURCE_COLUMNS.values()]
            + [np.log(share / (1 - share))]
        )
        growth = np.diff(levels, axis=0)
        periods = frame["period"].iloc[1:].reset_index(drop=True)
        quarter_of_year = np.array([p.quarter for p in periods])
        train_lo, train_hi = (pd.Period(q, freq="Q") for q in cfg.seasonal_train)
        train = np.array([(p >= train_lo) and (p <= train_hi) for p in periods])
        if train.sum() < 24:
            raise ValueError("Seasonal training window too short")

        transformed = np.empty_like(growth)
        seasonal: dict[str, dict[int, float]] = {}
        for j, name in enumerate(OBSERVABLES):
            seasonal[name] = {}
            for q in range(1, 5):
                sel = train & (quarter_of_year == q)
                mean_q = float(growth[sel, j].mean())
                seasonal[name][q] = mean_q
                transformed[quarter_of_year == q, j] = growth[quarter_of_year == q, j] - mean_q

        level_names: tuple[str, ...] = ()
        if cfg.informality_in_levels:
            j = OBSERVABLES.index("obs_s")
            s_mean = float(share.mean())
            dev = levels[1:, j] - np.log(s_mean / (1 - s_mean))
            offsets = {q: float(dev[train & (quarter_of_year == q)].mean()) for q in range(1, 5)}
            centre = float(np.mean(list(offsets.values())))
            seasonal["obs_s_level"] = {q: offsets[q] - centre for q in range(1, 5)}
            transformed[:, j] = dev - np.array([seasonal["obs_s_level"][q] for q in quarter_of_year])
            level_names = ("obs_s",)

        capital_obs = None
        if cfg.capital_in_levels:
            inv = frame["i"].to_numpy(float)
            d = cfg.capital_depreciation
            k = np.empty(len(inv))
            if cfg.capital_initial is not None:
                k[0] = cfg.capital_initial
            else:
                g = max(float(np.log(inv[min(40, len(inv) - 1)] / inv[0]) / min(40, len(inv) - 1)), 0.0)
                k[0] = inv[0] / (g + d)
            for t in range(1, len(inv)):
                k[t] = (1.0 - d) * k[t - 1] + inv[t - 1]
            log_ky = np.log(k[1:] / frame["pib"].to_numpy(float)[1:])
            dev = log_ky - log_ky.mean()
            offsets = {q: float(dev[train & (quarter_of_year == q)].mean()) for q in range(1, 5)}
            centre = float(np.mean(list(offsets.values())))
            seasonal["obs_k_level"] = {q: offsets[q] - centre for q in range(1, 5)}
            capital_obs = dev - np.array([seasonal["obs_k_level"][q] for q in quarter_of_year])

        excluded = {pd.Period(q, freq="Q") for q in cfg.missing_quarters}
        for t, period in enumerate(periods):
            if period in excluded:
                transformed[t, :] = np.nan

        if cfg.informality_observed_from is not None:
            cutoff = pd.Period(cfg.informality_observed_from, freq="Q")
            labor = [OBSERVABLES.index("obs_n"), OBSERVABLES.index("obs_s")]
            for t, period in enumerate(periods):
                if period <= cutoff:          # growth into the first observed quarter is also excluded
                    transformed[t, labor] = np.nan

        ratios = {
            name: float((frame[col] / frame["pib"]).mean())
            for name, col in (("c_y", "c"), ("i_y", "i"), ("g_y", "g"))
        }
        mean_growth = float(growth[train, 0].mean())
        names = OBSERVABLES
        if capital_obs is not None:
            capital_obs = capital_obs.copy()
            capital_obs[np.all(np.isnan(transformed), axis=1)] = np.nan
            transformed = np.column_stack((transformed, capital_obs))
            names = OBSERVABLES + ("obs_k",)
            level_names = level_names + ("obs_k",)
        return ObservableSet(
            quarters=pd.PeriodIndex(periods, freq="Q"),
            values=transformed,
            informality_rate=share[1:],
            ratios=ratios,
            mean_growth=mean_growth,
            informality_mean=float(share.mean()),
            seasonal_means=seasonal,
            names=names,
            level_observables=level_names,
        )
