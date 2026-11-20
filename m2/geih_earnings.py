"""Relative labor earnings of informal versus formal workers from GEIH microdata.

This module holds the computation shared by the two ways of reading the survey:

* ``scripts/geih_ingresos_relativos.py`` reads the author's normalized GEIH parquet
  files and classifies informality with the author's processing engine;
* ``m2/sources/geih.py`` downloads the monthly files published by DANE and
  classifies informality with :mod:`m2.sources.informality`.

Both pass the occupied persons of one month (columns prefixed ``SRC_``) and a
formality status (1 informal, 0 formal, NaN unclassified) to
:func:`month_sums`, and aggregate months with :func:`quarterly`.

Worker groups (P6430 position and unit size), because what the survey income
measures differs by group:

    own        own-account workers and employers (P6430 = 4, 5): mixed income or
               profit, which already contains the return to their own capital.
    emp_small  private employees and day laborers (P6430 = 1, 7, 8) in units of up
               to five persons.
    emp_large  private employees in units of six or more persons and government
               employees (P6430 = 2).
    domestic   domestic employees (P6430 = 3).
    emp_nosize employees without a usable unit size (kept apart, never imputed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

INCOME_COLUMNS = ("SRC_INGLABO", "SRC_P6500", "SRC_P6750", "SRC_P6760", "SRC_P6800", "SRC_P7070")
# National expansion factors in order of preference. NORM_FEX_VALOR is empty in some
# months of the normalized files (2008-04..12, 2011); the survey factor is used then.
WEIGHT_COLUMNS = ("NORM_FEX_VALOR", "SRC_FEX_C18", "SRC_FEX_C_2011", "SRC_FEX_C", "SRC_FEX_C_2005", "SRC_FEX")
# Unit size: P6870 until 2021, P3069 in the 2018-frame questionnaire; codes 1-3 = 1 to 5 persons.
SIZE_COLUMNS = ("SRC_P6870", "SRC_P3069")
SMALL_UNIT_CODES = (1, 2, 3)
MAX_MONTHLY_INCOME = 1e10      # larger values are codes or corrupt fields, never income
UNCLASSIFIED_LIMIT = 0.25      # same threshold as the formal-employment engine
PAID_POSITIONS = (1, 2, 3, 4, 5, 7, 8)   # P6430: excludes 6 (unpaid family worker)
WAGE_POSITIONS = (1, 2, 3, 7, 8)
GROUPS = ("own", "emp_small", "emp_large", "domestic", "emp_nosize")
STATUSES = (("informal", 1.0), ("formal", 0.0))
CALCULATED = "CALCULADO"


def to_number(series: pd.Series) -> pd.Series:
    """Parse survey fields that may use a decimal comma; invalid values become NaN."""
    text = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").astype("float64")


def _get(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in frame:
        return to_number(frame[column])
    return pd.Series(default, index=frame.index, dtype="float64")


def classify_group(position: pd.Series, size_code: pd.Series) -> pd.Series:
    """Assign each worker to one of :data:`GROUPS` from P6430 and the unit-size code."""
    group = pd.Series(pd.NA, index=position.index, dtype="string")
    private_emp = position.isin([1, 7, 8])
    group[position.isin([4, 5])] = "own"
    group[position.eq(3)] = "domestic"
    group[position.eq(2)] = "emp_large"
    group[private_emp & size_code.isin(SMALL_UNIT_CODES)] = "emp_small"
    group[private_emp & size_code.ge(4)] = "emp_large"
    group[private_emp & group.isna()] = "emp_nosize"
    return group


def weight_column(occ: pd.DataFrame) -> tuple[pd.Series, str | None]:
    """National person weight: the first factor column answered for most persons."""
    upper = {c.upper(): c for c in occ.columns}
    preferred = [upper[c] for c in WEIGHT_COLUMNS if c in upper]
    # then any other person-level factor; departmental factors are not national weights
    others = sorted(c for c in occ.columns if "FEX" in c.upper() and c not in preferred
                    and "DPTO" not in c.upper() and "CAMPO" not in c.upper())
    for col in preferred + others:
        w = to_number(occ[col])
        if w.gt(0).mean() > 0.5:
            return w, col
    return pd.Series(np.nan, index=occ.index), None


def size_code(occ: pd.DataFrame) -> tuple[pd.Series, str | None]:
    """Unit-size code from the first size column that has answers."""
    for col in SIZE_COLUMNS:
        if col in occ:
            code = to_number(occ[col])
            if code.notna().any():
                return code, col
    return pd.Series(np.nan, index=occ.index), None


def labor_income(occ: pd.DataFrame, position: pd.Series) -> tuple[pd.Series, str]:
    """Monthly labor income: DANE's INGLABO when available, otherwise rebuilt from items."""
    if "SRC_INGLABO" in occ:
        inglabo = to_number(occ["SRC_INGLABO"])
        if inglabo.notna().mean() > 0.5:
            return inglabo, "INGLABO"
    wage = _get(occ, "SRC_P6500")
    profit = _get(occ, "SRC_P6750")
    months = _get(occ, "SRC_P6760").where(lambda s: s.gt(0), 1.0)
    main = wage.where(position.isin(WAGE_POSITIONS), profit / months)
    second = _get(occ, "SRC_P7070", 0.0).fillna(0.0)
    return main + second, "P6500_P6750_P7070"


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


@dataclass
class MonthResult:
    """Weighted sums of one month, or the reason why the month was not computed."""

    period: str
    status: str
    sums: dict[str, float | int | str] = field(default_factory=dict)

    def row(self) -> dict:
        return {"periodo": self.period, "estado": self.status, **self.sums}


def month_sums(period: str, occ: pd.DataFrame, status: pd.Series) -> MonthResult:
    """Weighted income sums by formality status and worker group for one month.

    Parameters
    ----------
    period:
        Month as ``YYYY-MM``.
    occ:
        Occupied persons, survey variables prefixed ``SRC_``.
    status:
        Formality status aligned with ``occ``: 1 informal, 0 formal, NaN unclassified.
    """
    weight, weight_source = weight_column(occ)
    if weight_source is None:
        return MonthResult(period, "SIN_FACTOR_EXPANSION")
    position = _get(occ, "SRC_P6430")
    income, source = labor_income(occ, position)
    income = income.where(np.isfinite(income) & income.lt(MAX_MONTHLY_INCOME))
    hours = _get(occ, "SRC_P6800")
    code, size_source = size_code(occ)
    group = classify_group(position, code)

    valid_w = weight.gt(0)
    unclassified = float(weight[valid_w & status.isna()].sum() / weight[valid_w].sum())
    if unclassified > UNCLASSIFIED_LIMIT:
        return MonthResult(period, "CLASIFICACION_NO_REPRESENTATIVA",
                           {"sin_clasificar_pct": 100 * unclassified})
    sums: dict[str, float | int | str] = {
        "sin_clasificar_pct": 100 * unclassified, "fuente_ingreso": source,
        "fuente_factor": weight_source, "fuente_tamano": size_source or "NINGUNA"}
    paid = valid_w & position.isin(PAID_POSITIONS) & income.gt(0)
    for label, flag in STATUSES:
        m = paid & status.eq(flag)
        w, y = weight[m], income[m]
        sums[f"w_{label}"] = float(w.sum())
        sums[f"wy_{label}"] = float((w * y).sum())
        sums[f"n_{label}"] = int(m.sum())
        sums[f"median_{label}"] = weighted_median(y.to_numpy(), w.to_numpy())
        h = hours[m]
        hm = h.gt(0)
        sums[f"wh_{label}"] = float((w[hm] * h[hm] * 4.33).sum())       # monthly hours
        sums[f"why_{label}"] = float((w[hm] * y[hm]).sum())
        for g in GROUPS:
            mg = m & group.eq(g).fillna(False)
            sums[f"w_{label}_{g}"] = float(weight[mg].sum())
            sums[f"wy_{label}_{g}"] = float((weight[mg] * income[mg]).sum())
            sums[f"n_{label}_{g}"] = int(mg.sum())
    return MonthResult(period, CALCULATED, sums)


def quarterly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete quarters (three computed months) to ratios and group shares."""
    ok = monthly.loc[monthly.estado.eq(CALCULATED)].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["quarter"] = pd.PeriodIndex(ok.periodo, freq="M").asfreq("Q").astype(str)
    rows = []
    for quarter, g in ok.groupby("quarter"):
        if len(g) < 3 or (g.w_informal <= 0).any() or (g.w_formal <= 0).any():
            continue
        mi = g.wy_informal.sum() / g.w_informal.sum()
        mf = g.wy_formal.sum() / g.w_formal.sum()
        hi = g.why_informal.sum() / g.wh_informal.sum()
        hf = g.why_formal.sum() / g.wh_formal.sum()
        row = {"quarter": quarter, "ratio": mi / mf,
               "ratio_median": g.median_informal.mean() / g.median_formal.mean(),
               "ratio_hourly": hi / hf, "mean_informal": mi, "mean_formal": mf,
               "n_informal": int(g.n_informal.sum()), "n_formal": int(g.n_formal.sum()),
               "unclassified_pct": float(g.sin_clasificar_pct.mean()),
               "income_source": "|".join(sorted(set(g.fuente_ingreso))),
               "size_source": "|".join(sorted(set(g.fuente_tamano)))}
        for label, _ in STATUSES:
            total = g[f"w_{label}"].sum()
            for grp in GROUPS:
                w = g[f"w_{label}_{grp}"].sum()
                row[f"share_{label}_{grp}"] = w / total
                row[f"mean_{label}_{grp}"] = g[f"wy_{label}_{grp}"].sum() / w if w > 0 else np.nan
                row[f"n_{label}_{grp}"] = int(g[f"n_{label}_{grp}"].sum())
        rows.append(row)
    return pd.DataFrame(rows)
