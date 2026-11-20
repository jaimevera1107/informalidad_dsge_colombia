"""Value-added earnings ratio for the labor-only informal model.

The GEIH v3 file gives, for informal and formal workers, the employment share
and mean income of each group (own, emp_small, emp_large, domestic, emp_nosize).
In the labor-only accounting what matters is informal value added per worker:

  own, emp_small, domestic   income ~ value added (mixed income and the wages paid
                             out of it; household production uses no capital)
  emp_large                  informal employees of larger firms work with formal
                             capital: their value added is wage / (1 - alpha)
  emp_nosize                 treated like the average informal employee in its
                             quarter (its share is reported, never hidden)

The denominator is the mean wage of formal employees (gross formal wage concept
of the model), or of all formal workers when `formal_base="all"`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

INFORMAL_GROUPS = ("own", "emp_small", "emp_large", "domestic", "emp_nosize")
FORMAL_EMPLOYEES = ("emp_small", "emp_large", "domestic", "emp_nosize")


@dataclass(frozen=True)
class ValueAddedRatio:
    """Builds the quarterly value-added ratio from the GEIH group file."""

    alpha: float
    formal_base: str = "employees"

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.formal_base not in ("employees", "all"):
            raise ValueError("formal_base must be 'employees' or 'all'")

    @staticmethod
    def _check(frame: pd.DataFrame) -> None:
        needed = [f"{k}_informal_{g}" for k in ("share", "mean") for g in INFORMAL_GROUPS]
        missing = [c for c in needed + ["quarter", "mean_formal"] if c not in frame]
        if missing:
            raise KeyError(f"GEIH file is not version 3 (missing {missing[:3]}...): rerun the script")

    def _formal_mean(self, f: pd.DataFrame) -> pd.Series:
        if self.formal_base == "all":
            return f["mean_formal"]
        shares = sum(f[f"share_formal_{g}"].fillna(0.0) for g in FORMAL_EMPLOYEES)
        income = sum((f[f"share_formal_{g}"] * f[f"mean_formal_{g}"]).fillna(0.0) for g in FORMAL_EMPLOYEES)
        return income / shares

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._check(frame)
        f = frame.copy()
        scale = {"emp_large": 1.0 / (1.0 - self.alpha)}
        known = ("own", "emp_small", "emp_large", "domestic")
        va = {g: (f[f"share_informal_{g}"] * f[f"mean_informal_{g}"] * scale.get(g, 1.0)).fillna(0.0)
              for g in known}
        emp_known = ("emp_small", "emp_large")
        emp_share = sum(f[f"share_informal_{g}"].fillna(0.0) for g in emp_known)
        emp_va_per_worker = sum(va[g] for g in emp_known) / emp_share.replace(0.0, np.nan)
        nosize_va = f["share_informal_emp_nosize"].fillna(0.0) * emp_va_per_worker
        informal_va = sum(va.values()) + nosize_va.fillna(0.0)
        out = pd.DataFrame({"quarter": f["quarter"]})
        out["ratio_va"] = informal_va / self._formal_mean(f)
        out["ratio_raw"] = f["ratio"]
        out["share_emp_large"] = f["share_informal_emp_large"]
        out["share_emp_nosize"] = f["share_informal_emp_nosize"]
        out["alpha"] = self.alpha
        return out
