"""Statutory non-wage labor costs of formal employees in Colombia, quarterly 2007-2026.

The GEIH asks wage earners for their monthly pay "before deductions" (P6500), so the
employee's own contributions are included in the reported wage. What separates the
reported wage from the employer's total labor cost (the model's gross formal wage,
(1 - alpha) Yf / Nf) are the employer's contributions and the social benefits paid
outside the monthly wage. As a percentage of the base wage, for a worker earning less
than ten minimum wages in a firm that pays income tax:

    pension (employer 75% of the total rate)  11.625% in 2007, 12% from 2008 (Ley 797, 2003)
    health (employer)                          8.5% from 2007 (Ley 1122); 0 from Jan 2014 (Ley 1607, art. 31)
    occupational risk (class I)                0.522%
    family compensation fund                   4%
    SENA (2%) and ICBF (3%)                    5%; 0 from May 2013 (Ley 1607, art. 25)
    social benefits                            21.83% (prima 8.33, cesantias 8.33, interest 1.0, holidays 4.17)

Monthly rates are averaged to quarters. The measurement equation of the earnings ratio
uses offset_t = log(1 + total_t): log R_obs = log R_model + offset_t.
Two columns are written: with social benefits (baseline) and contributions only
(for the case in which survey income already includes the benefits).

Usage:
    python scripts/costo_no_salarial.py --salida datos/costos_laborales/costo_no_salarial.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BENEFITS = 8.33 + 8.33 + 1.0 + 4.17
OCCUPATIONAL_RISK = 0.522
FAMILY_FUND = 4.0


@dataclass(frozen=True)
class StatutorySchedule:
    """Employer-side rates (percent of the base wage) for a given month."""

    @staticmethod
    def pension(month: pd.Period) -> float:
        total = 15.5 if month.year <= 2007 else 16.0
        return 0.75 * total

    @staticmethod
    def health(month: pd.Period) -> float:
        return 0.0 if month >= pd.Period("2014-01", "M") else 8.5

    @staticmethod
    def sena_icbf(month: pd.Period) -> float:
        return 0.0 if month >= pd.Period("2013-05", "M") else 5.0

    def month(self, month: pd.Period) -> dict[str, float]:
        contributions = (self.pension(month) + self.health(month) + OCCUPATIONAL_RISK + FAMILY_FUND
                         + self.sena_icbf(month))
        return {"pension": self.pension(month), "salud": self.health(month), "arl": OCCUPATIONAL_RISK,
                "caja": FAMILY_FUND, "sena_icbf": self.sena_icbf(month), "prestaciones": BENEFITS,
                "aportes": contributions, "total": contributions + BENEFITS}


def quarterly(start: str = "2007-01", end: str = "2026-12") -> pd.DataFrame:
    schedule = StatutorySchedule()
    months = pd.period_range(start, end, freq="M")
    frame = pd.DataFrame([{"month": m, **schedule.month(m)} for m in months])
    frame["quarter"] = frame["month"].dt.asfreq("Q").astype(str)
    out = frame.drop(columns="month").groupby("quarter").mean().reset_index()
    out["offset_log"] = np.log1p(out["total"] / 100)
    out["offset_log_solo_aportes"] = np.log1p(out["aportes"] / 100)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=Path("datos/costos_laborales/costo_no_salarial.csv"))
    parser.add_argument("--hasta", default=f"{pd.Timestamp.today().year}-12",
                        help="Last month (YYYY-MM); default: December of the current year.")
    args = parser.parse_args()
    out = quarterly(end=args.hasta)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.salida, index=False)
    print(out.loc[out.quarter.isin(["2007Q1", "2012Q4", "2013Q2", "2013Q3", "2014Q1", "2026Q2"]),
                  ["quarter", "aportes", "total", "offset_log", "offset_log_solo_aportes"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
