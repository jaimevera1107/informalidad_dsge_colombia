"""Estimate the capital share alpha (aggregate and formal sector) by dynamic OLS.

Usage:
    python scripts/estimar_alfa.py --datos datos/datos_dsge_estimacion.csv ^
        --participacion resultados_m2/con_ingresos/informal_output_share.csv ^
        --salida resultados_m2/alfa
The --participacion file gives x_t = theta_t / (1 - theta_t) under equal
technologies, i.e. R_t s_t / (1 - s_t); it is only needed for the formal estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # project root: package m2

from m2.capital_share import AnnualData, CapitalConfig, CapitalShareEstimator


def annual_value_added_ratio(path: Path) -> pd.Series:
    """Annual mean of x_t = R_t s_t / (1 - s_t) from an equal-technology share file."""
    share = pd.read_csv(path)
    theta = share["informal_output_share_median"]
    share["x"] = theta / (1.0 - theta)
    share["year"] = share["quarter"].astype(str).str[:4].astype(int)
    return share.groupby("year")["x"].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=Path("datos/datos_dsge_estimacion.csv"))
    parser.add_argument("--participacion", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=Path("resultados_m2/alfa"))
    parser.add_argument("--depreciacion", type=float, nargs="+", default=[0.04, 0.06, 0.08],
                        help="tasas anuales; 0.04 es la tasa estandar (DNP 2004 estima 3,8 por ciento)")
    parser.add_argument("--desde", type=int, default=1993)
    parser.add_argument("--hasta", type=int, default=None)
    args = parser.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    quarterly = pd.read_csv(args.datos, encoding="utf-8-sig")
    ratio = annual_value_added_ratio(args.participacion) if args.participacion else None
    tables = []
    for d in args.depreciacion:
        config = CapitalConfig(depreciation=d, start_year=args.desde, end_year=args.hasta)
        data = AnnualData(quarterly, config)
        tables.append(CapitalShareEstimator(data, ratio).table())
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.salida / "capital_share.csv", index=False)
    print(table.round(4).to_string(index=False))
    central = table.loc[table.specification.str.startswith("formal") & table.depreciation.eq(args.depreciacion[0])]
    if len(central):
        print(json.dumps({f"alpha_formal_delta_{args.depreciacion[0]}": round(float(central.alpha.iloc[0]), 4)}))


if __name__ == "__main__":
    main()
