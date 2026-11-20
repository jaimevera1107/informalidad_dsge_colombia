"""Build the value-added earnings ratio (labor-only model) from the GEIH v3 file.

Usage:
    python scripts/construir_ratio_va.py --geih datos/ratio_ingresos_geih_v3.csv --alfa 0.33 ^
        --salida datos/ratio_va_geih.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # project root: package m2

from m2.earnings import ValueAddedRatio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geih", type=Path, required=True)
    parser.add_argument("--alfa", type=float, required=True)
    parser.add_argument("--base-formal", choices=("employees", "all"), default="employees")
    parser.add_argument("--salida", type=Path, required=True)
    args = parser.parse_args()
    out = ValueAddedRatio(args.alfa, args.base_formal).build(pd.read_csv(args.geih))
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.salida, index=False)
    print(json.dumps({"quarters": len(out), "ratio_va_mean": round(float(out.ratio_va.mean()), 4),
                      "ratio_raw_mean": round(float(out.ratio_raw.mean()), 4),
                      "share_emp_large_mean": round(float(out.share_emp_large.mean()), 4),
                      "share_emp_nosize_mean": round(float(out.share_emp_nosize.mean()), 4)}, indent=2))


if __name__ == "__main__":
    main()
