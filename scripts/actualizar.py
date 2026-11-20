"""Quarterly update in one command: download sources, rebuild inputs, refresh results.

Downloads DANE national accounts, the latest ICML dataset (Zenodo) and the GEIH
microdata months that are needed, splices them with the frozen history, checks the
result and rewrites the estimation inputs. Then it re-runs the Kalman smoother with
the estimated posterior (``scripts/resultados_finales.py``), which is fast. The
full re-estimation (``scripts/estimar_m2.py``) is a separate, annual step.

Usage (PowerShell, from the project root):
    python scripts\\actualizar.py
    python scripts\\actualizar.py --sin-resultados        # only the data
    python scripts\\actualizar.py --simular               # download and check, write nothing
    python scripts\\actualizar.py --validar-ingresos      # also recompute the last frozen quarter
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2.sources.http import SourceError  # noqa: E402
from m2.update import UpdateOptions, UpdatePaths, run_update  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alfa", type=float, default=0.354)
    parser.add_argument("--draws", type=int, default=400, help="Posterior draws for the final results")
    parser.add_argument("--sin-resultados", action="store_true", help="Update the data only")
    parser.add_argument("--simular", action="store_true", help="Download and check, but write nothing")
    parser.add_argument("--validar-ingresos", action="store_true",
                        help="Recompute the last frozen earnings quarter from DANE and report the difference")
    parser.add_argument("--conservar-zips", action="store_true", help="Keep the downloaded GEIH ZIP files")
    parser.add_argument("--anexo-pib", type=Path, help="Local DANE expenditure annex instead of downloading")
    parser.add_argument("--icml-zip", type=Path, help="Local ICML dataset ZIP instead of Zenodo")
    args = parser.parse_args()

    options = UpdateOptions(alpha=args.alfa, validate_last_quarter=args.validar_ingresos,
                            keep_zips=args.conservar_zips, accounts_file=args.anexo_pib,
                            icml_file=args.icml_zip, dry_run=args.simular)
    try:
        report = run_update(UpdatePaths(ROOT), options)
    except SourceError as exc:
        print(f"\nUPDATE STOPPED, nothing was written: {exc}", file=sys.stderr)
        return 2
    for note in report.notes:
        print("NOTE:", note)
    if args.simular or args.sin_resultados or not report.changed:
        return 0
    command = [sys.executable, str(ROOT / "scripts" / "resultados_finales.py"), "--draws", str(args.draws)]
    print("Running:", " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
