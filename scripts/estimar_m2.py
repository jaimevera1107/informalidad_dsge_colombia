"""Command-line entry point for the M2 estimation.

Example:
    python scripts/estimar_m2.py --datos datos/datos_dsge_estimacion.csv --salida resultados_m2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # project root: package m2

from m2.data import DataConfig
from m2.runner import M2Estimation, RunConfig


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate the M2 informal-economy DSGE.")
    parser.add_argument("--datos", type=Path, default=Path("datos/datos_dsge_estimacion.csv"))
    parser.add_argument("--salida", type=Path, default=Path("resultados_m2"))
    parser.add_argument("--desde", default="2005Q1")
    parser.add_argument("--laboral-observado-desde", default=None,
                        help="sensibilidad: excluir de la verosimilitud empleo e informalidad ICML antes de este trimestre")
    parser.add_argument("--cadenas", type=int, default=4)
    parser.add_argument("--draws", type=int, default=15000)
    parser.add_argument("--burn", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--theta-media", type=float, default=0.35)
    parser.add_argument("--theta-sd", type=float, default=0.10)
    parser.add_argument("--ingresos", type=Path, default=None,
                        help="CSV con columnas quarter, ratio (ingreso laboral medio informal / formal, GEIH)")
    parser.add_argument("--columna-ingresos", default="ratio", help="ratio, ratio_hourly o ratio_median")
    parser.add_argument("--ajuste-ingresos", type=float, default=None,
                        help="log(ratio observado / productividad relativa); por defecto -log(1 - tau_bar)")
    parser.add_argument("--ajuste-ingresos-archivo", type=Path, default=None,
                        help="CSV quarter,offset_log: ajuste log trimestral del salario formal (costos no salariales)")
    parser.add_argument("--columna-ajuste", default="offset_log",
                        help="offset_log (aportes + prestaciones) u offset_log_solo_aportes")
    parser.add_argument("--eta-media", type=float, default=None)
    parser.add_argument("--eta-sd", type=float, default=None)
    parser.add_argument("--capital-en-niveles", action="store_true",
                        help="observar la razon capital/PIB (inventario perpetuo) en niveles")
    parser.add_argument("--capital-inicial-archivo", type=Path, default=None,
                        help="CSV anual year,capital (datos/series_anuales_largas.csv) para el stock inicial")
    parser.add_argument("--informalidad-en-niveles", action="store_true",
                        help="observar el logit de la tasa de informalidad en niveles (recomendado con solo_trabajo)")
    parser.add_argument("--modelo", choices=("comun", "solo_trabajo"), default="comun",
                        help="comun: misma tecnologia; solo_trabajo: sector informal solo con trabajo")
    parser.add_argument("--alfa", type=float, default=None,
                        help="participacion del capital (del producto formal en solo_trabajo); ver estimar_alfa.py")
    parser.add_argument("--depreciacion", type=float, default=None,
                        help="depreciacion trimestral (solo_trabajo); por defecto 0.025")
    return parser.parse_args()


def initial_capital(args: argparse.Namespace) -> float | None:
    """Capital at the start of the first sample year, from the annual long series."""
    if not args.capital_en_niveles or args.capital_inicial_archivo is None:
        return None
    import pandas as pd
    table = pd.read_csv(args.capital_inicial_archivo).set_index("year")
    return float(table.loc[int(args.desde[:4]), "capital"])


def main() -> None:
    args = parse()
    config = RunConfig(data_path=args.datos, output=args.salida, n_chains=args.cadenas,
                       n_draws=args.draws, n_burn=args.burn, workers=args.workers,
                       theta_prior=(args.theta_media, args.theta_sd),
                       data_config=DataConfig(start=args.desde,
                                              informality_observed_from=args.laboral_observado_desde,
                                              informality_in_levels=args.informalidad_en_niveles,
                                              capital_in_levels=args.capital_en_niveles,
                                              capital_depreciation=args.depreciacion or 0.0102,
                                              capital_initial=initial_capital(args)), earnings_path=args.ingresos,
                       earnings_column=args.columna_ingresos, earnings_offset=args.ajuste_ingresos,
                       model={"comun": "common", "solo_trabajo": "labor_only"}[args.modelo],
                       alpha=args.alfa, delta=args.depreciacion,
                       earnings_offset_path=args.ajuste_ingresos_archivo, earnings_offset_column=args.columna_ajuste,
                       eta_prior=None if args.eta_media is None else (args.eta_media, args.eta_sd or 0.10))
    summary = M2Estimation(config).run()
    print(json.dumps({k: summary[k] for k in ("rhat_max", "ess_bulk_min", "converged", "acceptance", "minutes")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
