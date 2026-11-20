"""Relative labor earnings of informal vs formal workers from the GEIH pipeline.

Reuses the user's own GEIH processing (03_ACUMULADOS catalog, the recovered
informality classifier and its guard) and only adds the income variables.
National weights are the microdata factor (NORM_FEX_VALOR: FEX_C / FEX_C_2011 /
FEX_C18), with a fallback to the original survey factor when it is empty.

Version 3 also splits paid workers by occupational position and size of the
production unit, because what the survey income measures differs by group:

    own        own-account workers and employers (P6430 = 4, 5): their income is
               mixed income or profit, so it already contains the return to the
               capital of their own unit (income ~ value added per worker).
    emp_small  private employees and day laborers (P6430 = 1, 7, 8) in units of
               up to five persons: the employer's surplus is counted in "own".
    emp_large  private employees in units of six or more persons and government
               employees (P6430 = 2): the return to capital stays in the firm.
    domestic   domestic employees (P6430 = 3): household production, no capital.
    emp_nosize employees without a usable unit size (kept apart, never imputed).

Output (quarterly):
    quarter, ratio, ratio_median, ratio_hourly, mean_informal, mean_formal,
    and for each group g and status s in (informal, formal):
    share_<s>_<g>  employment share of g within status s (weighted)
    mean_<s>_<g>   weighted mean monthly labor income of g within status s
The value-added ratio used by the model is built afterwards with
m2/earnings.py, which needs the capital share of the formal sector.

Usage (PowerShell):
    python scripts\\geih_ingresos_relativos.py --raiz "G:\\My Drive\\SENA\\Procesamiento\\GEIH" ^
        --salida datos\\geih\\ratio_ingresos_geih_v3.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: package m2

from m2.geih_earnings import (GROUPS, INCOME_COLUMNS, SIZE_COLUMNS, WEIGHT_COLUMNS,  # noqa: E402
                              MonthResult, month_sums, quarterly)

CACHE_VERSION = "v3"


def load_engine(root: Path):
    """Import the user's ocupacion_formal_sector_geih.py as a module."""
    path = root / "CODIGO_GEIH" / "ocupacion_formal_sector_geih.py"
    if not path.is_file():
        raise FileNotFoundError(f"Engine not found: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("geih_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RelativeEarnings:
    """Monthly weighted income sums by formality status and worker group."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.engine = load_engine(root)
        self.catalog = self.engine.load_catalog(root)
        self.classifier, _ = self.engine.read_classifier(root)

    # -- loading, mirroring engine.module_for_month but keeping income columns --
    def _module(self, month: str, module: str) -> pd.DataFrame:
        import pyarrow.parquet as pq
        wanted = {"NORM_PERSONA_ID", "NORM_DPTO_ID", "NORM_FEX_VALOR", "NORM_FEX_CAMPO", "NORM_CIIU_CODIGO",
                  "NORM_CIIU_VERSION", "GEIH_ESTADO_FILA", "SRC_DIRECTORIO", "SRC_SECUENCIA_P", "SRC_ORDEN",
                  "SRC_P6430", "SRC_OFICIO", "SRC_OFICIO_C8", "SRC_RAMA2D_R4", "SRC_P6090",
                  "SRC_P6100", "SRC_P6110", "SRC_P6450", "SRC_P6920", "SRC_P6930", "SRC_P6940", "SRC_P6765",
                  "SRC_P3065", "SRC_P3066", "SRC_P3067", "SRC_P3067S1", "SRC_P3067S2", "SRC_P3068",
                  "SRC_P6775", *SIZE_COLUMNS, *INCOME_COLUMNS, *WEIGHT_COLUMNS}
        rows = self.catalog.loc[self.catalog.periodo.eq(month) & self.catalog.modulo.eq(module)]
        parts = []
        for r in rows.itertuples(index=False):
            file = Path(str(r.ruta))
            names = set(pq.read_schema(file).names)
            # any expansion-factor column, whatever its case or vintage (fex_c_2011, FEX_C, ...)
            fex_like = {c for c in names if "FEX" in c.upper()}
            use = sorted((wanted | fex_like) & names)
            part = pd.read_parquet(file, columns=use)
            part["segmento"] = r.segmento
            parts.append(part)
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)
        ids = df["NORM_PERSONA_ID"].fillna("").astype("string").str.strip()
        ok = ids.ne("")
        if "GEIH_ESTADO_FILA" in df:
            ok &= df["GEIH_ESTADO_FILA"].fillna("").astype("string").eq("APTA")
        df, ids = df.loc[ok].copy(), ids.loc[ok]
        split = ids.str.split("|", n=2, expand=True)
        df["persona_id"] = split[0] + "|" + split[2]
        # AREA is a subdomain of CABECERA: keep one record per person (CABECERA first)
        df = df.sort_values("segmento", key=lambda s: s.eq("AREA").astype(int))
        df = df.drop_duplicates("persona_id", keep="first")
        return df.set_index("persona_id", drop=False)

    # -- one month --
    def month(self, month: str) -> MonthResult:
        occ = self._module(month, "OCUPADOS")
        if occ.empty:
            return MonthResult(month, "SIN_OCUPADOS")
        char = self._module(month, "CARACTERISTICAS")
        if month.startswith("2021"):
            # parallel frames in 2021: keep only the 2018 frame, never add both
            campo = occ.get("NORM_FEX_CAMPO", pd.Series("", index=occ.index)).fillna("").astype("string").str.upper()
            if campo.eq("FEX_C18").any():
                occ = occ.loc[campo.eq("FEX_C18")]
        cols = [c for c in ("SRC_P6090", "SRC_P6100", "SRC_P6110") if not char.empty and c in char]
        base = char.reindex(occ.index)[cols].copy() if cols else pd.DataFrame(index=occ.index)
        base.attrs["clasificador_funcion"] = self.classifier
        status = self.engine.guard_informal(occ, base, month)

        return month_sums(month, occ, status)


def main() -> None:
    parser = argparse.ArgumentParser(description="GEIH relative labor earnings informal/formal by worker group")
    parser.add_argument("--raiz", type=Path, default=Path(r"G:\My Drive\SENA\Procesamiento\GEIH"))
    parser.add_argument("--salida", type=Path, default=Path("datos/geih/ratio_ingresos_geih_v3.csv"))
    parser.add_argument("--desde", default="2007-01")
    parser.add_argument("--hasta", default=None)
    args = parser.parse_args()

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    calc = RelativeEarnings(args.raiz)
    # own cache file: months from the v2 cache lack the group sums and are not reused
    cache = args.salida.with_name(f"{args.salida.stem}_mensual_{CACHE_VERSION}.csv")
    done = pd.read_csv(cache, dtype={"periodo": str}) if cache.exists() else pd.DataFrame()
    if not done.empty:
        # months with zero weight, non-finite sums or errors are recomputed, never reused
        good = done.estado.eq("CLASIFICACION_NO_REPRESENTATIVA") | (   # SIN_FACTOR_EXPANSION is retried
            done.estado.eq("CALCULADO") & done.get("w_informal", 0).fillna(0).gt(0)
            & np.isfinite(done.get("wy_informal", np.nan)) & np.isfinite(done.get("wy_formal", np.nan)))
        done = done.loc[good]
    finished = set(done.periodo) if not done.empty else set()
    months = sorted(m for m in calc.catalog.periodo.unique()
                    if m >= args.desde and (args.hasta is None or m <= args.hasta))
    rows = done.to_dict("records") if not done.empty else []
    for i, month in enumerate(months, 1):
        if month in finished:
            continue
        try:
            result = calc.month(month).row()
        except Exception as exc:            # report and continue; never invent values
            result = {"periodo": month, "estado": f"ERROR: {type(exc).__name__}: {exc}"}
        rows.append(result)
        pd.DataFrame(rows).to_csv(cache, index=False)
        print(f"[{i:03d}/{len(months)}] {month}: {result['estado']}", flush=True)
    monthly = pd.DataFrame(rows).sort_values("periodo")
    out = quarterly(monthly)
    out.to_csv(args.salida, index=False)
    bad_calc = monthly.estado.eq("CALCULADO") & ~(monthly.get("w_informal", 0).fillna(0) > 0)
    errors = monthly.loc[~monthly.estado.isin(["CALCULADO", "CLASIFICACION_NO_REPRESENTATIVA"]) | bad_calc]
    summary = {"trimestres": len(out), "meses_calculados": int(monthly.estado.eq("CALCULADO").sum()),
               "meses_con_error": len(errors), "meses_problema": sorted(errors.periodo.tolist()),
               "salida": str(args.salida), "ratio_media": float(out.ratio.mean()) if len(out) else None}
    if len(out):
        summary["participacion_informal_media"] = {
            g: round(float(out[f"share_informal_{g}"].mean()), 3) for g in GROUPS}
        summary["fuente_tamano"] = sorted(set("|".join(out.size_source).split("|")))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
