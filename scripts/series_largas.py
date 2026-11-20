"""Annual long series 1905-today: GRECO backcast spliced onto the estimation dataset.

Each series keeps the level of the current dataset (DANE base 2015 and ICML) from
1993 on and is carried backwards with the growth rates of the GRECO series
(retropolation by growth rates, linked in the first common year, 1993):

    x_t = x_1993 * greco_t / greco_1993,   t < 1993

Series and sources
    pib          real GDP: DANE (1993-) <- GRECO/CEPAL/López (1905-1992)
    fbkf         gross fixed capital formation: DANE (1993-) <- GRECO Table 3A (1925-1992)
    capital      fixed capital stock by perpetual inventory with a constant depreciation
                 rate (default 4% a year), run forward from 1925 with a steady-state initial
                 stock K0 = I0 / (g + delta), g = mean investment growth of the first years.
                 The GRECO stock is not used as a level: it mixes inventories and a different
                 depreciation (3.1%), and with DANE investment it implies an unstable K/Y.
    ocupados     employment: ICML/DANE (1993-) <- Flórez (1998) in GRECO Table 1.2 (1951-1992)
    pea_florez, poblacion: GRECO levels as published (persons), not linked
    c, g, informalidad: only from 1993 (no annual backcast source in GRECO)
The GRECO depreciation implied by Table 3A (2% buildings, 5% machinery, 3.1% on total
fixed capital) is reported for reference; the stock here uses the rate passed in.

Usage:
    python scripts/series_largas.py --datos datos/datos_dsge_estimacion.csv ^
        --greco datos/greco --salida datos/series_anuales_largas.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LINK_YEAR = 1993


@dataclass(frozen=True)
class LongSeriesConfig:
    depreciation: float = 0.04          # annual
    initial_growth_years: int = 10      # years used for the steady-state initial stock
    link_year: int = LINK_YEAR

    def __post_init__(self) -> None:
        if not 0.0 < self.depreciation < 0.3:
            raise ValueError("depreciation must lie in (0, 0.3)")


class CurrentData:
    """Annual aggregates of the quarterly estimation dataset (complete years only)."""

    def __init__(self, path: Path) -> None:
        q = pd.read_csv(path, encoding="utf-8-sig")
        q["year"] = q["quarter"].astype(str).str[:4].astype(int)
        complete = q.groupby("year")["pib"].count().loc[lambda s: s.eq(4)].index
        q = q.loc[q.year.isin(complete)]
        self.annual = q.groupby("year").agg(
            pib=("pib", "sum"), c=("c", "sum"), fbkf=("i", "sum"), g=("g", "sum"),
            ocupados=("ocupados", "mean"), informalidad=("informalidad_ponderada_ocupados", "mean"))


class GrecoData:
    """The CSV files written by extraer_greco.py."""

    def __init__(self, folder: Path) -> None:
        read = lambda name: pd.read_csv(folder / name).set_index("year")  # noqa: E731
        self.series = read("greco_tabla2a_series.csv")
        self.investment = read("greco_tabla3a_inversion.csv")
        self.assets = read("greco_tabla3a_capital_por_activo.csv")
        self.employment = read("greco_tabla1_2_empleo.csv")

    def fixed_capital(self) -> pd.Series:
        return self.assets["capital_construccion_1975"] + self.assets["capital_maquinaria_1975"]

    def implied_depreciation(self) -> dict[str, float]:
        a = self.assets
        lag = a.shift(1)
        rates = {
            "construccion": a.depreciacion_construccion_1975 / lag.capital_construccion_1975,
            "maquinaria": a.depreciacion_maquinaria_1975 / lag.capital_maquinaria_1975,
            "capital_fijo_total": (a.depreciacion_construccion_1975 + a.depreciacion_maquinaria_1975)
            / (lag.capital_construccion_1975 + lag.capital_maquinaria_1975),
        }
        return {k: round(float(v.mean()), 4) for k, v in rates.items()}


class LongSeriesBuilder:
    """Splices GRECO growth rates onto the current levels and builds the capital stock."""

    def __init__(self, current: CurrentData, greco: GrecoData, config: LongSeriesConfig) -> None:
        self.cur = current.annual
        self.greco = greco
        self.cfg = config

    def _splice(self, current: pd.Series, backcast: pd.Series) -> tuple[pd.Series, pd.Series]:
        link = self.cfg.link_year
        if link not in current.index or link not in backcast.index or not np.isfinite(backcast[link]):
            raise ValueError(f"link year {link} missing in one of the series")
        back = current[link] * backcast.loc[:link - 1] / backcast[link]
        out = pd.concat([back.dropna(), current.loc[link:]])
        source = pd.Series(np.where(out.index < link, "GRECO (tasas de crecimiento)", "actual"), index=out.index)
        return out, source

    def _capital(self, fbkf: pd.Series) -> pd.Series:
        """Perpetual inventory from the first investment year with a steady-state initial stock."""
        d, n = self.cfg.depreciation, self.cfg.initial_growth_years
        inv = fbkf.dropna()
        growth = max(float(np.log(inv.iloc[n] / inv.iloc[0]) / n), 0.0)
        k = pd.Series(np.nan, index=inv.index)
        k.iloc[0] = inv.iloc[0] / (growth + d)
        for j in range(1, len(inv)):
            k.iloc[j] = (1.0 - d) * k.iloc[j - 1] + inv.iloc[j - 1]
        return k

    def build(self) -> pd.DataFrame:
        pib, src_pib = self._splice(self.cur["pib"], self.greco.series["pib_1975"])
        fbkf, src_fbkf = self._splice(self.cur["fbkf"], self.greco.investment["fbkf_1975"])
        ocup, src_ocup = self._splice(self.cur["ocupados"], self.greco.employment["ocupados_florez"].dropna())
        frame = pd.DataFrame({"pib": pib, "fbkf": fbkf, "ocupados": ocup})
        frame = frame.join(self.cur[["c", "g", "informalidad"]], how="left")
        frame["capital"] = self._capital(frame["fbkf"])
        frame = frame.join(self.greco.series[["pea_florez", "poblacion"]], how="left")
        frame["fuente_pib"], frame["fuente_fbkf"], frame["fuente_ocupados"] = src_pib, src_fbkf, src_ocup
        frame.index.name = "year"
        return frame.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=Path("datos/datos_dsge_estimacion.csv"))
    parser.add_argument("--greco", type=Path, default=Path("datos/greco"))
    parser.add_argument("--salida", type=Path, default=Path("datos/series_anuales_largas.csv"))
    parser.add_argument("--depreciacion", type=float, default=0.04)
    args = parser.parse_args()
    greco = GrecoData(args.greco)
    frame = LongSeriesBuilder(CurrentData(args.datos), greco, LongSeriesConfig(args.depreciacion)).build()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.salida, index=False)
    print(json.dumps({"years": [int(frame.year.min()), int(frame.year.max())],
                      "depreciacion_usada": args.depreciacion,
                      "depreciacion_implicita_greco": greco.implied_depreciation(),
                      "salida": str(args.salida)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
