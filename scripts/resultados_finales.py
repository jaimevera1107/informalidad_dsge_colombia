"""Final results of the baseline estimate: every quantity the model delivers, in one folder.

From the posterior draws of the baseline (labor-only informal sector, statutory non-wage
costs) the Kalman smoother is run for a random subset of draws and the following are
computed for every quarter, with posterior median and 90% band:
    informal share of output, informality rate (survey noise removed), earnings ratio,
    informal value added per worker relative to formal, informal and formal output in
    pesos, informal and formal employment, value added per worker in each sector,
    the cyclical component of output and the seven structural shocks.
Annual aggregates, parameters, steady state, variance decomposition, fit, sensitivity,
comparison with the literature, identification checks and the capital share are copied
or computed next to them. Column names are in Spanish; see resultados_finales/LEEME.md.

Usage:
    python scripts/resultados_finales.py --draws 400 --salida resultados_finales
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2.data import DataConfig  # noqa: E402
from m2.model import STATE_INDEX, VARIABLES  # noqa: E402
from m2.runner import M2Estimation, RunConfig  # noqa: E402
from m2.statespace import KalmanFilter  # noqa: E402

BASE = "solo_trabajo_cuna_legal"
QUANTILES = ((0.05, "p05"), (0.5, "mediana"), (0.95, "p95"))
SHOCK_STATES = {"choque_crecimiento_tendencia_mu": "mu", "choque_gasto_publico_zg": "zg",
                "choque_eficiencia_inversion_lnnu": "lv", "choque_oferta_laboral_lnpsi": "lpsi",
                "choque_cuna_formalidad_omega": "om", "choque_absorcion_x": "zx",
                "choque_productividad_informal_zr": "zr"}


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "datos" / "datos_dsge_estimacion.csv"

    @property
    def earnings(self) -> Path:
        return self.root / "datos" / "geih" / "ratio_va_0354.csv"

    @property
    def offsets(self) -> Path:
        return self.root / "datos" / "costos_laborales" / "costo_no_salarial.csv"

    @property
    def results(self) -> Path:
        return self.root / "resultados_m2"


class BaselineDraws:
    """Rebuilds the baseline posterior and smooths the states for a subset of draws."""

    def __init__(self, paths: Paths, n_draws: int, seed: int = 2026) -> None:
        self.paths = paths
        config = RunConfig(
            data_path=paths.data, output=Path("/tmp/no_usado"),
            data_config=DataConfig(start="1993Q1", informality_in_levels=True),
            earnings_path=paths.earnings, earnings_column="ratio_va", model="labor_only",
            alpha=0.354, delta=0.0102, earnings_offset_path=paths.offsets)
        self.est = M2Estimation(config)
        draws = pd.read_csv(paths.results / BASE / "posterior_draws_thin5.csv")
        rng = np.random.default_rng(seed)
        self.draws = draws.iloc[rng.choice(len(draws), size=min(n_draws, len(draws)), replace=False)]
        raw = pd.read_csv(paths.data, encoding="utf-8-sig").set_index("quarter")
        q = self.est.obs.quarters.astype(str)
        self.quarters = np.asarray(q)
        self.gdp = raw.loc[q, "pib"].to_numpy(float)
        self.employment = raw.loc[q, "ocupados"].to_numpy(float)

    def one(self, params: dict[str, float]) -> dict[str, np.ndarray]:
        post = self.est.post
        post._set_break(params)
        model, space = post.state_space(params)
        out = KalmanFilter(space).run(post.adjusted_data(model), keep=True, smooth=True)
        mapping = model.solve().full_map()
        states = out.smoothed_states[:, :len(STATE_INDEX)]
        dev = lambda name: states @ mapping[VARIABLES.index(name)]  # noqa: E731
        obs = self.est.obs
        s_bar = post.cal.s_bar
        season = np.array([obs.seasonal_means["obs_s_level"][p.quarter] for p in obs.quarters])
        logit_s = np.log(s_bar / (1 - s_bar)) + season + dev("lni") - dev("lnf")
        s = 1.0 / (1.0 + np.exp(-logit_s))
        loading = sum(c * mapping[VARIABLES.index(v)] for v, c in model.earnings_loadings().items())
        ratio = np.exp(model.earnings_level() + states @ loading)
        theta = model.output_share(ratio, s)
        alpha = post.cal.alpha
        informal_output, formal_output = theta * self.gdp, (1 - theta) * self.gdp
        informal_jobs, formal_jobs = s * self.employment, (1 - s) * self.employment
        result = {
            "participacion_informal_pib": theta,
            "tasa_informalidad_laboral": s,
            "razon_ingresos_R": ratio,
            "productividad_relativa_informal": (1 - alpha) * ratio,
            "pib_informal_mmillones_2015": informal_output,
            "pib_formal_mmillones_2015": formal_output,
            "ocupados_informales": informal_jobs,
            "ocupados_formales": formal_jobs,
            "va_por_trabajador_informal_millones": 1e3 * informal_output / informal_jobs,
            "va_por_trabajador_formal_millones": 1e3 * formal_output / formal_jobs,
            "componente_ciclico_pib": dev("ly"),
        }
        for label, state in SHOCK_STATES.items():
            result[label] = dev(state)
        return result

    def all(self) -> dict[str, np.ndarray]:
        stacked: dict[str, list[np.ndarray]] = {}
        for _, row in self.draws.iterrows():
            for key, value in self.one(row.to_dict()).items():
                stacked.setdefault(key, []).append(value)
        return {k: np.array(v) for k, v in stacked.items()}


class FinalTables:
    """Turns draws into quarterly and annual tables and collects the other results."""

    SUM_VARIABLES = ("pib_informal_mmillones_2015", "pib_formal_mmillones_2015")
    LEVEL_VARIABLES = ("ocupados_informales", "ocupados_formales")

    def __init__(self, draws: BaselineDraws, paths: Paths, out: Path) -> None:
        self.d = draws
        self.paths = paths
        self.out = out
        out.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _bands(name: str, values: np.ndarray) -> dict[str, np.ndarray]:
        return {f"{name}_{lab}": np.quantile(values, q, axis=0) for q, lab in QUANTILES}

    def quarterly(self, sims: dict[str, np.ndarray]) -> pd.DataFrame:
        frame = pd.DataFrame({"trimestre": self.d.quarters,
                              "pib_observado_mmillones_2015": self.d.gdp,
                              "ocupados_observados": self.d.employment,
                              "tasa_informalidad_observada": self.d.est.obs.informality_rate})
        for name, values in sims.items():
            for col, band in self._bands(name, values).items():
                frame[col] = band
        return frame

    def annual(self, sims: dict[str, np.ndarray]) -> pd.DataFrame:
        years = np.array([int(q[:4]) for q in self.d.quarters])
        complete = [y for y in np.unique(years) if (years == y).sum() == 4]
        rows = []
        for y in complete:
            m = years == y
            row: dict[str, float] = {"anio": y, "pib_observado_mmillones_2015": self.d.gdp[m].sum()}
            informal = sims["pib_informal_mmillones_2015"][:, m].sum(axis=1)
            share = informal / self.d.gdp[m].sum()          # annual share = annual informal / annual GDP
            for name, values in (("participacion_informal_pib", share),
                                 ("pib_informal_mmillones_2015", informal),
                                 ("pib_formal_mmillones_2015", sims["pib_formal_mmillones_2015"][:, m].sum(axis=1)),
                                 ("tasa_informalidad_laboral", sims["tasa_informalidad_laboral"][:, m].mean(axis=1)),
                                 ("razon_ingresos_R", sims["razon_ingresos_R"][:, m].mean(axis=1)),
                                 ("productividad_relativa_informal",
                                  sims["productividad_relativa_informal"][:, m].mean(axis=1)),
                                 ("ocupados_informales", sims["ocupados_informales"][:, m].mean(axis=1))):
                for q, lab in QUANTILES:
                    row[f"{name}_{lab}"] = float(np.quantile(values, q))
            rows.append(row)
        return pd.DataFrame(rows)

    def steady_state(self) -> pd.DataFrame:
        table = pd.read_csv(self.paths.results / BASE / "posterior_summary.csv").set_index("parameter")
        params = {k: float(v) for k, v in table["post_median"].items()}
        model = self.d.est.post.state_space(params)[0]
        ss, cal = model.ss, self.d.est.post.cal
        yf, yi = ss.y * (1 - params["theta"]), ss.y * params["theta"]
        ratio = float(np.exp(model.earnings_level()))
        k_tilde = ss.k / np.exp(cal.gamma)
        rows = [
            ("theta", params["theta"], "Participacion informal en el PIB de estado estacionario"),
            ("R", ratio, "Ingreso informal por trabajador / costo laboral formal, estado estacionario"),
            ("productividad_relativa", (1 - cal.alpha) * ratio, "VA por trabajador informal / formal"),
            ("r_trimestral", ss.rk, "Tasa de renta del capital (trimestral)"),
            ("K_sobre_Yf_anual", k_tilde / yf / 4, "Capital / PIB formal (anios)"),
            ("K_sobre_Y_anual", k_tilde / ss.y / 4, "Capital / PIB total (anios)"),
            ("I_sobre_Y_modelo", ss.i / ss.y, "Inversion / PIB implicada por el modelo"),
            ("xbar", ss.xbar, "Residuo de no aditividad de volumenes encadenados"),
            ("tasa_informalidad_media", cal.s_bar, "Tasa de informalidad media (calibrada)"),
            ("alpha", cal.alpha, "Participacion del capital en el PIB formal (DOLS)"),
            ("delta_trimestral", cal.delta, "Depreciacion trimestral (4% anual)"),
            ("crecimiento_tendencia_trimestral", cal.gamma, "Crecimiento medio trimestral de la tendencia"),
            ("yi_sobre_yf", yi / yf, "PIB informal / PIB formal, estado estacionario"),
        ]
        return pd.DataFrame(rows, columns=["objeto", "valor", "descripcion"])

    def copy_existing(self) -> None:
        res = self.paths.results
        copies = {
            "03_parametros_posterior.csv": res / BASE / "posterior_summary.csv",
            "05_descomposicion_varianza.csv": res / BASE / "variance_decomposition.csv",
            "06_ajuste_un_paso.csv": res / BASE / "fit_one_step.csv",
            "07_resumen_corrida.json": res / BASE / "run_summary.json",
            "09_comparacion_literatura.csv": res / "comparacion" / "comparacion_referencias.csv",
            "10_participacion_capital_alfa.csv": res / "alfa" / "capital_share.csv",
            "11_identificacion_perfil.csv": res / "identificacion_solo_trabajo" / "perfil_theta.csv",
            "12_identificacion_simulacion.csv": res / "identificacion_solo_trabajo" / "simulacion.csv",
            "13_identificacion_trayectorias_theta_fijo.csv":
                res / "identificacion_solo_trabajo" / "trayectorias_theta_fijo.csv",
        }
        for name, src in copies.items():
            if src.exists():
                shutil.copy2(src, self.out / name)
            else:
                print(f"no encontrado (se omite): {src}")
        figures = self.out / "figuras"
        figures.mkdir(exist_ok=True)
        # Figures of the estimation results, if the paper sources are present (they are not
        # part of the public repository); the figures of the estimated series are always
        # regenerated from the files just written, so they follow each quarterly update.
        for fig in (ROOT / "paper" / "figures").glob("*.pdf"):
            if fig.stem not in ("fig_estimated_series", "fig_shocks"):
                shutil.copy2(fig, figures / fig.name)
        sys.path.insert(0, str(ROOT / "scripts"))
        from series_paper import EstimatedSeries
        series = EstimatedSeries(self.out, figures, formats=("pdf", "png"))
        for build in (series.share_figure, series.series_figure, series.shocks_figure):
            build()

    def sensitivity(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        from tabla_sensibilidad import SensitivityTable, SPECIFICATIONS
        table = SensitivityTable(self.paths.results)
        rows = [r for r in (table.row(s) for s in SPECIFICATIONS) if r and not r.get("panel")]
        frame = pd.DataFrame(rows).rename(columns={"label": "especificacion", "rho": "R_mediana",
                                                   "theta": "theta_ee_mediana_[p05,p95]",
                                                   "first": "1993Q2_mediana_[p05,p95]",
                                                   "last": "2026Q2_mediana_[p05,p95]"})
        frame["especificacion"] = (frame["especificacion"].str.replace(r"\\bar\\tau", "tau", regex=True)
                                   .str.replace(r"\\(alpha|delta|eta)", r"\1", regex=True)
                                   .str.replace(r"[\$\\{}^]|dagger", "", regex=True)
                                   .str.replace("--", "-", regex=False))
        frame.to_csv(self.out / "08_sensibilidad.csv", index=False)

    def write(self) -> None:
        sims = self.d.all()
        self.quarterly(sims).to_csv(self.out / "01_series_trimestrales.csv", index=False)
        self.annual(sims).to_csv(self.out / "02_series_anuales.csv", index=False)
        self.steady_state().to_csv(self.out / "04_estado_estacionario.csv", index=False)
        self.copy_existing()
        self.sensitivity()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--salida", type=Path, default=ROOT / "resultados_finales")
    args = parser.parse_args()
    paths = Paths(ROOT)
    FinalTables(BaselineDraws(paths, args.draws), paths, args.salida).write()
    annual = pd.read_csv(args.salida / "02_series_anuales.csv")
    print(annual[["anio", "participacion_informal_pib_mediana", "pib_informal_mmillones_2015_mediana",
                  "tasa_informalidad_laboral_mediana", "productividad_relativa_informal_mediana"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
