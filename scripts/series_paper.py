"""Figures and table of the estimated series for the paper.

Reads ``resultados_finales/01_series_trimestrales.csv`` and ``02_series_anuales.csv``
(written by ``scripts/resultados_finales.py``) and writes:

    figures/fig_estimated_series.pdf  informal and formal value added, value added per
                                      worker, relative productivity and earnings ratio,
                                      informal and formal employment (median and 90% band;
                                      levels as four-quarter moving averages, because the
                                      national accounts are not seasonally adjusted)
    figures/fig_shocks.pdf            the seven smoothed structural shocks
    tables/estimated_series.tex       annual estimated series, 1994 to the last full year

It also writes ``fig_participacion_informal`` (informal share and labor informality)
for the repository. ``scripts/resultados_finales.py`` calls this module so that the
figures in ``resultados_finales/figuras`` follow each quarterly update.

Usage:
    python scripts\\series_paper.py --results resultados_finales --out paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
BLUE, ORANGE, GREEN = "#2563eb", "#d97706", "#059669"
BAND_ALPHA = 0.22
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6, "legend.frameon": False,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

SHOCKS = (
    ("choque_productividad_informal_zr", r"Informal productivity $z^r_t$"),
    ("choque_cuna_formalidad_omega", r"Formality wedge $\omega_t$"),
    ("choque_oferta_laboral_lnpsi", r"Labor supply $\ln\psi_t$"),
    ("choque_crecimiento_tendencia_mu", r"Trend growth $\mu_t$"),
    ("choque_eficiencia_inversion_lnnu", r"Investment efficiency $\ln\nu_t$"),
    ("choque_gasto_publico_zg", r"Government spending $z^g_t$"),
    ("choque_absorcion_x", r"Absorption $x_t$"),
)


class EstimatedSeries:
    """Builds the paper's figures and table from the final-results files."""

    def __init__(self, results: Path, figures: Path, tables: Path | None = None,
                 formats: tuple[str, ...] = ("pdf",)) -> None:
        quarterly = results / "01_series_trimestrales.csv"
        annual = results / "02_series_anuales.csv"
        for path in (quarterly, annual):
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found; run scripts/resultados_finales.py first")
        self.q = pd.read_csv(quarterly)
        self.q["date"] = pd.PeriodIndex(self.q.trimestre, freq="Q").to_timestamp(how="end")
        self.a = pd.read_csv(annual)
        self.figures = figures
        self.tables = tables
        self.formats = formats
        self.figures.mkdir(parents=True, exist_ok=True)
        if self.tables is not None:
            self.tables.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: plt.Figure, stem: str) -> Path:
        for fmt in self.formats:
            fig.savefig(self.figures / f"{stem}.{fmt}")
        plt.close(fig)
        return self.figures / f"{stem}.{self.formats[0]}"

    def share_figure(self) -> Path:
        fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
        self._band(a, "participacion_informal_pib", BLUE, "Informal share of output", 100)
        a.set_title("A. Informal share of output (%), median and 90% band")
        q = self.q
        b.plot(q.date, 100 * q.tasa_informalidad_observada, color=MUTED, lw=1, label="Observed (ICML / GEIH)")
        self._band(b, "tasa_informalidad_laboral", ORANGE, "Smoothed", 100)
        b.set_title("B. Labor informality rate (%)")
        b.legend(loc="upper right")
        last = q.iloc[-1]
        a.annotate(f"{last.trimestre}: {100 * last.participacion_informal_pib_mediana:.1f}%",
                   (last.date, 100 * last.participacion_informal_pib_mediana), xytext=(-95, 32),
                   textcoords="offset points", fontsize=8, color=INK,
                   arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
        fig.tight_layout()
        return self._save(fig, "fig_participacion_informal")

    def _band(self, ax: plt.Axes, name: str, color: str, label: str, scale: float = 1.0,
              smooth: bool = False) -> None:
        q = self.q
        cols = {k: q[f"{name}_{k}"] for k in ("p05", "mediana", "p95")}
        if smooth:      # four-quarter moving average removes the seasonal pattern of the levels
            cols = {k: v.rolling(4).mean() for k, v in cols.items()}
        ax.fill_between(q.date, scale * cols["p05"], scale * cols["p95"], color=color, alpha=BAND_ALPHA, lw=0)
        ax.plot(q.date, scale * cols["mediana"], color=color, label=label)
        ax.set_xlim(q.date.iloc[0], q.date.iloc[-1])

    def series_figure(self) -> Path:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
        (a, b), (c, d) = axes

        self._band(a, "pib_formal_mmillones_2015", BLUE, "Formal", 1e-3, smooth=True)
        self._band(a, "pib_informal_mmillones_2015", ORANGE, "Informal", 1e-3, smooth=True)
        a.set_title("A. Value added, trillion 2015 pesos per quarter")
        a.legend(loc="upper left")

        self._band(b, "va_por_trabajador_formal_millones", BLUE, "Formal", smooth=True)
        self._band(b, "va_por_trabajador_informal_millones", ORANGE, "Informal", smooth=True)
        b.set_title("B. Value added per worker, million 2015 pesos per quarter")
        b.legend(loc="center left")
        b.set_ylim(0, None)

        self._band(c, "productividad_relativa_informal", GREEN, "Relative productivity (value added per worker)")
        self._band(c, "razon_ingresos_R", ORANGE, "Earnings ratio $R_t$ (smoothed)")
        c.set_title("C. Informal relative to formal")
        c.legend(loc="lower left")
        c.set_ylim(0.1, 0.4)

        self._band(d, "ocupados_informales", ORANGE, "Informal", 1e-6)
        self._band(d, "ocupados_formales", BLUE, "Formal", 1e-6)
        d.set_title("D. Employment, million persons")
        d.legend(loc="upper left")
        for ax in axes.flat:
            ax.tick_params(labelsize=8)
        fig.tight_layout()
        return self._save(fig, "fig_estimated_series")

    def shocks_figure(self) -> Path:
        fig, axes = plt.subplots(4, 2, figsize=(7.2, 7.6), sharex=True)
        flat = axes.flat
        for ax, (name, title) in zip(flat, SHOCKS):
            self._band(ax, name, BLUE, title, 100)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(title)
            ax.tick_params(labelsize=8)
        axes.flat[-1].axis("off")
        fig.supylabel("% deviation from steady state", fontsize=8.5, color=INK)
        fig.tight_layout()
        return self._save(fig, "fig_shocks")

    def annual_table(self) -> Path:
        if self.tables is None:
            raise ValueError("no tables directory was given")
        a = self.a
        lines = [
            r"\begin{tabular}{rrrrrrrr}",
            r"\toprule",
            r" & GDP & \multicolumn{2}{c}{Informal share of output (\%)} & Informal VA & Formal VA"
            r" & Labor & Relative \\",
            r"Year & (trillion) & Median & 90\% band & (trillion) & (trillion) & informality (\%)"
            r" & productivity \\",
            r"\midrule",
        ]
        for _, r in a.iterrows():
            band = (f"[{100 * r.participacion_informal_pib_p05:.1f}, "
                    f"{100 * r.participacion_informal_pib_p95:.1f}]")
            lines.append(
                f"{int(r.anio)} & {r.pib_observado_mmillones_2015 / 1e3:,.1f} & "
                f"{100 * r.participacion_informal_pib_mediana:.1f} & {band} & "
                f"{r.pib_informal_mmillones_2015_mediana / 1e3:,.1f} & "
                f"{r.pib_formal_mmillones_2015_mediana / 1e3:,.1f} & "
                f"{100 * r.tasa_informalidad_laboral_mediana:.1f} & "
                f"{r.productividad_relativa_informal_mediana:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        path = self.tables / "estimated_series.tex"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("resultados_finales"))
    parser.add_argument("--out", type=Path, default=Path("paper"))
    args = parser.parse_args()
    builder = EstimatedSeries(args.results, args.out / "figures", args.out / "tables")
    for path in (builder.series_figure(), builder.shocks_figure(), builder.annual_table()):
        print("written", path)


if __name__ == "__main__":
    main()
