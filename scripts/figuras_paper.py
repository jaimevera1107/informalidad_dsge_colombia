"""Regenerate the figures and tables of the paper from the M2 result files.

Usage: python scripts/figuras_paper.py --results resultados_m2 --out paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def informal_share(rho, s):
    """Equal technologies: output share = labor-income share."""
    return rho * s / ((1 - s) + rho * s)


def share_labor_only(ratio, s, alpha):
    """Labor-only informal sector: informal earnings ~ value added; formal wage = (1 - alpha) VA."""
    x = (1 - alpha) * ratio * s / (1 - s)
    return x / (1 + x)


class PaperFigures:
    def __init__(self, results: Path, out: Path) -> None:
        self.res, self.out = results, out
        self.lp = results / "solo_trabajo_cuna_legal"   # baseline: labor-only, legal non-wage costs, all data + GEIH
        self.eq = results / "con_ingresos"        # equal technologies (upper bound)
        self.agg = results / "desde_1993"        # aggregates only
        (out / "figures").mkdir(parents=True, exist_ok=True)
        (out / "tables").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load(folder: Path) -> pd.DataFrame:
        d = pd.read_csv(folder / "informal_output_share.csv")
        d["date"] = pd.PeriodIndex(d["quarter"], freq="Q").to_timestamp()
        return d

    def share_path(self) -> None:
        d, e, g = self._load(self.lp), self._load(self.eq), self._load(self.agg)
        fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
        a.fill_between(d.date, 100 * d.informal_output_share_p05, 100 * d.informal_output_share_p95,
                       color=BLUE, alpha=0.25, lw=0)
        a.plot(d.date, 100 * d.informal_output_share_median, color=BLUE, label="Labor-only informal sector (baseline)")
        a.plot(e.date, 100 * e.informal_output_share_median, color=BLUE, ls="--", lw=1.5,
               label="Equal technologies (upper bound)")
        a.fill_between(g.date, 100 * g.informal_output_share_p05, 100 * g.informal_output_share_p95,
                       color=ORANGE, alpha=0.12, lw=0)
        a.plot(g.date, 100 * g.informal_output_share_median, color=ORANGE, ls=":", label="Aggregates only, no earnings")
        a.set_ylabel("% of GDP")
        a.set_title("A. Informal-sector share of output", loc="left", fontsize=10)
        a.set_ylim(0, 50)
        a.legend(loc="lower left", fontsize=8)
        b.plot(d.date, 100 * d.informality_rate_observed, color=MUTED, lw=1, label="ICML series")
        b.plot(d.date, 100 * d.informality_rate_clean_median, color=ORANGE, label="Smoothed, survey noise removed")
        b.axvspan(d.date.iloc[0], pd.Timestamp("2001-01-01"), color=GRID, alpha=0.6, lw=0)
        b.text(pd.Timestamp("1993-09-01"), 56, "ICML\nreconstruction", color=MUTED, fontsize=8)
        b.set_ylabel("% of employed")
        b.set_title("B. Labor-informality rate", loc="left", fontsize=10)
        b.legend(loc="upper right")
        fig.savefig(self.out / "figures" / "fig_share_path.pdf")
        plt.close(fig)

    def profiles(self) -> None:
        variants = [
            ("2005Q2-2026Q2", self.res / "base" / "profile_theta.csv", BLUE, "-"),
            ("2001Q2-2026Q2", self.res / "desde_1993" / "profile_theta_desde_2001.csv", AQUA, "-"),
            ("1993Q2-2026Q2, all aggregate data",
             self.res / "desde_1993" / "profile_theta.csv", INK, "-"),
            ("1993Q2-2026Q2, ICML labor series 1993-2000 excluded",
             self.res / "largo_plazo_1993" / "profile_theta.csv", ORANGE, "--"),
        ]
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for label, path, color, ls in variants:
            p = pd.read_csv(path)
            ax.plot(p.theta, p.loglik - p.loglik.max(), color=color, ls=ls, marker="o", ms=5, label=label)
        ax.axhline(-1.92, color=MUTED, lw=1, ls=":")
        ax.text(0.80, -1.5, "-1.92", color=MUTED, fontsize=8, ha="right")
        ax.set_xlabel(r"Steady-state informal output share $\theta$ (fixed)")
        ax.set_ylabel("Profile log likelihood minus maximum")
        ax.legend(loc="lower left", fontsize=8)
        fig.savefig(self.out / "figures" / "fig_profiles.pdf")
        plt.close(fig)

    def prior_posterior(self) -> None:
        draws = pd.read_csv(self.agg / "posterior_draws_thin5.csv")["theta"]
        draws_w = pd.read_csv(self.lp / "posterior_draws_thin5.csv")["theta"]
        m, s = 0.35, 0.10
        k = m * (1 - m) / s**2 - 1
        grid = np.linspace(0.001, 0.999, 400)
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.hist(draws, bins=60, density=True, color=ORANGE, alpha=0.40, label="Posterior, aggregates only")
        ax.hist(draws_w, bins=60, density=True, color=BLUE, alpha=0.70, label="Posterior, baseline (with relative earnings)")
        ax.plot(grid, stats.beta(m * k, (1 - m) * k).pdf(grid), color=INK, lw=1.5, label="Prior Beta(mean 0.35, sd 0.10)")
        ax.set_yscale("log"); ax.set_ylim(0.05, None)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("Density")
        ax.legend()
        fig.savefig(self.out / "figures" / "fig_prior_posterior.pdf")
        plt.close(fig)

    def anchor_map(self) -> None:
        import json
        d = self._load(self.lp)
        alpha = json.loads((self.lp / "run_summary.json").read_text())["calibration"]["alpha"]
        grid = np.linspace(0.10, 0.90, 300)
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        for color, idx in ((BLUE, 0), (ORANGE, -1)):
            s = d.informality_rate_clean_median.iloc[idx]
            q = d.quarter.iloc[idx]
            ax.plot(grid, 100 * share_labor_only(grid, s, alpha), color=color,
                    label=f"{q}, labor-only (informality {100 * s:.1f}%)")
            ax.plot(grid, 100 * informal_share(grid, s), color=color, ls="--", lw=1.2,
                    label=f"{q}, equal technologies")
        for (path, label), y in zip(((self.agg, "aggregates only"), (self.lp, "with relative earnings")), (55, 66)):
            rho = json.loads((path / "run_summary.json").read_text())["rho_tilde_median"]
            ax.axvline(rho, color=MUTED, lw=1, ls=":")
            ax.text(rho + 0.01, y, f"{label}\n" + r"$R$" + f" = {rho:.2f}", color=MUTED, fontsize=8)
        ax.set_xlabel(r"Informal earnings relative to the gross formal wage, $R$")
        ax.set_ylabel("Informal output share (%)")
        ax.legend(loc="lower right", fontsize=7)
        fig.savefig(self.out / "figures" / "fig_anchor_map.pdf")
        plt.close(fig)

    def earnings(self) -> None:
        geih = self.res.parent / "datos" / "geih"
        path = next((geih / n for n in ("ratio_ingresos_geih_v3.csv", "ratio_ingresos_geih_v2.csv")
                     if (geih / n).exists()), None)
        if path is None:
            raise FileNotFoundError(f"no GEIH earnings file in {geih}")
        e = pd.read_csv(path)
        e["date"] = pd.PeriodIndex(e.quarter, freq="Q").to_timestamp()
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        ax.plot(e.date, e.ratio, color=BLUE, marker="o", ms=3, lw=1.5, label="Mean earnings per worker")
        ax.plot(e.date, e.ratio_hourly, color=AQUA, marker="o", ms=3, lw=1.5, label="Per hour")
        va = geih / "ratio_va_0354.csv"
        if va.exists():
            v = pd.read_csv(va)
            v["date"] = pd.PeriodIndex(v.quarter, freq="Q").to_timestamp()
            ax.plot(v.date, v.ratio_va, color=ORANGE, marker="o", ms=3, lw=1.5,
                    label="Value added per worker / formal employee wage (model input)")
        ax.axvline(pd.Timestamp("2022-01-01"), color=MUTED, lw=1, ls=":")
        ax.text(pd.Timestamp("2022-03-01"), 0.52, "new informality\ndefinition", color=MUTED, fontsize=8)
        ax.set_ylabel("Informal / formal mean labor earnings")
        ax.legend(loc="lower left", fontsize=8)
        fig.savefig(self.out / "figures" / "fig_earnings.pdf")
        plt.close(fig)

    def benchmarks(self) -> None:
        import shutil
        src = self.res / "comparacion" / "fig_benchmarks.pdf"
        if not src.exists():
            raise FileNotFoundError(f"{src}: run scripts/comparar_referencias.py first")
        shutil.copy2(src, self.out / "figures" / "fig_benchmarks.pdf")

    def all(self) -> None:
        self.share_path(); self.profiles(); self.prior_posterior(); self.anchor_map(); self.earnings()
        self.benchmarks()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    PaperFigures(a.results, a.out).all()
