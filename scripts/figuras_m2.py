"""Figures for the M2 results (labels in Spanish, for the article)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # project root: package m2

from m2.informal import informal_share
from m2.priors import PRIORS

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


class M2Figures:
    def __init__(self, results: Path) -> None:
        self.res = results
        self.out = results / "figuras"
        self.out.mkdir(exist_ok=True)
        self.share = pd.read_csv(results / "informal_output_share.csv")
        self.share["date"] = pd.PeriodIndex(self.share["quarter"], freq="Q").to_timestamp()
        self.summary = pd.read_csv(results / "posterior_summary.csv").set_index("parameter")

    def informal_share(self) -> None:
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
        d = self.share
        top.fill_between(d["date"], 100 * d["informal_output_share_p05"], 100 * d["informal_output_share_p95"],
                         color=BLUE, alpha=0.18, lw=0, label="Banda del 90 %")
        top.plot(d["date"], 100 * d["informal_output_share_median"], color=BLUE, label="Mediana posterior")
        top.set_ylabel("% del PIB")
        top.set_title("A. Participación del sector informal en el PIB (M2)", loc="left", fontsize=10)
        top.legend(loc="upper right")
        bottom.plot(d["date"], 100 * d["informality_rate_observed"], color=MUTED, lw=1, label="Observada (ICML/GEIH)")
        bottom.plot(d["date"], 100 * d["informality_rate_clean_median"], color=ORANGE, label="Sin ruido muestral (suavizada)")
        bottom.set_ylabel("% de los ocupados")
        bottom.set_title("B. Tasa de informalidad laboral", loc="left", fontsize=10)
        bottom.legend(loc="upper right")
        fig.savefig(self.out / "fig1_pib_informal.png")
        plt.close(fig)

    def prior_posterior_theta(self) -> None:
        draws = pd.read_csv(self.res / "posterior_draws_thin5.csv")["theta"]
        prior = next(p for p in PRIORS if p.name == "theta")
        row = self.summary.loc["theta"]
        dist = __import__("m2.priors", fromlist=["Prior"]).Prior("theta", "beta", row.prior_mean, row.prior_sd).frozen()
        grid = np.linspace(0.001, 0.999, 500)
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.hist(draws, bins=60, density=True, color=BLUE, alpha=0.45, label="Posterior")
        ax.plot(grid, dist.pdf(grid), color=ORANGE, label=f"Prior Beta({row.prior_mean:.2f}, {row.prior_sd:.2f})")
        ax.set_xlabel("θ: participación informal en el PIB de estado estacionario")
        ax.set_ylabel("Densidad")
        ax.set_title("Prior y posterior de θ: los datos agregados casi no lo mueven", loc="left", fontsize=10)
        ax.legend()
        del prior
        fig.savefig(self.out / "fig2_prior_posterior_theta.png")
        plt.close(fig)

    def profile(self) -> None:
        path = self.res / "profile_theta.csv"
        if not path.exists():
            return
        prof = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.plot(prof["theta"], prof["loglik"] - prof["loglik"].max(), color=BLUE, marker="o", ms=6)
        ax.axhline(-1.92, color=MUTED, lw=1, ls="--")
        ax.text(prof["theta"].max(), -1.80, "umbral del 95 % (−1,92)", ha="right", color=MUTED, fontsize=9)
        ax.set_xlabel("θ fijado")
        ax.set_ylabel("Log-verosimilitud perfilada (relativa al máximo)")
        ax.set_title("Verosimilitud perfilada de θ", loc="left", fontsize=10)
        fig.savefig(self.out / "fig3_perfil_theta.png")
        plt.close(fig)

    def anchor_map(self) -> None:
        d = self.share
        grid = np.linspace(0.15, 0.85, 200)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for color, (label, idx) in zip((BLUE, ORANGE), (("2005T2", 0), (d["quarter"].iloc[-1].replace("Q", "T"), -1))):
            s = d["informality_rate_clean_median"].iloc[idx]
            ax.plot(grid, 100 * informal_share(grid, s), color=color, label=f"{label} (informalidad {100 * s:.0f} %)")
        ax.set_xlabel("Productividad relativa del trabajo informal, ρ̃ (≈ ingreso laboral informal / formal)")
        ax.set_ylabel("% del PIB informal")
        ax.set_title("Tamaño del sector informal según el ancla de productividad relativa", loc="left", fontsize=10)
        ax.legend(loc="upper left")
        fig.savefig(self.out / "fig4_mapa_ancla.png")
        plt.close(fig)

    def all(self) -> None:
        self.informal_share()
        self.prior_posterior_theta()
        self.profile()
        self.anchor_map()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultados", type=Path, default=Path("resultados_m2/base"))
    M2Figures(parser.parse_args().resultados).all()
