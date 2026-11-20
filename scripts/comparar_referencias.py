"""Compare the estimated informal share of output with published estimates for Colombia.

Inputs
    --referencias  CSV of published estimates (datos/referencias/referencias_colombia.csv)
    --banco-mundial optional: the World Bank Informal Economy Database Excel file; when
                   given, its full DGE and MIMIC series for Colombia replace the few
                   values typed in the CSV
    --base         result folder of the baseline estimate (labor-only model)
    --cota         optional result folder of the equal-technology estimate (upper bound)
Outputs (in --salida)
    comparacion_referencias.csv   annual own estimate next to every published value
    fig_benchmarks.pdf/.png       figure for the paper

Usage:
    python scripts/comparar_referencias.py --base resultados_m2/solo_trabajo ^
        --cota resultados_m2/con_ingresos --salida resultados_m2/comparacion
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # source -> (color, linestyle, marker); identity is also in the legend and marker
    "Medina and Schneider (2019)": (ORANGE, "-", None),
    "Asllani, Dell'Anno and Schneider (2024)": (AQUA, "-", None),
    "Elgin et al. (2021), DGE": (YELLOW, "-", "o"),
    "Elgin et al. (2021), MIMIC": (MUTED, ":", "s"),
    "Schneider, Buehn and Montenegro (2010)": (MUTED, "", "^"),
    "Vuletin (2008)": (INK, "", "D"),
}


@dataclass(frozen=True)
class Series:
    label: str
    annual: pd.DataFrame          # index year; columns median, p05, p95


class OwnEstimate:
    """Annual means of the quarterly informal output share of one result folder."""

    @staticmethod
    def load(folder: Path, label: str) -> Series:
        path = folder / "informal_output_share.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        q = pd.read_csv(path)
        q["year"] = q["quarter"].astype(str).str[:4].astype(int)
        cols = {"informal_output_share_median": "median", "informal_output_share_p05": "p05",
                "informal_output_share_p95": "p95"}
        annual = q.groupby("year")[list(cols)].mean().rename(columns=cols) * 100.0
        return Series(label, annual)


class WorldBankDatabase:
    """Colombia rows of the World Bank Informal Economy Database (DGE and MIMIC sheets)."""

    SHEETS = {"DGE": "Elgin et al. (2021), DGE", "MIMIC": "Elgin et al. (2021), MIMIC"}

    def __init__(self, path: Path) -> None:
        self.book = pd.read_excel(path, sheet_name=None, header=None)

    def colombia(self) -> pd.DataFrame:
        rows = []
        for sheet_name, frame in self.book.items():
            source = next((v for k, v in self.SHEETS.items() if k.lower() in sheet_name.lower()), None)
            if source is None:
                continue
            header_row = self._header_row(frame)
            if header_row is None:
                continue
            header = frame.iloc[header_row]
            years = {j: int(float(h)) for j, h in header.items() if self._is_year(h)}
            for _, r in frame.iloc[header_row + 1:].iterrows():
                if not any(str(x).strip() in ("COL", "Colombia") for x in r.values[:4]):
                    continue
                for j, y in years.items():
                    v = pd.to_numeric(r[j], errors="coerce")
                    if np.isfinite(v):
                        rows.append({"source": source, "key": "elgin2021understanding",
                                     "method": sheet_name, "concept": "informal output, % of official GDP",
                                     "year_start": y, "year_end": y, "value": float(v),
                                     "location": "World Bank Informal Economy Database"})
        if not rows:
            raise ValueError("no Colombia rows found in DGE/MIMIC sheets of the World Bank file")
        return pd.DataFrame(rows)

    @staticmethod
    def _is_year(value: object) -> bool:
        try:
            return 1950 <= float(value) <= 2100 and float(value).is_integer()
        except (TypeError, ValueError):
            return False

    def _header_row(self, frame: pd.DataFrame) -> int | None:
        for i in range(min(15, len(frame))):
            if sum(self._is_year(v) for v in frame.iloc[i].values) >= 10:
                return i
        return None


class BenchmarkComparison:
    """Aligns published values with the own estimate and draws the figure."""

    def __init__(self, references: pd.DataFrame, base: Series, upper: Series | None) -> None:
        self.refs = references
        self.base = base
        self.upper = upper

    def table(self) -> pd.DataFrame:
        out = []
        for r in self.refs.itertuples(index=False):
            years = range(int(r.year_start), int(r.year_end) + 1)
            own = self.base.annual["median"].reindex(years).mean()
            row = {"source": r.source, "method": r.method, "period": f"{r.year_start}-{r.year_end}"
                   if r.year_start != r.year_end else str(r.year_start), "published": r.value,
                   self.base.label: own, "difference": own - r.value}
            if self.upper is not None:
                row[self.upper.label] = self.upper.annual["median"].reindex(years).mean()
            out.append(row)
        return pd.DataFrame(out)

    def figure(self, path: Path) -> None:
        plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                             "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                             "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False})
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        b = self.base.annual
        ax.fill_between(b.index, b.p05, b.p95, color=BLUE, alpha=0.20, lw=0)
        ax.plot(b.index, b["median"], color=BLUE, label=self.base.label)
        if self.upper is not None:
            u = self.upper.annual
            ax.plot(u.index, u["median"], color=BLUE, ls="--", lw=1.5, label=self.upper.label)
        for source, g in self.refs.groupby("source", sort=False):
            color, ls, marker = STYLE.get(source, (MUTED, ":", "x"))
            single = g.year_start.eq(g.year_end)
            if single.all() and len(g) > 1:
                # reindex to every year so that gaps break the line instead of bridging them
                s = g.set_index("year_start")["value"].sort_index()
                s = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
                ax.plot(s.index, s.values, color=color, ls=ls or "none", marker=marker, ms=5,
                        lw=1.5, label=source)
                continue
            for k, r in enumerate(g.itertuples(index=False)):
                lab = source if k == 0 else None
                if r.year_start == r.year_end:
                    ax.plot([r.year_start], [r.value], color=color, marker=marker or "o", ms=6, ls="none", label=lab)
                else:
                    ax.plot([r.year_start - 0.4, r.year_end + 0.4], [r.value, r.value], color=color, lw=2.5,
                            marker=None, label=lab)
        ax.set_ylabel("% of GDP")
        ax.set_xlim(1990, 2027)
        ax.set_ylim(0, 55)
        ax.legend(loc="lower left", fontsize=7.5, ncol=2)
        fig.tight_layout()
        fig.savefig(path.with_suffix(".pdf"))
        fig.savefig(path.with_suffix(".png"), dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--referencias", type=Path, default=Path("datos/referencias/referencias_colombia.csv"))
    parser.add_argument("--banco-mundial", type=Path, default=None)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--etiqueta-base", default="This paper: labor-only informal sector (baseline)")
    parser.add_argument("--cota", type=Path, default=None)
    parser.add_argument("--etiqueta-cota", default="This paper: equal technologies (upper bound)")
    parser.add_argument("--salida", type=Path, default=Path("resultados_m2/comparacion"))
    args = parser.parse_args()

    refs = pd.read_csv(args.referencias)
    if args.banco_mundial is not None:
        wb = WorldBankDatabase(args.banco_mundial).colombia()
        refs = pd.concat([refs.loc[~refs.key.eq("elgin2021understanding")], wb], ignore_index=True)
    base = OwnEstimate.load(args.base, args.etiqueta_base)
    upper = OwnEstimate.load(args.cota, args.etiqueta_cota) if args.cota else None
    comp = BenchmarkComparison(refs, base, upper)
    args.salida.mkdir(parents=True, exist_ok=True)
    table = comp.table()
    table.to_csv(args.salida / "comparacion_referencias.csv", index=False)
    comp.figure(args.salida / "fig_benchmarks")
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
