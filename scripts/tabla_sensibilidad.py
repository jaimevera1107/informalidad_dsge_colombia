"""Build the sensitivity table of the paper from the estimation result folders."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Specification:
    """One estimated specification shown as a row of the table."""

    label: str
    folder: str


PANEL = "panel"
SPECIFICATIONS: tuple[Specification, ...] = (
    Specification(r"\emph{A. Formal wage concept (labor-only, $\alpha=0.354$, $\delta=4\%$)}", PANEL),
    Specification(r"Baseline: statutory non-wage costs, contributions and benefits", "solo_trabajo_cuna_legal"),
    Specification(r"Statutory contributions only", "solo_trabajo_cuna_aportes"),
    Specification(r"Constant wedge, $\bar\tau=0.25$", "solo_trabajo"),
    Specification(r"\emph{B. Other assumptions (labor-only, constant wedge)}", PANEL),
    Specification(r"$\alpha=0.33$, $\delta=6\%$", "solo_trabajo_d6"),
    Specification(r"$\alpha=0.50$", "solo_trabajo_a050"),
    Specification(r"Prior of $\eta$ centered at 0.5", "solo_trabajo_eta05"),
    Specification(r"Prior of $\eta$ centered at 0.9", "solo_trabajo_eta09"),
    Specification(r"Sample from 2007Q1", "solo_trabajo_desde2007"),
    Specification(r"Capital--output ratio observed in levels$^{\dagger}$", "solo_trabajo_capital"),
    Specification(r"\emph{C. Equal technologies and aggregates only}", PANEL),
    Specification("Equal technologies, constant wedge (upper bound)", "con_ingresos"),
    Specification("Equal technologies, no wedge adjustment", "con_ingresos_sin_cuna"),
    Specification("Equal technologies, hourly earnings", "con_ingresos_por_hora"),
    Specification("Aggregates only (no earnings)", "desde_1993"),
)


class SensitivityTable:
    """Collect key posterior quantities for each specification."""

    def __init__(self, results_dir: Path) -> None:
        if not results_dir.is_dir():
            raise FileNotFoundError(results_dir)
        self.results_dir = results_dir

    @staticmethod
    def _band(row: pd.Series) -> str:
        return (f"{100 * row.informal_output_share_median:.1f} "
                f"[{100 * row.informal_output_share_p05:.1f}, "
                f"{100 * row.informal_output_share_p95:.1f}]")

    def row(self, spec: Specification) -> dict[str, str] | None:
        """Return the formatted row, or None when the run is missing."""
        if spec.folder == PANEL:
            return {"label": spec.label, "panel": True}
        folder = self.results_dir / spec.folder
        summary_path = folder / "run_summary.json"
        if not summary_path.exists():
            return None
        summary = json.loads(summary_path.read_text())
        post = pd.read_csv(folder / "posterior_summary.csv").set_index("parameter")
        share = pd.read_csv(folder / "informal_output_share.csv").set_index("quarter")
        first = share.loc["1993Q2"] if "1993Q2" in share.index else None
        theta = post.loc["theta"]
        return {
            "label": spec.label,
            "rho": f"{summary['rho_tilde_median']:.2f}",
            "theta": f"{theta.post_median:.3f} [{theta.p05:.3f}, {theta.p95:.3f}]",
            "first": self._band(first) if first is not None else "--",
            "last": self._band(share.iloc[-1]),
        }

    def to_latex(self, out_path: Path) -> pd.DataFrame:
        """Write the LaTeX body and return the rows as a frame."""
        rows = [r for r in (self.row(s) for s in SPECIFICATIONS) if r is not None]
        lines = [rf"\multicolumn{{5}}{{l}}{{{r['label']}}} \\" if r.get("panel") else
                 f"{r['label']} & {r['rho']} & {r['theta']} & {r['first']} & {r['last']} \\\\" for r in rows]
        out_path.write_text("\n".join(lines) + "\n")
        return pd.DataFrame([r for r in rows if not r.get("panel")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = SensitivityTable(args.results).to_latex(args.out / "tables" / "sensitivity.tex")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
