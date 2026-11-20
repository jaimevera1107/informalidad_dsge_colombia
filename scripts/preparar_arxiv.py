"""Build the arXiv source package of the paper with its ancillary files.

Copies the LaTeX sources of `paper/` and, under `anc/`, the code (package m2 and
scripts), the input data and the result files, then writes a .tar.gz. The PDF is
compiled first when latexmk is available, so a broken build is caught here.

Usage:
    python scripts/preparar_arxiv.py --salida dist/arxiv_informalidad.tar.gz
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_FILES = ("main.tex", "references.bib", "main.bbl")
PAPER_DIRS = ("sections", "tables", "figures")
ANC_DATA = ("datos_dsge_estimacion.csv", "geih", "referencias", "costos_laborales")
DRAWS = "posterior_draws_thin5.csv"
KEEP_DRAWS = ("con_ingresos", "desde_1993", "solo_trabajo_cuna_legal")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*_mensual*.csv")


class ArxivPackage:
    """Assembles the source tree and the ancillary files in a temporary folder."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paper = root / "paper"

    def compile(self) -> None:
        if shutil.which("latexmk") is None:
            print("latexmk not found: PDF not rebuilt (main.bbl must already exist)")
            return
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "main.tex"],
                       cwd=self.paper, check=True, capture_output=True)

    def assemble(self, target: Path) -> None:
        for name in PAPER_FILES:
            src = self.paper / name
            if not src.exists():
                raise FileNotFoundError(f"{src} is missing: compile the paper first")
            shutil.copy2(src, target / name)
        for name in PAPER_DIRS:
            shutil.copytree(self.paper / name, target / name)
        anc = target / "anc"
        shutil.copytree(self.root / "m2", anc / "code" / "m2", ignore=IGNORE)
        shutil.copytree(self.root / "scripts", anc / "code" / "scripts", ignore=IGNORE)
        shutil.copytree(self.root / "tests", anc / "code" / "tests", ignore=IGNORE)
        for name in ANC_DATA:
            src = self.root / "datos" / name
            dst = anc / "data" / name
            if src.is_dir():
                shutil.copytree(src, dst, ignore=IGNORE)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        results = self.root / "resultados_m2"
        for item in results.iterdir():
            dst = anc / "results" / item.name
            if item.is_dir():
                shutil.copytree(item, dst, ignore=IGNORE)
                if item.name not in KEEP_DRAWS and (dst / DRAWS).exists():
                    (dst / DRAWS).unlink()
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)
        final = self.root / "resultados_finales"
        if final.exists():
            shutil.copytree(final, anc / "final_results", ignore=IGNORE)
        shutil.copy2(self.root / "docs" / "ANC_README.md", anc / "README.md")

    def build(self, output: Path) -> Path:
        self.compile()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            self.assemble(target)
            output.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(output, "w:gz") as tar:
                for path in sorted(target.rglob("*")):
                    tar.add(path, arcname=path.relative_to(target), recursive=False)
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=ROOT / "dist" / "arxiv_informalidad.tar.gz")
    args = parser.parse_args()
    out = ArxivPackage(ROOT).build(args.salida)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
