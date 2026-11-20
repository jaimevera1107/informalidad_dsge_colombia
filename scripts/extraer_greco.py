"""Extract the annual GRECO tables used to backcast the estimation series.

Sources (Banco de la República, Grupo de Estudios del Crecimiento Económico, GRECO):
    borra121.pdf  "El desempeño macroeconómico colombiano. Series estadísticas (1905-1997)",
                  Borrador de Economía 121, segunda versión, 1999.
                  Table 1.2 (PEA, ocupados and desocupados, Flórez 1998).
    borra134.pdf  "El crecimiento económico colombiano en el siglo XX: aspectos globales",
                  Borrador de Economía 134, 1999.
                  Annex Table 2A (capital, GDP, PEA, population 1905-1997) and Annex Table 3A
                  (gross fixed capital formation by asset, depreciation and capital stocks).

The PDFs are converted with `pdftotext -layout` (poppler). The extracted CSV files are
shipped with the project, so this script only needs to be rerun to audit the extraction.

Usage:
    python scripts/extraer_greco.py --pdf-dir datos/greco --out datos/greco
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

YEAR_ROW = re.compile(r"^\s*(1\.?[89]\d{2})(?:prov|py)?\s+(.*)$")


def spanish_number(token: str) -> float | None:
    """'1.694.072' -> 1694072; '9,0302' -> 9.0302; '#N/A' -> None."""
    token = token.strip()
    if not token or "N/A" in token or token.endswith("%"):
        return None
    try:
        return float(token.replace(".", "").replace(",", "."))
    except ValueError:
        return None


def pdf_text(pdf: Path) -> list[str]:
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext (poppler) is required to rerun the extraction")
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def year_rows(lines: list[str], start: int, end: int) -> dict[int, list[str]]:
    """Rows of a table between two line numbers, keyed by year, raw tokens."""
    rows: dict[int, list[str]] = {}
    for line in lines[start:end]:
        m = YEAR_ROW.match(line)
        if m:
            year = int(m.group(1).replace(".", ""))
            rows.setdefault(year, m.group(2).split())
    return rows


def find(lines: list[str], text: str, after: int = 0) -> int:
    for i in range(after, len(lines)):
        if text in lines[i]:
            return i
    raise ValueError(f"marker not found: {text!r}")


GRECO_URLS = {"borra121.pdf": "https://www.banrep.gov.co/docum/ftp/borra121.pdf",
              "borra134.pdf": "https://www.banrep.gov.co/docum/ftp/borra134.pdf"}


def ensure_pdfs(pdf_dir: Path) -> None:
    """Download the two Banco de la Republica working papers if they are not present."""
    import urllib.request

    pdf_dir.mkdir(parents=True, exist_ok=True)
    for name, url in GRECO_URLS.items():
        target = pdf_dir / name
        if not target.is_file():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, target)


class GrecoExtractor:
    """Parses the three GRECO tables into tidy annual frames."""

    def __init__(self, pdf_dir: Path) -> None:
        self.t121 = pdf_text(pdf_dir / "borra121.pdf")
        self.t134 = pdf_text(pdf_dir / "borra134.pdf")

    def series_2a(self) -> pd.DataFrame:
        """Annex Table 2A of BE 134: capital (incl. inventories), GDP, PEA, population."""
        lines = self.t134
        start = find(lines, "SERIES UTILIZADAS")
        end = find(lines, "Hoja: T2 ESTAD")
        records = []
        for year, tokens in year_rows(lines, start, end).items():
            values = [t for t in tokens if not t.endswith("%")]
            capital, gdp, pea, pop = (spanish_number(v) for v in values[:4])
            records.append({"year": year, "capital_total_1975": capital, "pib_1975": gdp,
                            "pea_florez": pea, "poblacion": pop})
        return pd.DataFrame(records).sort_values("year")

    def investment_3a(self) -> pd.DataFrame:
        """Annex Table 3A of BE 134: gross fixed capital formation (1975 pesos)."""
        lines = self.t134
        start = find(lines, "DATOS BASICOS PARA EL CALCULO DEL CAPITAL")
        end = find(lines, "CALCULO DEL CAPITAL PARA MAQUINARIA Y EQUIPO")
        records = []
        for year, tokens in year_rows(lines, start, end).items():
            nums = [spanish_number(t) for t in tokens]
            nums = [n for n, t in zip(nums, tokens) if n is not None or "N/A" in t]
            if year == 1925:         # first row has no growth rate: FBKF total, inventories, ...
                fbkf, construction, machinery = nums[0], None, None
            elif year < 1950:        # growth CEPAL, FBKF total, inventories, ...
                fbkf, construction, machinery = nums[1], None, None
            elif year == 1950:       # growth, construction, machinery, total, ...
                construction, machinery, fbkf = nums[1], nums[2], nums[3]
            else:                    # construction, machinery, total, ...
                construction, machinery, fbkf = nums[0], nums[1], nums[2]
            records.append({"year": year, "fbkf_1975": fbkf, "fbkf_construccion_1975": construction,
                            "fbkf_maquinaria_1975": machinery})
        return pd.DataFrame(records).sort_values("year")

    def capital_3a(self) -> pd.DataFrame:
        """Depreciation and stocks by asset (1950-1997), from the two asset tables of 3A."""
        lines = self.t134
        out = {}
        for label, marker in (("construccion", "CALCULO DEL CAPITAL PARA CONSTRUCCION Y VIVIENDA"),
                              ("maquinaria", "CALCULO DEL CAPITAL PARA MAQUINARIA Y EQUIPO")):
            start = find(lines, marker)
            end = find(lines, "Hoja:T3 Capital", start + 5)
            for year, tokens in year_rows(lines, start, end).items():
                nums = [spanish_number(t) for t in tokens]
                nums = [n for n in nums if n is not None]
                if len(nums) < 4:
                    continue
                stock = nums[3] if label == "construccion" else nums[-2] if len(nums) >= 6 else nums[-1]
                row = out.setdefault(year, {"year": year})
                row[f"depreciacion_{label}_1975"] = nums[1]
                row[f"capital_{label}_1975"] = stock
        return pd.DataFrame(out.values()).sort_values("year")

    def employment_121(self) -> pd.DataFrame:
        """Table 1.2 of BE 121: PEA (GRECO and Flórez), employed and unemployed (Flórez)."""
        lines = self.t121
        start = find(lines, "Hoja: Cuadro No. 1.1.2")       # first page of Table 1.2 (after the index)
        end = find(lines, "Hoja: 1.2.1. Gráfico 2", start)
        records = {}
        for line in lines[start:end]:
            m = re.match(r"^\s*(19\d{2}|18\d{2})\s+(.*)$", line)
            if not m:
                continue
            nums = [spanish_number(t) for t in m.group(2).split()]
            nums = [n for n in nums if n is not None]
            if not nums:
                continue
            row = {"year": int(m.group(1)), "pea_greco": nums[0]}
            if len(nums) > 1:
                row["pea_florez"] = nums[1]
            if len(nums) > 3:
                row["ocupados_florez"], row["desocupados_florez"] = nums[2], nums[3]
            if row["year"] not in records:      # first occurrence is the table row; later ones are labels
                records[row["year"]] = row
        return pd.DataFrame(records.values()).sort_values("year")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path("datos/greco"))
    parser.add_argument("--out", type=Path, default=Path("datos/greco"))
    args = parser.parse_args()
    ensure_pdfs(args.pdf_dir)
    ex = GrecoExtractor(args.pdf_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    frames = {"greco_tabla2a_series.csv": ex.series_2a(), "greco_tabla3a_inversion.csv": ex.investment_3a(),
              "greco_tabla3a_capital_por_activo.csv": ex.capital_3a(),
              "greco_tabla1_2_empleo.csv": ex.employment_121()}
    for name, frame in frames.items():
        frame.to_csv(args.out / name, index=False)
        print(f"{name}: {len(frame)} rows, {int(frame.year.min())}-{int(frame.year.max())}")


if __name__ == "__main__":
    main()
