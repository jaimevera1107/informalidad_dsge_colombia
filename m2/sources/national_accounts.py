"""DANE quarterly GDP by expenditure, chained volumes (reference year 2015).

The DANE page "PIB - información técnica" links the annex
``anex-GastoConstantes-<quarter>trim<year>.xlsx``. Table ("Cuadro") 1 holds the
original (not seasonally adjusted) series in billions of 2015 pesos, 2005 onwards.
The model uses four of its rows:

==============================  ======  ===========================================
DANE code                       column  concept
==============================  ======  ===========================================
B.1b                            pib     gross domestic product
P.311 + P.313 + P.323           c       household (and NPISH) final consumption
P.51                            i       gross fixed capital formation
P.312 + P.322                   g       general government final consumption
==============================  ======  ===========================================
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from m2.sources.http import HttpClient, SourceError

PAGE = ("https://www.dane.gov.co/index.php/estadisticas-por-tema/cuentas-nacionales/"
        "cuentas-nacionales-trimestrales/pib-informacion-tecnica")
BASE = "https://www.dane.gov.co"
ANNEX_PATTERN = r'href="([^"]*anex-GastoConstantes-(I{1,3}V?|IV)trim(\d{4})\.xlsx)"'
ROWS = {"B.1b": "pib", "P.311 + P.313 + P.323": "c", "P.51": "i", "P.312 + P.322": "g"}
QUARTERS = ("I", "II", "III", "IV")
SHEET = "Cuadro 1"


class NationalAccounts:
    """Downloads and parses the latest DANE expenditure annex at constant prices."""

    def __init__(self, http: HttpClient, cache_dir: Path) -> None:
        self.http = http
        self.cache_dir = cache_dir

    def latest_url(self) -> tuple[str, str]:
        """URL and label (e.g. '2026Q3') of the most recent annex linked on DANE's page."""
        found = re.findall(ANNEX_PATTERN, self.http.text(PAGE))
        if not found:
            raise SourceError("DANE GDP page has no 'anex-GastoConstantes' link; the page may have changed")
        candidates = {f"{year}Q{QUARTERS.index(q) + 1}": href for href, q, year in found}
        label = max(candidates)
        href = candidates[label]
        return (href if href.startswith("http") else BASE + href), label

    def download(self, local: Path | None = None) -> tuple[pd.DataFrame, dict]:
        """Quarterly series (index 'YYYYQn', columns pib, c, i, g) and provenance."""
        if local is not None:
            path, url, label = local, str(local), "local"
        else:
            url, label = self.latest_url()
            path = self.http.download(url, self.cache_dir / Path(url).name, min_bytes=20_000)
        frame = self.parse(path)
        if label != "local" and frame.index[-1] != label:
            raise SourceError(f"Annex {path.name} ends in {frame.index[-1]}, expected {label}")
        return frame, {"url": url, "file": path.name, "last_quarter": frame.index[-1]}

    @staticmethod
    def parse(path: Path) -> pd.DataFrame:
        from openpyxl import load_workbook

        book = load_workbook(path, read_only=True, data_only=True)
        if SHEET not in book.sheetnames:
            raise SourceError(f"{path.name}: sheet '{SHEET}' not found")
        rows = list(book[SHEET].iter_rows(values_only=True))
        head = " ".join(str(c) for r in rows[:8] for c in r if c is not None).lower()
        if "encadenadas de volumen" not in head or "datos originales" not in head:
            raise SourceError(f"{path.name}: '{SHEET}' is not chained volumes, original data")
        start = next((i for i, r in enumerate(rows) if r and r[0] == "Clasificación Cuentas Nacionales"), None)
        if start is None:
            raise SourceError(f"{path.name}: header row not found")
        columns: list[tuple[int, str]] = []
        year = None
        for j, (y, q) in enumerate(zip(rows[start], rows[start + 1])):
            if j >= 2 and y is not None:
                year = int(str(y)[:4])
            if q in QUARTERS and year is not None:
                columns.append((j, f"{year}Q{QUARTERS.index(q) + 1}"))
        series: dict[str, pd.Series] = {}
        for r in rows[start + 2:]:
            if r[0] is not None and str(r[0]).startswith("Fuente"):
                break                       # only the first block holds levels
            code = str(r[0]).strip() if r[0] is not None else ""
            if code in ROWS:
                series[ROWS[code]] = pd.Series({q: r[j] for j, q in columns}, dtype="float64")
        missing = set(ROWS.values()) - set(series)
        if missing:
            raise SourceError(f"{path.name}: rows not found: {sorted(missing)}")
        frame = pd.DataFrame(series)[list(ROWS.values())].dropna(how="all")
        if frame.isna().any().any() or (frame <= 0).any().any():
            raise SourceError(f"{path.name}: empty or non-positive values")
        if frame.index[0] != "2005Q1":
            raise SourceError(f"{path.name}: series starts in {frame.index[0]}, expected 2005Q1")
        frame.index.name = "quarter"
        return frame
