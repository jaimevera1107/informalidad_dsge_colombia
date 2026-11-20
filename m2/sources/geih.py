"""Monthly GEIH microdata downloaded from DANE's microdata catalog (ANDA).

DANE publishes one catalog study per year ("Gran Encuesta Integrada de Hogares -
GEIH - YYYY") with one ZIP per month containing the survey modules in CSV, DTA and
SAV. This module locates a month, downloads it once, reads the two modules needed
(occupied persons and general characteristics) and returns, per month:

* the weighted earnings sums of :func:`m2.geih_earnings.month_sums`;
* national employment and the informality rate, used to check the downloaded
  months against the ICML series.
"""
from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from m2.geih_earnings import MonthResult, month_sums, to_number
from m2.sources.http import HttpClient, SourceError
from m2.sources.informality import InformalityClassifier

CATALOG = "https://microdatos.dane.gov.co/index.php/catalog"
MONTHS_ES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre")
PERSON_KEY = ("DIRECTORIO", "SECUENCIA_P", "ORDEN")
OCCUPIED_COLUMNS = ("FEX_C18", "P6430", "P6450", "P6920", "P6930", "P6940", "P6765", "P3065", "P3066",
                    "P3067", "P3067S1", "P3067S2", "P3068", "P3069", "P6775", "OFICIO_C8", "RAMA2D_R4",
                    "INGLABO", "P6500", "P6750", "P6760", "P6800", "P7070")
CHARACTERISTICS_COLUMNS = ("P6100", "P6110", "P6090")
MIN_ZIP_BYTES = 1_000_000


def _plain(text: str) -> str:
    """Lower-case ASCII text, decoding the '#U00ed' escapes found in some DANE ZIPs."""
    text = re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().lower()


class GeihCatalog:
    """Finds the download URL of each monthly GEIH file in DANE's catalog."""

    def __init__(self, http: HttpClient) -> None:
        self.http = http
        self._studies: dict[int, int] = {}
        self._months: dict[int, dict[str, str]] = {}

    def study_id(self, year: int) -> int:
        if year not in self._studies:
            url = f"{CATALOG}/central/search?sk=GEIH+{year}&ps=100"
            html = self.http.text(url)
            title = f"gran encuesta integrada de hogares - geih - {year}"
            found = [int(i) for i, t in re.findall(r'catalog/(\d+)"[^>]*title="([^"]+)"', html)
                     if _plain(t) == title]
            if not found:
                raise SourceError(f"GEIH {year} is not in DANE's microdata catalog yet")
            self._studies[year] = max(found)
        return self._studies[year]

    def month_urls(self, year: int) -> dict[str, str]:
        """Published months of ``year`` as {'YYYY-MM': download URL}."""
        if year not in self._months:
            html = self.http.text(f"{CATALOG}/{self.study_id(year)}/get-microdata")
            urls: dict[str, str] = {}
            pattern = r"title=\"([^\"]+?)\.zip\"\s+onclick=\"mostrarModal\('[^']*'\s*,\s*'([^']+)'"
            for label, url in re.findall(pattern, html):
                words = _plain(label).split()
                if len(words) == 2 and words[0] in MONTHS_ES and words[1] == str(year):
                    urls[f"{year}-{MONTHS_ES.index(words[0]) + 1:02d}"] = url.strip()
            self._months[year] = urls
        return self._months[year]

    def url(self, period: str) -> str | None:
        try:
            return self.month_urls(int(period[:4])).get(period)
        except SourceError:
            return None


@dataclass(frozen=True)
class MonthSummary:
    """Outputs of one downloaded month."""

    period: str
    earnings: MonthResult
    employed: float
    informality_rate: float
    unclassified_share: float


class GeihMonth:
    """Reads one monthly ZIP and computes earnings sums, employment and informality."""

    def __init__(self, path: Path, period: str, classifier: InformalityClassifier | None = None) -> None:
        self.path = path
        self.period = period
        self.classifier = classifier or InformalityClassifier()

    def _module(self, archive: zipfile.ZipFile, keyword: str, columns: tuple[str, ...]) -> pd.DataFrame:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if keyword == "ocupados":
            chosen = [n for n in names if _plain(Path(n).stem) == "ocupados"]
        else:
            chosen = [n for n in names if keyword in _plain(Path(n).stem)]
        if len(chosen) != 1:
            raise SourceError(f"{self.period}: module '{keyword}' not found once in {self.path.name}: {chosen}")
        raw = archive.read(chosen[0])
        text = raw.decode("utf-8-sig", errors="replace") if raw[:3] == b"\xef\xbb\xbf" else raw.decode("latin-1")
        header = text.split("\n", 1)[0]
        sep = ";" if header.count(";") > header.count(",") else ","
        wanted = set(PERSON_KEY) | set(columns)
        frame = pd.read_csv(io.StringIO(text), sep=sep, dtype=str,
                            usecols=lambda c: c.strip().upper() in wanted)
        frame.columns = [c.strip().upper() for c in frame.columns]
        missing = set(PERSON_KEY) - set(frame.columns)
        if missing:
            raise SourceError(f"{self.period}: key columns missing in '{keyword}': {sorted(missing)}")
        key = frame[list(PERSON_KEY)].apply(lambda s: s.str.strip()).agg("|".join, axis=1)
        frame = frame.drop(columns=list(PERSON_KEY)).add_prefix("SRC_")
        frame.index = pd.Index(key, name="persona_id")
        if frame.index.duplicated().any():
            raise SourceError(f"{self.period}: duplicated persons in '{keyword}'")
        return frame

    def summarize(self) -> MonthSummary:
        with zipfile.ZipFile(self.path) as archive:
            occ = self._module(archive, "ocupados", OCCUPIED_COLUMNS)
            chars = self._module(archive, "generales", CHARACTERISTICS_COLUMNS)
        status = self.classifier.classify(occ, chars, self.period)
        occ = occ.join(chars[[c for c in chars.columns if c not in occ.columns]], how="left")
        weight = to_number(occ["SRC_FEX_C18"])
        valid = weight.gt(0)
        employed = float(weight[valid].sum())
        classified = valid & status.notna()
        rate = float((weight[classified] * status[classified]).sum() / weight[classified].sum())
        unclassified = float(weight[valid & status.isna()].sum() / employed)
        return MonthSummary(self.period, month_sums(self.period, occ, status), employed, rate, unclassified)


class GeihDownloader:
    """Downloads and summarizes GEIH months, caching both the ZIPs and the results."""

    def __init__(self, http: HttpClient, cache_dir: Path, summaries_file: Path,
                 keep_zips: bool = False) -> None:
        self.http = http
        self.catalog = GeihCatalog(http)
        self.cache_dir = cache_dir
        self.summaries_file = summaries_file
        self.keep_zips = keep_zips

    def _load(self) -> pd.DataFrame:
        if self.summaries_file.is_file():
            return pd.read_csv(self.summaries_file, dtype={"periodo": str})
        return pd.DataFrame(columns=["periodo", "estado"])

    def months(self, periods: list[str], log=print) -> pd.DataFrame:
        """Summaries of ``periods`` (months not yet published by DANE are skipped)."""
        done = self._load()
        reuse = done.loc[done.estado.eq("CALCULADO")] if len(done) else done
        rows = reuse.to_dict("records")
        finished = set(reuse.periodo) if len(reuse) else set()
        for period in periods:
            if period in finished:
                continue
            url = self.catalog.url(period)
            if url is None:
                log(f"GEIH {period}: not yet published by DANE")
                continue
            target = self.cache_dir / f"GEIH_{period}.zip"
            log(f"GEIH {period}: downloading {url}")
            self.http.download(url, target, min_bytes=MIN_ZIP_BYTES)
            summary = GeihMonth(target, period).summarize()
            row = summary.earnings.row()
            row.update({"ocupados_microdato": summary.employed,
                        "tasa_informalidad_microdato": summary.informality_rate,
                        "sin_clasificar_ocupados": summary.unclassified_share})
            rows.append(row)
            self.summaries_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values("periodo").to_csv(self.summaries_file, index=False)
            log(f"GEIH {period}: {row['estado']}, informality {summary.informality_rate:.3f}")
            if not self.keep_zips:
                target.unlink(missing_ok=True)
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.loc[out.periodo.isin(periods)].sort_values("periodo").reset_index(drop=True)


def month_range(first: str, last: str) -> list[str]:
    """Months from ``first`` to ``last`` inclusive, as 'YYYY-MM'."""
    if first > last:
        return []
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def quarter_months(quarter: str) -> list[str]:
    q = pd.Period(quarter, freq="Q")
    start = q.asfreq("M", how="start")
    return [str(start + k) for k in range(3)]


__all__ = ["GeihCatalog", "GeihDownloader", "GeihMonth", "MonthSummary", "month_range",
           "quarter_months"]
