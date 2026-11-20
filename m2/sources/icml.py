"""Employment and labor informality from the ICML datasets published on Zenodo.

The ICML (Vera-Jaramillo, 2025; Zenodo concept DOI 10.5281/zenodo.16899685)
reconstructs the Colombian labor market monthly from 1993. Two national files
of the GEIH universe are used:

* ``labor_national_geih_*.csv``: ``employed`` (persons);
* ``informality_national_geih_*.csv``: ``informality_rate``.

Quarterly columns of the estimation dataset (quarter means of months):

    ocupados                         mean employment
    informalidad_media_mensual       mean informality rate
    ocupados_informales              mean of employment x rate
    ocupados_formales                ocupados - ocupados_informales
    informalidad_ponderada_ocupados  ocupados_informales / ocupados
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from m2.sources.http import HttpClient, SourceError

ZENODO = "https://zenodo.org/api/records"
FIRST_RECORD = 16899686          # any version of the concept record; 'latest' is resolved
LABOR_FILE = "universe_geih/labor/labor_national_geih"
INFORMALITY_FILE = "universe_geih/informality/informality_national_geih"
LABOR_COLUMNS = ("ocupados", "informalidad_media_mensual", "ocupados_informales",
                 "ocupados_formales", "informalidad_ponderada_ocupados")


class IcmlZenodo:
    """Downloads the latest ICML dataset version and returns its national monthly series."""

    def __init__(self, http: HttpClient, cache_dir: Path, record: int = FIRST_RECORD) -> None:
        self.http = http
        self.cache_dir = cache_dir
        self.record = record

    def latest(self) -> dict:
        meta = self.http.json(f"{ZENODO}/{self.record}/versions/latest")
        if "files" not in meta or not meta["files"]:
            raise SourceError("Zenodo record has no files")
        return meta

    def download(self, local: Path | None = None) -> tuple[pd.DataFrame, dict]:
        """Monthly frame (index 'YYYY-MM': employed, informality_rate) and provenance."""
        if local is not None:
            path, info = local, {"file": str(local)}
        else:
            meta = self.latest()
            zips = [f for f in meta["files"] if f["key"].endswith(".zip")]
            if len(zips) != 1:
                raise SourceError(f"Expected one ZIP in Zenodo record {meta.get('id')}")
            entry = zips[0]
            path = self.http.download(entry["links"]["self"], self.cache_dir / entry["key"],
                                      min_bytes=int(entry.get("size", 1) * 0.99))
            info = {"record": meta.get("id"), "version": meta["metadata"].get("version"),
                    "doi": meta.get("doi"), "file": entry["key"]}
        monthly = self.parse(path)
        info["last_month"] = monthly.index[-1]
        return monthly, info

    @staticmethod
    def _read(archive: zipfile.ZipFile, prefix: str) -> pd.DataFrame:
        names = [n for n in archive.namelist() if n.startswith(prefix) and n.endswith(".csv")]
        if len(names) != 1:
            raise SourceError(f"ICML ZIP: expected one file '{prefix}*', found {names}")
        return pd.read_csv(io.BytesIO(archive.read(names[0])), dtype={"dep_code": str})

    @classmethod
    def parse(cls, path: Path) -> pd.DataFrame:
        with zipfile.ZipFile(path) as archive:
            labor = cls._read(archive, LABOR_FILE)
            informality = cls._read(archive, INFORMALITY_FILE)
        for frame, column in ((labor, "employed"), (informality, "informality_rate")):
            if "period" not in frame or column not in frame:
                raise SourceError(f"ICML: columns 'period' and '{column}' are required")
        out = (labor.set_index("period")[["employed"]]
               .join(informality.set_index("period")[["informality_rate"]], how="inner"))
        out = out.sort_index()
        if out.isna().any().any() or not out.informality_rate.between(0, 1).all():
            raise SourceError("ICML: missing or out-of-range values")
        expected = pd.period_range(out.index[0], out.index[-1], freq="M").astype(str)
        if list(out.index) != list(expected):
            raise SourceError("ICML: the monthly series has gaps")
        return out


def to_quarters(monthly: pd.DataFrame) -> pd.DataFrame:
    """Quarterly labor columns from complete quarters of a monthly (employed, rate) frame."""
    frame = monthly.copy()
    frame["quarter"] = pd.PeriodIndex(frame.index, freq="M").asfreq("Q").astype(str)
    frame["informal"] = frame.employed * frame.informality_rate
    grouped = frame.groupby("quarter")
    out = pd.DataFrame({
        "ocupados": grouped.employed.mean(),
        "informalidad_media_mensual": grouped.informality_rate.mean(),
        "ocupados_informales": grouped.informal.mean(),
    })
    out = out.loc[grouped.size().eq(3)]
    out["ocupados_formales"] = out.ocupados - out.ocupados_informales
    out["informalidad_ponderada_ocupados"] = out.ocupados_informales / out.ocupados
    return out[list(LABOR_COLUMNS)]
