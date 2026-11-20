"""Quarterly update of the estimation dataset from public sources.

Sources (all downloaded):

1. DANE national accounts (expenditure, chained volumes 2015): pib, c, i, g, 2005+.
2. ICML datasets on Zenodo: employment and informality, 1993+.
3. DANE GEIH microdata: relative earnings for quarters after the frozen series, and
   employment and informality for months after the last ICML month.

What is never recomputed
------------------------
* National accounts before 2005 (temporal disaggregation of DANE's annual
  retropolation). If DANE revises the level of 2005, the historical block is
  rescaled by the 2005 annual ratio, which preserves its growth rates.
* Earnings quarters already in ``datos/geih/ratio_ingresos_geih_v3.csv``; new
  quarters are appended.

Splicing of the informality rate after the last ICML month: the microdata rate plus
the mean ICML-minus-microdata difference over the last ``overlap_months`` ICML
months. Employment from the microdata equals the ICML national figure exactly.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from m2.earnings import ValueAddedRatio
from m2.geih_earnings import quarterly as earnings_quarterly
from m2.sources.geih import GeihDownloader, month_range, quarter_months
from m2.sources.http import HttpClient, SourceError
from m2.sources.icml import IcmlZenodo, to_quarters
from m2.sources.national_accounts import NationalAccounts

ACCOUNT_COLUMNS = ("pib", "c", "i", "g")
FIRST_OFFICIAL_QUARTER = "2005Q1"
MAX_LEVEL_REVISION = 0.05          # larger 2005 revisions suggest a new base year
MAX_ANNUAL_LOG_CHANGE = 0.35       # year-on-year change above this (outside 2020-21) stops the update
MAX_SPLICE_GAP = 0.02              # mean ICML vs microdata gap in the informality rate

Logger = Callable[[str], None]


@dataclass
class UpdatePaths:
    """Project files read and written by the update."""

    root: Path

    @property
    def dataset(self) -> Path:
        return self.root / "datos" / "datos_dsge_estimacion.csv"

    @property
    def earnings(self) -> Path:
        return self.root / "datos" / "geih" / "ratio_ingresos_geih_v3.csv"

    @property
    def geih_monthly(self) -> Path:
        return self.root / "datos" / "geih" / "geih_descargado_mensual.csv"

    @property
    def costs(self) -> Path:
        return self.root / "datos" / "costos_laborales" / "costo_no_salarial.csv"

    def value_added_ratio(self, alpha: float) -> Path:
        return self.root / "datos" / "geih" / f"ratio_va_{str(alpha).replace('.', '')}.csv"

    @property
    def downloads(self) -> Path:
        return self.root / "datos" / "descargas"

    @property
    def versions(self) -> Path:
        return self.root / "datos" / "versiones"

    @property
    def manifest(self) -> Path:
        return self.root / "datos" / "actualizacion.json"


@dataclass
class UpdateOptions:
    alpha: float = 0.354
    overlap_months: int = 3
    validate_last_quarter: bool = False
    keep_zips: bool = False
    accounts_file: Path | None = None
    icml_file: Path | None = None
    dry_run: bool = False


@dataclass
class UpdateReport:
    started: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sources: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    last_quarter: str | None = None
    new_quarters: list[str] = field(default_factory=list)
    changed: bool = False


class DatasetUpdater:
    """Downloads the sources, splices them with the frozen history and writes the inputs."""

    def __init__(self, paths: UpdatePaths, options: UpdateOptions, log: Logger = print,
                 http: HttpClient | None = None) -> None:
        self.paths = paths
        self.options = options
        self.log = log
        self.http = http or HttpClient()
        self.report = UpdateReport()
        self.geih = GeihDownloader(self.http, paths.downloads / "geih", paths.geih_monthly,
                                   keep_zips=options.keep_zips)

    # ------------------------------------------------------------------ accounts
    def accounts(self, previous: pd.DataFrame) -> pd.DataFrame:
        official, info = NationalAccounts(self.http, self.paths.downloads).download(self.options.accounts_file)
        self.report.sources["cuentas_nacionales"] = info
        self.log(f"DANE accounts: {info['file']} ({FIRST_OFFICIAL_QUARTER}-{info['last_quarter']})")
        history = previous.loc[previous.index < FIRST_OFFICIAL_QUARTER, list(ACCOUNT_COLUMNS)].copy()
        old_2005 = previous.loc[previous.index.str.startswith("2005"), list(ACCOUNT_COLUMNS)].sum()
        new_2005 = official.loc[official.index.str.startswith("2005")].sum()
        factor = new_2005 / old_2005
        if (factor - 1).abs().max() > MAX_LEVEL_REVISION:
            raise SourceError(f"2005 levels changed by more than {MAX_LEVEL_REVISION:.0%} "
                              f"({factor.round(4).to_dict()}); DANE may have changed the base year. "
                              "Rebuild the historical block before updating.")
        if (factor - 1).abs().max() > 1e-9:
            self.report.notes.append(f"Pre-2005 block rescaled by the 2005 revision: {factor.round(6).to_dict()}")
        return pd.concat([history * factor, official])

    # ------------------------------------------------------------------ labor
    def labor(self, last_month: str) -> pd.DataFrame:
        monthly, info = IcmlZenodo(self.http, self.paths.downloads).download(self.options.icml_file)
        self.report.sources["icml"] = info
        self.log(f"ICML: version {info.get('version')} up to {info['last_month']}")
        icml_last = monthly.index[-1]
        extra = month_range(str(pd.Period(icml_last, "M") + 1), last_month)
        if extra:
            overlap = month_range(str(pd.Period(icml_last, "M") - self.options.overlap_months + 1), icml_last)
            micro = self.geih.months(overlap + extra, log=self.log)
            micro = micro.loc[micro.estado.eq("CALCULADO")].set_index("periodo")
            available = [m for m in extra if m in micro.index]
            if available:
                common = [m for m in overlap if m in micro.index]
                gap = (float((monthly.loc[common, "informality_rate"]
                              - micro.loc[common, "tasa_informalidad_microdato"]).mean())
                       if common else 0.0)
                if abs(gap) > MAX_SPLICE_GAP:
                    raise SourceError(f"ICML and microdata informality differ by {gap:.3f}; check the classifier")
                ext = pd.DataFrame({"employed": micro.loc[available, "ocupados_microdato"],
                                    "informality_rate": micro.loc[available, "tasa_informalidad_microdato"] + gap})
                monthly = pd.concat([monthly, ext])
                self.report.sources["geih_extension"] = {"months": available, "overlap": common,
                                                         "rate_adjustment": gap}
                self.log(f"Labor extended with GEIH microdata {available[0]}..{available[-1]} "
                         f"(informality adjustment {gap:+.4f})")
        return to_quarters(monthly)

    # ------------------------------------------------------------------ earnings
    def earnings(self, last_quarter: str) -> pd.DataFrame:
        frozen = pd.read_csv(self.paths.earnings)
        last_frozen = frozen.quarter.iloc[-1]
        quarters = [str(q) for q in pd.period_range(pd.Period(last_frozen, "Q") + 1,
                                                   pd.Period(last_quarter, "Q"), freq="Q")]
        months = [m for q in quarters for m in quarter_months(q)]
        if self.options.validate_last_quarter:
            months = quarter_months(last_frozen) + months
        if not months:
            return frozen
        monthly = self.geih.months(months, log=self.log)
        computed = earnings_quarterly(monthly) if len(monthly) else pd.DataFrame()
        if self.options.validate_last_quarter and len(computed):
            check = computed.loc[computed.quarter.eq(last_frozen)]
            if len(check):
                diff = float(check.ratio.iloc[0] / frozen.ratio.iloc[-1] - 1)
                self.report.sources["validacion_ingresos"] = {"quarter": last_frozen, "relative_diff": diff}
                self.log(f"Earnings check {last_frozen}: frozen {frozen.ratio.iloc[-1]:.5f}, "
                         f"downloaded {check.ratio.iloc[0]:.5f} ({diff:+.3%})")
        new = computed.loc[computed.quarter.isin(quarters)] if len(computed) else computed
        if len(new):
            self.report.new_quarters = new.quarter.tolist()
            self.log(f"Earnings: new quarters {new.quarter.tolist()}")
            return pd.concat([frozen, new[frozen.columns.intersection(new.columns)]], ignore_index=True)
        return frozen

    # ------------------------------------------------------------------ checks
    @staticmethod
    def check(dataset: pd.DataFrame) -> None:
        if dataset.isna().any().any():
            bad = dataset.index[dataset.isna().any(axis=1)].tolist()
            raise SourceError(f"Missing values in quarters {bad}")
        expected = pd.period_range(dataset.index[0], dataset.index[-1], freq="Q").astype(str)
        if list(dataset.index) != list(expected):
            raise SourceError("The quarterly dataset has gaps")
        # Series are not seasonally adjusted: compare each quarter with the same quarter a year before.
        change = np.log(dataset[list(ACCOUNT_COLUMNS) + ["ocupados"]]).diff(4).abs()
        change = change.loc[~change.index.str[:4].isin(["2020", "2021"])]
        large = change.where(change > MAX_ANNUAL_LOG_CHANGE).stack().dropna()
        if len(large):
            raise SourceError(f"Implausible year-on-year change:\n{large}")

    def _backup(self, path: Path, stamp: str) -> None:
        if path.is_file():
            self.paths.versions.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, self.paths.versions / f"{path.stem}_{stamp}{path.suffix}")


def _statutory_costs(end_month: str) -> pd.DataFrame:
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "costo_no_salarial.py"
    spec = importlib.util.spec_from_file_location("costo_no_salarial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.quarterly(end=end_month)


def run_update(paths: UpdatePaths, options: UpdateOptions, log: Logger = print) -> UpdateReport:
    """Full update: download, splice, check and write every input of the estimation."""
    updater = DatasetUpdater(paths, options, log)
    previous = pd.read_csv(paths.dataset, encoding="utf-8-sig").set_index("quarter")
    accounts = updater.accounts(previous)
    last_month = str(pd.Period(accounts.index[-1], "Q").asfreq("M", how="end"))
    labor = updater.labor(last_month)
    dataset = accounts.join(labor, how="inner")
    dataset = dataset.loc[dataset.index >= previous.index[0], list(previous.columns)]
    dataset.index.name = "quarter"
    updater.check(dataset)
    earnings = updater.earnings(dataset.index[-1])
    costs = _statutory_costs(last_month)
    ratio = ValueAddedRatio(options.alpha, "employees").build(earnings)

    report = updater.report
    report.last_quarter = dataset.index[-1]
    common = previous.index.intersection(dataset.index)
    old_earnings = pd.read_csv(paths.earnings)
    report.changed = bool(
        dataset.index[-1] != previous.index[-1]
        or len(earnings) != len(old_earnings)
        or not np.allclose(dataset.loc[common].to_numpy(), previous.loc[common].to_numpy(), rtol=1e-10))
    report.notes.append(f"Dataset: {dataset.index[0]}-{dataset.index[-1]} (previous {previous.index[-1]}); "
                        f"{'changed' if report.changed else 'no new data'}")
    if options.dry_run:
        log("Dry run: nothing written")
        return report
    if not report.changed:
        log("No new data since the last update: nothing written")
        return report
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in (paths.dataset, paths.earnings, paths.value_added_ratio(options.alpha)):
        updater._backup(path, stamp)
    dataset.to_csv(paths.dataset, encoding="utf-8-sig")
    earnings.to_csv(paths.earnings, index=False)
    costs.to_csv(paths.costs, index=False)
    ratio.to_csv(paths.value_added_ratio(options.alpha), index=False)
    paths.manifest.write_text(json.dumps(report.__dict__, indent=2, ensure_ascii=False, default=str),
                              encoding="utf-8")
    log(f"Written: dataset to {dataset.index[-1]}, earnings to {earnings.quarter.iloc[-1]}")
    return report


__all__ = ["DatasetUpdater", "UpdateOptions", "UpdatePaths", "UpdateReport", "run_update"]
