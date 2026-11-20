"""Offline tests of the update pipeline (no network)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m2.geih_earnings import classify_group, month_sums, quarterly  # noqa: E402
from m2.sources.geih import _plain, month_range, quarter_months  # noqa: E402
from m2.sources.icml import to_quarters  # noqa: E402
from m2.sources.informality import InformalityClassifier  # noqa: E402
from m2.sources.national_accounts import NationalAccounts  # noqa: E402


def _persons(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    occ = pd.DataFrame([{f"SRC_{k}": str(v) for k, v in r.items() if k not in ("P6090", "P6100", "P6110")}
                        for r in rows])
    chars = pd.DataFrame([{f"SRC_{k}": str(r[k]) for k in ("P6090", "P6100", "P6110") if k in r}
                          for r in rows])
    return occ, chars


def test_classifier_wage_earners_and_skip_pattern() -> None:
    occ, chars = _persons([
        # contributory health and pension -> formal
        {"P6430": 1, "RAMA2D_R4": 47, "P6090": 1, "P6100": 1, "P6110": 1, "P6920": 1, "P6930": 1, "P6940": 1},
        # subsidized health: P6110 blank by skip pattern -> informal, not unclassified
        {"P6430": 1, "RAMA2D_R4": 47, "P6090": 1, "P6100": 3, "P6920": 1, "P6930": 1, "P6940": 1},
        # no pension -> informal
        {"P6430": 1, "RAMA2D_R4": 47, "P6090": 1, "P6100": 1, "P6110": 1, "P6920": 2},
        # public administration -> formal
        {"P6430": 2, "RAMA2D_R4": 84, "P6090": 1, "P6100": 2, "P6110": 1, "P6920": 1, "P6930": 1, "P6940": 1},
        # pension unanswered -> unclassified
        {"P6430": 1, "RAMA2D_R4": 47, "P6090": 1, "P6100": 1, "P6110": 1},
        # unpaid family worker -> informal
        {"P6430": 6, "RAMA2D_R4": 47},
        # own-account registered in chamber of commerce -> formal
        {"P6430": 4, "RAMA2D_R4": 47, "P6765": 1, "P3065": 1},
        # own-account not registered, registry not renewed -> informal
        {"P6430": 4, "RAMA2D_R4": 47, "P6765": 1, "P3065": 2, "P3066": 2},
    ])
    status = InformalityClassifier().classify(occ, chars, "2026-08")
    expected = [0, 1, 1, 0, np.nan, 1, 0, 1]
    assert np.allclose(status.to_numpy(), expected, equal_nan=True)


def test_classifier_rejects_old_frame() -> None:
    occ, chars = _persons([{"P6430": 1}])
    with pytest.raises(ValueError):
        InformalityClassifier().classify(occ, chars, "2019-05")


def test_groups_and_quarterly_ratio() -> None:
    group = classify_group(pd.Series([4.0, 1.0, 1.0, 2.0, 3.0, 1.0]), pd.Series([np.nan, 2, 5, 9, np.nan, np.nan]))
    assert group.tolist() == ["own", "emp_small", "emp_large", "emp_large", "domestic", "emp_nosize"]
    occ = pd.DataFrame({"SRC_FEX_C18": ["100", "100", "200"], "SRC_P6430": ["4", "1", "1"],
                        "SRC_INGLABO": ["1000", "3000", "2000"], "SRC_P3069": ["", "5", "5"]})
    status = pd.Series([1.0, 0.0, 0.0])
    rows = [month_sums(p, occ, status).row() for p in ("2026-07", "2026-08", "2026-09")]
    out = quarterly(pd.DataFrame(rows))
    assert out.quarter.tolist() == ["2026Q3"]
    formal_mean = (100 * 3000 + 200 * 2000) / 300
    assert out.ratio.iloc[0] == pytest.approx(1000 / formal_mean)


def test_labor_quarters_require_three_months() -> None:
    monthly = pd.DataFrame({"employed": [10.0, 20.0, 30.0, 40.0], "informality_rate": [0.5, 0.5, 0.6, 0.5]},
                           index=["2026-01", "2026-02", "2026-03", "2026-04"])
    out = to_quarters(monthly)
    assert out.index.tolist() == ["2026Q1"]
    assert out.ocupados.iloc[0] == pytest.approx(20.0)
    assert out.ocupados_informales.iloc[0] == pytest.approx((5 + 10 + 18) / 3)


def test_accounts_parser(tmp_path: Path) -> None:
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.title = "Cuadro 1"
    sheet.append(["Producto Interno Bruto (PIB)"])
    sheet.append(["Series encadenadas de volumen con año de referencia 2015"])
    sheet.append(["Datos originales"])
    sheet.append(["Clasificación Cuentas Nacionales", "Concepto", 2005, None, None, None, "2006p"])
    sheet.append([None, None, "I", "II", "III", "IV", "I"])
    for code, base in (("B.1b", 100), ("P.311 + P.313 + P.323", 60), ("P.51", 20), ("P.312 + P.322", 15)):
        sheet.append([code, "x"] + [base + k for k in range(5)])
    sheet.append(["Fuente: DANE"])
    sheet.append(["Clasificación Cuentas Nacionales", "Concepto", 2005])   # growth block, ignored
    path = tmp_path / "anex.xlsx"
    book.save(path)
    frame = NationalAccounts.parse(path)
    assert frame.index.tolist() == ["2005Q1", "2005Q2", "2005Q3", "2005Q4", "2006Q1"]
    assert frame.columns.tolist() == ["pib", "c", "i", "g"]
    assert frame.loc["2006Q1", "pib"] == 104


def test_helpers() -> None:
    assert _plain("Caracter#U00edsticas generales") == "caracteristicas generales"
    assert month_range("2026-11", "2027-01") == ["2026-11", "2026-12", "2027-01"]
    assert quarter_months("2026Q3") == ["2026-07", "2026-08", "2026-09"]
