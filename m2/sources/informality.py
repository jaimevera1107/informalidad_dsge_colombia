"""Labor informality of GEIH occupied persons, 2018 sampling frame (2022 onwards).

Port of the author's classifier ``CODIGO_GEIH/estadistica_recuperada/informalidad.py``
(branch ``MARCO2018_EXPERIMENTAL``, reconciled with DANE for April-June 2025) and of
the guard in ``ocupacion_formal_sector_geih.guard_informal``: a missing answer never
implies informality; persons without the required evidence stay unclassified (NaN).
Blanks created by the questionnaire's skip pattern are answers, not missing data.

Only the 2018-frame branch is ported because downloaded months are always recent;
the historical series before the last frozen quarter is never recomputed.

Rules
-----
* Unpaid family workers and other workers (P6430 = 6, 8): informal.
* Public administration and extraterritorial bodies (RAMA2D_R4 = 84, 99): formal.
* Wage earners (P6430 = 1, 2, 3, 7): formal if they contribute to health
  (P6100, P6110, P6450) and to a pension (P6920, P6930, P6940).
* Own-account workers and employers (P6430 = 4, 5): formal if the unit is
  registered (P3065, P3066, P3067...) or by the size and occupation rules for
  doubtful cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PUBLIC_BRANCHES = (84, 99)
FIRST_2018_FRAME_YEAR = 2022


class InformalityClassifier:
    """Formality status of occupied persons: 1 informal, 0 formal, NaN unclassified."""

    @staticmethod
    def _text(frame: pd.DataFrame, key: str) -> pd.Series:
        column = "SRC_" + key
        if column in frame:
            return frame[column].astype("string").str.strip().fillna("")
        return pd.Series("", index=frame.index, dtype="string")

    @classmethod
    def _number(cls, frame: pd.DataFrame, key: str) -> pd.Series:
        text = cls._text(frame, key).str.replace(",", ".", regex=False)
        return pd.to_numeric(text, errors="coerce").astype("float64")

    def classify(self, occupied: pd.DataFrame, characteristics: pd.DataFrame, period: str) -> pd.Series:
        """Status for ``occupied``; ``characteristics`` must share its index (P6100, P6110)."""
        year = int(period[:4])
        if year < FIRST_2018_FRAME_YEAR:
            raise ValueError(f"{period}: only the 2018 sampling frame (2022+) is supported")
        idx = occupied.index
        chars = characteristics.reindex(idx)
        pos = self._number(occupied, "P6430")
        out = pd.Series(np.nan, index=idx, dtype="float64")
        out.loc[pos.isin([6, 8])] = 1.0

        rama = self._number(occupied, "RAMA2D_R4")
        public = rama.isin(PUBLIC_BRANCHES)
        out.loc[public] = 0.0

        wage = pos.isin([1, 2, 3, 7]) & ~public
        p6100, p6110 = self._number(chars, "P6100"), self._number(chars, "P6110")
        p6450 = self._number(occupied, "P6450")
        health = ((p6100.isin([1, 2]) & p6110.isin([1, 2, 4]))
                  | ((p6100.eq(9) | p6110.eq(9)) & p6450.eq(2)))
        p6920, p6930, p6940 = (self._number(occupied, k) for k in ("P6920", "P6930", "P6940"))
        pension = p6920.eq(3) | (p6920.eq(1) & p6930.isin([1, 2, 3]) & p6940.isin([1, 3]))
        protected = (health & pension).fillna(False)
        out.loc[wage] = np.where(protected.loc[wage].to_numpy(dtype=bool), 0.0, 1.0)

        out.loc[pos.isin([4, 5]) & ~public] = 1.0 - self._independent(occupied, pos, public, year)

        # Guard: a wage earner stays unclassified only when health affiliation (P6090) or
        # pension (P6920) is unanswered. P6100 and P6110 are blank by the questionnaire's
        # skip pattern for persons not affiliated or in the subsidized regime; those
        # blanks are valid "no contribution" answers, not missing data.
        known = self._text(chars, "P6090").ne("") & self._text(occupied, "P6920").ne("")
        out.loc[pos.isin([1, 2, 3, 7]) & ~known] = np.nan
        return out

    def _independent(self, occ: pd.DataFrame, pos: pd.Series, public: pd.Series, year: int) -> pd.Series:
        """Formal (1) / informal (0) for own-account workers and employers."""
        independent = pos.isin([4, 5]) & ~public
        n = {k: self._number(occ, k) for k in ("P6765", "P3065", "P3066", "P3067", "P3067S1",
                                               "P3067S2", "P3068", "P3069", "P6775")}
        oficio = self._text(occ, "OFICIO_C8")
        if oficio.eq("").all():
            oficio = self._text(occ, "OFICIO")
        oficio2 = pd.to_numeric(oficio.str[:2], errors="coerce").astype("float64")
        formal = pd.Series(np.nan, index=occ.index, dtype="float64")

        def by_size_or_occupation(mask: pd.Series) -> None:
            employer = mask & pos.eq(5) & n["P3069"].notna()
            formal.loc[employer] = (n["P3069"].loc[employer] >= 4).astype(float)
            own = mask & pos.eq(4) & oficio2.notna()
            formal.loc[own] = (oficio2.loc[own] <= 20).astype(float)

        regular = independent & n["P6765"].ne(7)
        formal.loc[regular & n["P3065"].eq(1)] = 1.0
        no_registry = regular & n["P3065"].isin([2, 9])
        formal.loc[no_registry & n["P3066"].eq(1)] = 1.0
        formal.loc[no_registry & n["P3066"].eq(2)] = 0.0
        by_size_or_occupation(no_registry & n["P3066"].eq(9))

        special = independent & n["P6765"].eq(7)
        a = special & n["P3067"].eq(1) & n["P3067S1"].eq(1) & n["P3067S2"].notna()
        formal.loc[a] = (n["P3067S2"].loc[a] >= year - 1).astype(float)
        b = special & ((n["P3067"].eq(1) & n["P3067S1"].eq(2)) | n["P3067"].eq(2))
        formal.loc[b & n["P6775"].eq(1) & n["P3067"].eq(1)] = 1.0
        formal.loc[b & n["P6775"].eq(2)] = 0.0
        b3 = b & n["P6775"].eq(3) & oficio2.notna()
        formal.loc[b3] = (oficio2.loc[b3] <= 20).astype(float)
        formal.loc[b & n["P6775"].eq(1) & n["P3067"].eq(2) & n["P3068"].eq(1)] = 1.0
        formal.loc[b & n["P6775"].eq(1) & n["P3067"].eq(2) & n["P3068"].isin([2, 9])] = 0.0
        by_size_or_occupation(b & n["P6775"].eq(9))
        return formal.loc[independent]
