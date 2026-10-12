"""Qualitaetsbericht nach dem Einlesen: was wurde gelesen, was passt nicht zusammen."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from kardio.constants import COL_DATE, COL_GRADE, COL_PATIENT, COL_REJECTION, VALUE_COLUMNS

MAX_LISTED = 3  # laengere Patientenlisten werden gekuerzt: "A, B, C … (+12)"


def _format_date(d) -> str:
    return d.strftime("%d.%m.%Y") if d is not None and not pd.isna(d) else "-"


def _format_names(names: list[str]) -> str:
    names = [str(n) for n in names]
    if not names:
        return "-"
    if len(names) <= MAX_LISTED + 1:  # "(+1)" spart nichts, dann lieber ausschreiben
        return ", ".join(names)
    return ", ".join(names[:MAX_LISTED]) + f" … (+{len(names) - MAX_LISTED})"


@dataclass
class QualityReport:
    n_patients_ecg: int
    n_patients_biopsy: int
    n_patients_matched: int
    patients_only_ecg: list[str]
    patients_only_biopsy: list[str]
    n_days: int
    n_empty_days: int
    n_dropped_duplicates: int
    n_biopsies: int
    n_rejections: int
    n_borderline: int
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None

    def as_rows(self) -> list[tuple[str, str]]:
        if self.date_min is None and self.date_max is None:
            period = "-"
        else:
            period = f"{_format_date(self.date_min)} bis {_format_date(self.date_max)}"
        return [
            ("Patienten mit EKG", str(self.n_patients_ecg)),
            ("Patienten mit Biopsien", str(self.n_patients_biopsy)),
            ("Patienten in beiden Tabellen", str(self.n_patients_matched)),
            ("Nur in EKG-Tabelle", _format_names(self.patients_only_ecg)),
            ("Nur in Biopsie-Tabelle", _format_names(self.patients_only_biopsy)),
            ("Messtage gesamt", str(self.n_days)),
            ("Davon Tage ohne Werte", str(self.n_empty_days)),
            ("Verworfene Mehrfachmessungen", str(self.n_dropped_duplicates)),
            ("Biopsien", str(self.n_biopsies)),
            ("Davon Abstoßung", str(self.n_rejections)),
            ("Davon Zwischengruppe", str(self.n_borderline)),
            ("Zeitraum", period),
        ]


def _patients(df: pd.DataFrame) -> set[str]:
    if df is None or COL_PATIENT not in df.columns:
        return set()
    return {str(p) for p in df[COL_PATIENT].dropna()}


def _count_grade(biopsies: pd.DataFrame, grade: int) -> int:
    if biopsies is None or len(biopsies) == 0:
        return 0
    if COL_GRADE in biopsies.columns:
        return int((biopsies[COL_GRADE] == grade).sum())
    if grade == 2 and COL_REJECTION in biopsies.columns:
        return int(biopsies[COL_REJECTION].astype(bool).sum())
    return 0


def build_report(daily: pd.DataFrame, dropped: pd.DataFrame, biopsies: pd.DataFrame) -> QualityReport:
    """daily/dropped aus select_daily(load_ecg(...)), biopsies aus load_biopsies(...)."""
    pe, pb = _patients(daily), _patients(biopsies)
    n_days = len(daily) if daily is not None else 0
    value_cols = [c for c in VALUE_COLUMNS if n_days and c in daily.columns]
    has_dates = bool(n_days) and COL_DATE in daily.columns and daily[COL_DATE].notna().any()
    return QualityReport(
        n_patients_ecg=len(pe), n_patients_biopsy=len(pb), n_patients_matched=len(pe & pb),
        patients_only_ecg=sorted(pe - pb), patients_only_biopsy=sorted(pb - pe),
        n_days=n_days,
        n_empty_days=int(daily[value_cols].isna().all(axis=1).sum()) if value_cols else 0,
        n_dropped_duplicates=len(dropped) if dropped is not None else 0,
        n_biopsies=len(biopsies) if biopsies is not None else 0,
        n_rejections=_count_grade(biopsies, 2), n_borderline=_count_grade(biopsies, 1),
        date_min=daily[COL_DATE].min() if has_dates else None,
        date_max=daily[COL_DATE].max() if has_dates else None,
    )
