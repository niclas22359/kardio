"""Station 3: condense the daily deviations into windows before biopsies and into a
trailing window for every measured day (daily risk curve).

Both outputs come from the same internal computation on a gapless calendar per patient,
so the window that ends at a biopsy date is identical to the rolling window of that day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kardio.config import AnalysisConfig
from kardio.constants import (AGGS, COL_BIOPSY_NO, COL_DATE, COL_GRADE, COL_PATIENT, DEV_FEATURES,
                              WINDOW_FEATURE_COLUMNS)
from kardio.features.baseline import rejection_flags, slope_from_view, to_day_number, trailing_view

WINDOW_META_COLUMNS = ["patient", "biopsy_no", "date", "grade", "rejection",
                       "n_data_days", "under_treatment", "usable"]
ROLLING_META_COLUMNS = ["patient", "date", "n_data_days"]

assert AGGS == ["zmean", "zmaxabs", "slope", "alertfrac"], "aggregate order is hard-wired below"


@dataclass
class _Calendar:
    """Trailing-window aggregates for every calendar day of every patient.

    Patient p owns the rows first_row[p] .. first_row[p] + n_rows[p] - 1; row i of that block is
    calendar day first_day[p] + i. The calendar runs window_days past the last measurement.
    """
    patients: np.ndarray      # (P,) patient names, sorted
    first_day: np.ndarray     # (P,) day number of the first measurement
    first_row: np.ndarray     # (P,)
    n_rows: np.ndarray        # (P,)
    patient_code: np.ndarray  # (N,)
    day: np.ndarray           # (N,) day number
    measured: np.ndarray      # (N,) the day has a row in the deviations table
    n_data_days: np.ndarray   # (N,)
    features: np.ndarray      # (N, 132) in WINDOW_FEATURE_COLUMNS order


def _empty_calendar() -> _Calendar:
    zero = np.zeros(0, dtype=np.int64)
    return _Calendar(np.zeros(0, dtype=object), zero, zero, zero, zero, zero, np.zeros(0, dtype=bool),
                     zero, np.zeros((0, len(WINDOW_FEATURE_COLUMNS))))


def _window_aggregates(dense_z: np.ndarray, window: int, z_alert: float) -> np.ndarray:
    """dense_z: (days, 33) on a gapless calendar -> (days, 132) aggregates of the trailing window."""
    view = trailing_view(dense_z, window)               # (days, 33, window)
    valid = np.isfinite(view)
    n = valid.sum(axis=-1)
    has = n > 0
    div = np.where(has, n, 1)
    absolute = np.abs(np.where(valid, view, 0.0))
    zmean = np.where(has, np.where(valid, view, 0.0).sum(axis=-1) / div, np.nan)
    zmaxabs = np.where(has, absolute.max(axis=-1), np.nan)
    alertfrac = np.where(has, (valid & (absolute > z_alert)).sum(axis=-1) / div, np.nan)
    slope = slope_from_view(view)
    return np.stack([zmean, zmaxabs, slope, alertfrac], axis=-1).reshape(dense_z.shape[0], -1)


def _rolling_calendar(deviations: pd.DataFrame, cfg: AnalysisConfig) -> _Calendar:
    """Shared core of build_windows and rolling_windows."""
    if len(deviations) == 0:
        return _empty_calendar()
    feat = pd.Index(DEV_FEATURES).get_indexer(deviations["feature"])
    keep = feat >= 0
    if not keep.any():
        return _empty_calendar()
    feat = feat[keep]
    codes, patients = pd.factorize(deviations[COL_PATIENT].to_numpy()[keep], sort=True)
    days = to_day_number(deviations[COL_DATE])[keep]
    z = deviations["z"].to_numpy(dtype=float)[keep]
    has_value = np.isfinite(deviations["value"].to_numpy(dtype=float)[keep])

    order = np.argsort(codes, kind="stable")
    codes, feat, days, z, has_value = codes[order], feat[order], days[order], z[order], has_value[order]
    bounds = np.flatnonzero(np.diff(codes)) + 1
    window = cfg.window_days + 1                        # [date - window_days, date] inclusive
    n_feat = len(DEV_FEATURES)

    first_day, n_rows, blocks, counts, measured_all, day_all, code_all = [], [], [], [], [], [], []
    for p, (start, stop) in enumerate(zip(np.r_[0, bounds], np.r_[bounds, len(codes)])):
        day0 = int(days[start:stop].min())
        offset = days[start:stop] - day0
        length = int(offset.max()) + 1 + cfg.window_days
        dense = np.full((length, n_feat), np.nan)
        dense[offset, feat[start:stop]] = z[start:stop]
        data_day = np.zeros(length, dtype=bool)
        data_day[offset[has_value[start:stop]]] = True
        measured = np.zeros(length, dtype=bool)
        measured[offset] = True
        csum = np.cumsum(np.r_[0, data_day.astype(np.int64)])
        lo = np.maximum(np.arange(length) + 1 - window, 0)
        counts.append(csum[1:] - csum[lo])
        blocks.append(_window_aggregates(dense, window, cfg.z_alert))
        first_day.append(day0)
        n_rows.append(length)
        measured_all.append(measured)
        day_all.append(day0 + np.arange(length, dtype=np.int64))
        code_all.append(np.full(length, p, dtype=np.int64))

    n_rows = np.asarray(n_rows, dtype=np.int64)
    return _Calendar(
        patients=np.asarray(patients, dtype=object),
        first_day=np.asarray(first_day, dtype=np.int64),
        first_row=np.r_[0, np.cumsum(n_rows)[:-1]].astype(np.int64),
        n_rows=n_rows,
        patient_code=np.concatenate(code_all),
        day=np.concatenate(day_all),
        measured=np.concatenate(measured_all),
        n_data_days=np.concatenate(counts),
        features=np.vstack(blocks),
    )


def _day_to_timestamp(day: np.ndarray) -> np.ndarray:
    return day.astype("datetime64[D]").astype("datetime64[ns]")


def rolling_windows(deviations: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Trailing window ending at every measured day of every patient.

    Columns: patient, date, n_data_days + WINDOW_FEATURE_COLUMNS; sorted by patient, date.
    A measured day is a day with a row in `deviations` (also if all its values are NaN).
    """
    cal = _rolling_calendar(deviations, cfg)
    rows = np.flatnonzero(cal.measured)
    out = pd.DataFrame(cal.features[rows], columns=WINDOW_FEATURE_COLUMNS)
    out.insert(0, "n_data_days", cal.n_data_days[rows].astype(int))
    out.insert(0, "date", _day_to_timestamp(cal.day[rows]))
    out.insert(0, "patient", cal.patients[cal.patient_code[rows]] if len(rows) else pd.Series(dtype=object))
    return out[ROLLING_META_COLUMNS + WINDOW_FEATURE_COLUMNS]


def _under_treatment(patient: np.ndarray, day: np.ndarray, rejection: np.ndarray, max_days: int) -> np.ndarray:
    """True if the same patient had a grade-2 biopsy strictly before, at most max_days earlier."""
    out = np.zeros(len(patient), dtype=bool)
    if not rejection.any():
        return out
    left = pd.DataFrame({"row": np.arange(len(patient)), "patient": patient, "day": day})
    right = pd.DataFrame({"patient": patient[rejection], "rej_day": day[rejection]})
    pairs = left.merge(right, on="patient", how="inner")
    gap = pairs["day"] - pairs["rej_day"]
    out[pairs.loc[(gap > 0) & (gap <= max_days), "row"].unique()] = True
    return out


def build_windows(deviations: pd.DataFrame, biopsies: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """One row per biopsy with the aggregates of the window [date - window_days, date].

    Columns: patient, biopsy_no, date, grade, rejection, n_data_days, under_treatment, usable
    + WINDOW_FEATURE_COLUMNS; sorted by patient, date, biopsy_no.
    """
    bx = biopsies.sort_values([COL_PATIENT, COL_DATE, COL_BIOPSY_NO], kind="stable").reset_index(drop=True)
    n = len(bx)
    patient = bx[COL_PATIENT].to_numpy(dtype=object)
    rejection = rejection_flags(bx)
    grade = (bx[COL_GRADE].to_numpy() if COL_GRADE in bx.columns else np.where(rejection, 2, 0)).astype(int)
    day = to_day_number(bx[COL_DATE]) if n else np.zeros(0, dtype=np.int64)

    features = np.full((n, len(WINDOW_FEATURE_COLUMNS)), np.nan)
    n_data_days = np.zeros(n, dtype=int)
    cal = _rolling_calendar(deviations, cfg)
    if n and len(cal.patients):
        code = pd.Index(cal.patients).get_indexer(patient)
        safe = np.maximum(code, 0)
        offset = day - cal.first_day[safe]
        found = (code >= 0) & (offset >= 0) & (offset < cal.n_rows[safe])
        rows = cal.first_row[safe[found]] + offset[found]
        features[found] = cal.features[rows]
        n_data_days[found] = cal.n_data_days[rows]

    under = _under_treatment(patient, day, rejection, cfg.window_days + cfg.treatment_days)
    out = pd.DataFrame(features, columns=WINDOW_FEATURE_COLUMNS)
    out.insert(0, "usable", (n_data_days >= cfg.min_window_data_days) & ~under)
    out.insert(0, "under_treatment", under)
    out.insert(0, "n_data_days", n_data_days)
    out.insert(0, "rejection", rejection)
    out.insert(0, "grade", grade)
    out.insert(0, "date", bx[COL_DATE].to_numpy())
    out.insert(0, "biopsy_no", bx[COL_BIOPSY_NO].to_numpy().astype(int))
    out.insert(0, "patient", patient)
    return out[WINDOW_META_COLUMNS + WINDOW_FEATURE_COLUMNS]
