"""Station 2: personal baseline per patient and daily deviations from it.

Everything is computed on dense numpy matrices (rows = patient days, columns = features);
there are no per-row Python loops, only one cheap loop over patients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from kardio.config import AnalysisConfig
from kardio.constants import (COL_DATE, COL_GRADE, COL_PATIENT, COL_REJECTION, DEV_FEATURES,
                              FEATURE_GROUPS)

MAD_TO_SD = 1.4826
REL_SCALE_FLOOR = 0.05   # scale of mean/cv is at least 5 % of |center|
ABS_SCALE_FLOOR = 1e-9
DEV_SCALE_FLOOR = 0.02   # scale of the relative deviation is at least 2 percentage points
TREND_DAYS = 7
MIN_SLOPE_POINTS = 3

BASELINE_COLUMNS = ["patient", "feature", "center", "scale", "n_days", "weak"]
DEVIATION_COLUMNS = ["patient", "date", "feature", "value", "z", "delta_prev", "trend_7d"]

_N_GROUPS = len(FEATURE_GROUPS)
_MEAN_COLS = [f"{g}_mean" for g in FEATURE_GROUPS]
_SD_COLS = [f"{g}_sd" for g in FEATURE_GROUPS]


# ---------------------------------------------------------------- shared numeric helpers

def to_day_number(dates) -> np.ndarray:
    """Calendar day as integer (days since 1970-01-01); time of day is dropped."""
    return pd.to_datetime(pd.Series(dates)).to_numpy().astype("datetime64[D]").astype(np.int64)


def trailing_view(dense: np.ndarray, window: int) -> np.ndarray:
    """dense: (days, features) on a gapless calendar. Returns a (days, features, window) view;
    entry [t, f, k] is the value of day t - (window - 1) + k (NaN before the first day)."""
    pad = np.full((window - 1, dense.shape[1]), np.nan)
    return sliding_window_view(np.vstack([pad, dense]), window, axis=0)


def slope_from_view(view: np.ndarray, min_points: int = MIN_SLOPE_POINTS) -> np.ndarray:
    """Least-squares slope per calendar day over the last axis, ignoring NaN.
    NaN where fewer than `min_points` values are present."""
    valid = np.isfinite(view)
    x = np.arange(view.shape[-1], dtype=float)
    y = np.where(valid, view, 0.0)
    n = valid.sum(axis=-1)
    sx = (valid * x).sum(axis=-1)
    sxx = (valid * (x * x)).sum(axis=-1)
    sy = y.sum(axis=-1)
    sxy = (y * x).sum(axis=-1)
    den = n * sxx - sx * sx
    ok = (n >= min_points) & (den > 0)
    return np.where(ok, (n * sxy - sx * sy) / np.where(ok, den, 1.0), np.nan)


def rejection_flags(biopsies: pd.DataFrame) -> np.ndarray:
    """Boolean array: biopsy is a rejection (grade 2)."""
    if COL_REJECTION in biopsies.columns:
        return biopsies[COL_REJECTION].fillna(False).to_numpy(dtype=bool)
    if COL_GRADE in biopsies.columns:
        return (biopsies[COL_GRADE] == 2).to_numpy(dtype=bool)
    return np.zeros(len(biopsies), dtype=bool)


# ---------------------------------------------------------------- internal preparation

def _prepare(daily: pd.DataFrame):
    """Sort by patient/date and return (sorted frame, patient codes, patient names, day numbers)."""
    d = daily.sort_values([COL_PATIENT, COL_DATE], kind="stable").reset_index(drop=True)
    codes, patients = pd.factorize(d[COL_PATIENT], sort=True)
    return d, codes, np.asarray(patients, dtype=object), to_day_number(d[COL_DATE])


def _mean_and_cv(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(rows, 11) matrices: absolute value and coefficient of variation sd / |mean|."""
    mean = d.reindex(columns=_MEAN_COLS).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    sd = d.reindex(columns=_SD_COLS).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    mean = np.where(np.isfinite(mean), mean, np.nan)
    sd = np.where(np.isfinite(sd), sd, np.nan)
    ok = np.isfinite(mean) & (mean != 0)
    cv = np.where(ok, sd / np.where(ok, np.abs(mean), 1.0), np.nan)
    return mean, cv


def _relative_deviation(mean: np.ndarray, center: np.ndarray) -> np.ndarray:
    """|mean - center| / |center|, NaN where the center is 0 or missing."""
    ok = np.isfinite(center) & (center != 0)
    return np.where(ok, np.abs(mean - center) / np.where(ok, np.abs(center), 1.0), np.nan)


def _rejection_phase_mask(codes, patients, days, biopsies: pd.DataFrame, cfg: AnalysisConfig) -> np.ndarray:
    """True for days from window_days before until treatment_days after a grade-2 biopsy."""
    mask = np.zeros(len(codes), dtype=bool)
    if len(biopsies) == 0:
        return mask
    rej = biopsies[rejection_flags(biopsies)]
    if len(rej) == 0:
        return mask
    rej_codes = pd.Index(patients).get_indexer(rej[COL_PATIENT])
    rej_days = to_day_number(rej[COL_DATE])
    for code, day in zip(rej_codes, rej_days):
        if code >= 0:
            mask |= (codes == code) & (days >= day - cfg.window_days) & (days <= day + cfg.treatment_days)
    return mask


def _group_stats(values: np.ndarray, groups: np.ndarray, n_groups: int):
    """Median, 1.4826*MAD and count of non-NaN values per group and column -> (n_groups, cols) each."""
    frame = pd.DataFrame(values)
    idx = pd.RangeIndex(n_groups)
    if len(frame) == 0:
        nan = np.full((n_groups, values.shape[1]), np.nan)
        return nan, nan.copy(), np.zeros((n_groups, values.shape[1]), dtype=int)
    grouped = frame.groupby(groups)
    median = grouped.median().reindex(idx).to_numpy(dtype=float)
    count = grouped.count().reindex(idx, fill_value=0).to_numpy(dtype=int)
    spread = (frame - median[groups]).abs().groupby(groups).median().reindex(idx).to_numpy(dtype=float)
    return median, MAD_TO_SD * spread, count


def _pooled_mad(values: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values)
    return MAD_TO_SD * (frame - frame.median()).abs().median().to_numpy(dtype=float)


# ---------------------------------------------------------------- public API

def compute_baselines(daily: pd.DataFrame, biopsies: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Personal normal per patient and feature from the first stable days.

    Returns one row per patient x 33 features: patient, feature, center, scale, n_days, weak.
    """
    if len(daily) == 0:
        return pd.DataFrame({"patient": pd.Series(dtype=object), "feature": pd.Series(dtype=object),
                             "center": pd.Series(dtype=float), "scale": pd.Series(dtype=float),
                             "n_days": pd.Series(dtype=int), "weak": pd.Series(dtype=bool)})[BASELINE_COLUMNS]
    d, codes, patients, days = _prepare(daily)
    n_pat = len(patients)
    mean, cv = _mean_and_cv(d)
    values = np.hstack([mean, cv])                      # (rows, 22)

    has_data = np.isfinite(values).any(axis=1)
    # Standard: die Normallinie kennt die Befunde NICHT. Wuerde man nur die Tage um Abstossungen
    # ausschliessen, laegen Biopsien ohne Abstossung in ihrer eigenen Normallinie, Abstossungen aber nie.
    # Diese einseitige Behandlung kann Unterschiede vortaeuschen, vor allem bei wenigen Messtagen.
    # Median und MAD sind robust genug, um einzelne auffaellige Phasen zu verkraften.
    clean = has_data
    if cfg.baseline_exclude_rejection:
        clean = clean & ~_rejection_phase_mask(codes, patients, days, biopsies, cfg)
    rank = pd.Series(clean.astype(int)).groupby(codes).cumsum().to_numpy()
    # baseline_days = 0 (Standard): alle stabilen Tage des Patienten bilden die Normallinie.
    # Nur die ersten N Tage zu nehmen verfaelscht den Vergleich: fruehe Biopsien OHNE Abstossung
    # laegen dann fast deckungsgleich in der Normallinie (Abweichung per Konstruktion klein),
    # Abstossungsfenster sind dagegen immer ausgeschlossen. Das erzeugt Scheintreffer ohne jedes Muster.
    n_base = cfg.baseline_days if cfg.baseline_days and cfg.baseline_days > 0 else np.iinfo(np.int64).max
    is_base = clean & (rank <= n_base)

    median, mad, count = _group_stats(values[is_base], codes[is_base], n_pat)
    weak = count < cfg.baseline_min_days

    # population fallback: median of patient medians, pooled spread over all clean days
    pat_median, _, _ = _group_stats(values[clean], codes[clean], n_pat)
    pop_center = pd.DataFrame(pat_median).median().to_numpy(dtype=float)
    pop_scale = _pooled_mad(values[clean])

    center = np.where(weak, pop_center, median)
    scale = np.where(weak, pop_scale, mad)
    floor = np.maximum(REL_SCALE_FLOOR * np.abs(center), ABS_SCALE_FLOOR)
    scale = np.where(np.isfinite(scale), np.maximum(scale, floor), floor)
    scale = np.where(np.isfinite(center), scale, np.nan)

    # relative deviation from the personal normal of g_mean
    mean_center = center[:, :_N_GROUPS]
    weak_mean = weak[:, :_N_GROUPS]
    dev = _relative_deviation(mean, mean_center[codes])
    _, dev_mad, dev_count = _group_stats(dev[is_base], codes[is_base], n_pat)
    dev_scale = np.where(weak_mean, _pooled_mad(dev[clean]), dev_mad)
    dev_scale = np.where(np.isfinite(dev_scale), np.maximum(dev_scale, DEV_SCALE_FLOOR), DEV_SCALE_FLOOR)

    def interleave(m, c, v):
        """(patients, 11) x 3 -> flat in DEV_FEATURES order (g_mean, g_cv, g_dev per group)."""
        return np.stack([m, c, v], axis=-1).reshape(-1)

    g = _N_GROUPS
    return pd.DataFrame({
        "patient": np.repeat(patients, len(DEV_FEATURES)),
        "feature": np.tile(np.asarray(DEV_FEATURES, dtype=object), n_pat),
        "center": interleave(center[:, :g], center[:, g:], np.zeros((n_pat, g))),
        "scale": interleave(scale[:, :g], scale[:, g:], dev_scale),
        "n_days": interleave(count[:, :g], count[:, g:], dev_count).astype(int),
        "weak": interleave(weak[:, :g], weak[:, g:], weak_mean).astype(bool),
    })[BASELINE_COLUMNS]


def _baseline_matrix(baselines: pd.DataFrame, patients: np.ndarray, column: str) -> np.ndarray:
    """(patients, 33) matrix of one baseline column; NaN where a baseline is missing."""
    if len(baselines) == 0:
        return np.full((len(patients), len(DEV_FEATURES)), np.nan)
    wide = (baselines.drop_duplicates(["patient", "feature"], keep="last")
            .pivot(index="patient", columns="feature", values=column))
    return wide.reindex(index=patients, columns=DEV_FEATURES).to_numpy(dtype=float)


def compute_deviations(daily: pd.DataFrame, baselines: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Every day of every patient as deviation from the personal baseline (long format).

    One row per patient x date x 33 features, sorted by patient, feature, date; rows without
    a value are kept. Columns: patient, date, feature, value, z, delta_prev, trend_7d.
    """
    if len(daily) == 0:
        return pd.DataFrame({"patient": pd.Series(dtype=object), "date": pd.to_datetime([]),
                             "feature": pd.Series(dtype=object), "value": pd.Series(dtype=float),
                             "z": pd.Series(dtype=float), "delta_prev": pd.Series(dtype=float),
                             "trend_7d": pd.Series(dtype=float)})[DEVIATION_COLUMNS]
    d, codes, patients, days = _prepare(daily)
    n_rows, n_feat = len(d), len(DEV_FEATURES)
    center = _baseline_matrix(baselines, patients, "center")[codes]   # (rows, 33)
    scale = _baseline_matrix(baselines, patients, "scale")[codes]

    mean, cv = _mean_and_cv(d)
    value = np.empty((n_rows, n_feat))
    value[:, 0::3] = mean
    value[:, 1::3] = cv
    value[:, 2::3] = _relative_deviation(mean, center[:, 0::3])

    ok = np.isfinite(scale) & (scale > 0)
    z = np.where(ok, (value - center) / np.where(ok, scale, 1.0), np.nan)
    z = np.where(np.isfinite(z), z, np.nan)

    # value minus the previous available value of the same patient/feature
    frame = pd.DataFrame(value)
    previous = frame.groupby(codes).ffill().groupby(codes).shift(1).to_numpy(dtype=float)
    delta_prev = value - previous

    # slope of z over the last 7 calendar days, on a gapless calendar per patient
    trend = np.full((n_rows, n_feat), np.nan)
    bounds = np.flatnonzero(np.diff(codes)) + 1
    for start, stop in zip(np.r_[0, bounds], np.r_[bounds, n_rows]):
        offset = days[start:stop] - days[start]
        dense = np.full((int(offset[-1]) + 1, n_feat), np.nan)
        dense[offset] = z[start:stop]
        trend[start:stop] = slope_from_view(trailing_view(dense, TREND_DAYS))[offset]

    feat_idx = np.tile(np.arange(n_feat), n_rows)
    row_idx = np.repeat(np.arange(n_rows), n_feat)
    order = np.lexsort((row_idx, feat_idx, codes[row_idx]))   # patient, feature, date
    row_idx, feat_idx = row_idx[order], feat_idx[order]
    return pd.DataFrame({
        "patient": patients[codes[row_idx]],
        "date": d[COL_DATE].to_numpy()[row_idx],
        "feature": np.asarray(DEV_FEATURES, dtype=object)[feat_idx],
        "value": value[row_idx, feat_idx],
        "z": z[row_idx, feat_idx],
        "delta_prev": delta_prev[row_idx, feat_idx],
        "trend_7d": trend[row_idx, feat_idx],
    })[DEVIATION_COLUMNS]
