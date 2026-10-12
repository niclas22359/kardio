"""Station 4: statistischer Vergleich der Fenster-Merkmale zwischen den Biopsie-Gruppen.

Bildet die bisherige GraphPad-Prism-Auswertung der Klinik nach: drei Gruppen
(0 keine Abstossung, 1 Zwischengruppe, 2 Abstossung), Kruskal-Wallis ueber alle
Gruppen plus paarweise Mann-Whitney-U-Tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from kardio.config import AnalysisConfig
from kardio.constants import WINDOW_FEATURE_COLUMNS

RANKING_COLUMNS = ["column", "feature", "group", "metric", "agg", "n_no", "n_mid", "n_rej",
                   "p_kw", "p_01", "p_12", "p_02", "p_raw", "p_adj", "cliffs_delta", "auc",
                   "median_no", "median_mid", "median_rej",
                   # Tests, die mehrere Biopsien je Patient beruecksichtigen (stats/patient_level.py)
                   "n_paired", "diff_within", "share_same_direction",
                   "p_within", "p_within_adj", "p_perm", "p_perm_adj"]

_MIN_PER_GROUP = 2


def usable_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Only rows flagged usable; tolerates an empty table or a missing flag column."""
    if "usable" not in windows.columns:
        return windows
    return windows[windows["usable"].eq(True).to_numpy()]


def window_grades(windows: pd.DataFrame) -> np.ndarray:
    """Grade per row as float (NaN if unknown). Falls back to the boolean rejection column."""
    if "grade" in windows.columns:
        return pd.to_numeric(windows["grade"], errors="coerce").to_numpy(dtype=float)
    if "rejection" in windows.columns:
        return np.where(windows["rejection"].eq(True).to_numpy(), 2.0, 0.0)
    return np.full(len(windows), np.nan)


def _values(windows: pd.DataFrame, col: str) -> np.ndarray:
    if col not in windows.columns:
        return np.full(len(windows), np.nan)
    v = pd.to_numeric(windows[col], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(v), v, np.nan)


def _is_constant(values: np.ndarray) -> bool:
    return len(values) == 0 or bool(np.all(values == values[0]))


def _mwu(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < _MIN_PER_GROUP or len(b) < _MIN_PER_GROUP or _is_constant(np.r_[a, b]):
        return np.nan
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def _kruskal(groups: list[np.ndarray]) -> float:
    present = [g for g in groups if len(g) >= _MIN_PER_GROUP]
    if len(present) < 2 or _is_constant(np.concatenate(present)):
        return np.nan
    return float(kruskal(*present).pvalue)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Share of pairs with a > b minus share with a < b; NaN if a group is empty."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() - (diff < 0).mean())


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else np.nan


def compare_features(windows: pd.DataFrame, cfg: AnalysisConfig) -> tuple[pd.DataFrame, bool]:
    """Ranking with one row per window feature column plus a flag whether group sizes suffice."""
    w = usable_windows(windows)
    grade = window_grades(w)
    masks = {g: grade == g for g in (0, 1, 2)}

    rows = []
    for col in WINDOW_FEATURE_COLUMNS:
        feature, agg = col.split("__")
        group, metric = feature.rsplit("_", 1)
        v = _values(w, col)
        no, mid, rej = (v[masks[g]] for g in (0, 1, 2))
        no, mid, rej = no[~np.isnan(no)], mid[~np.isnan(mid)], rej[~np.isnan(rej)]
        delta = cliffs_delta(rej, no)
        # ROC AUC of a single column equals (delta + 1) / 2, ties counted half
        auc = np.nan if np.isnan(delta) else max((delta + 1) / 2, 1 - (delta + 1) / 2)
        p_02 = _mwu(no, rej)
        rows.append({
            "column": col, "feature": feature, "group": group, "metric": metric, "agg": agg,
            "n_no": len(no), "n_mid": len(mid), "n_rej": len(rej),
            "p_kw": _kruskal([no, mid, rej]),
            "p_01": _mwu(no, mid), "p_12": _mwu(mid, rej), "p_02": p_02,
            "p_raw": p_02, "p_adj": np.nan,
            "cliffs_delta": delta, "auc": auc,
            "median_no": _median(no), "median_mid": _median(mid), "median_rej": _median(rej),
        })

    from kardio.stats.patient_level import patient_level_tests
    rank = pd.DataFrame(rows).merge(patient_level_tests(windows, cfg), on="column", how="left")
    rank = rank[RANKING_COLUMNS]
    tested = rank["p_raw"].notna()
    if tested.any():
        rank.loc[tested, "p_adj"] = multipletests(rank.loc[tested, "p_raw"].to_numpy(),
                                                  method="fdr_bh")[1]
    rank = rank.sort_values(["p_adj", "p_raw"], na_position="last",
                            kind="stable").reset_index(drop=True)

    need = cfg.min_rejection_windows_stats
    reliable = int(masks[2].sum()) >= need and int(masks[0].sum()) >= need
    return rank, bool(reliable)
