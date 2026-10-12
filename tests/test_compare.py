import warnings

import numpy as np
import pandas as pd

from kardio.config import AnalysisConfig
from kardio.constants import AGGS, FEATURE_GROUPS, METRICS, WINDOW_FEATURE_COLUMNS
from kardio.stats.compare import compare_features
from tests.conftest import make_windows

COL = "R_rate_mean__zmean"
EXPECTED_COLUMNS = ["column", "feature", "group", "metric", "agg", "n_no", "n_mid", "n_rej",
                    "p_kw", "p_01", "p_12", "p_02", "p_raw", "p_adj", "cliffs_delta", "auc",
                    "median_no", "median_mid", "median_rej"]


def test_detects_shifted_feature():
    rank, reliable = compare_features(make_windows(20, 60, shift_col=COL, shift=2.0), AnalysisConfig())
    top = rank.iloc[0]
    assert reliable
    assert top["column"] == COL
    assert top["p_adj"] < 0.05 and top["auc"] > 0.8 and top["cliffs_delta"] > 0.5
    assert top["median_rej"] > top["median_no"]


def test_shape_and_name_parts():
    rank, _ = compare_features(make_windows(20, 60), AnalysisConfig())
    assert list(rank.columns)[:len(EXPECTED_COLUMNS)] == EXPECTED_COLUMNS
    assert len(rank) == 132 and set(rank["column"]) == set(WINDOW_FEATURE_COLUMNS)
    assert list(rank.index) == list(range(132))
    row = rank[rank["column"] == "qRT_interval_cv__alertfrac"].iloc[0]
    assert (row["feature"], row["group"], row["metric"], row["agg"]) == (
        "qRT_interval_cv", "qRT_interval", "cv", "alertfrac")
    assert set(rank["group"]) == set(FEATURE_GROUPS)
    assert set(rank["metric"]) == set(METRICS) and set(rank["agg"]) == set(AGGS)
    assert (rank["p_raw"] == rank["p_02"]).all()
    assert rank["auc"].between(0.5, 1.0).all()


def test_sorted_by_p_adj_then_p_raw():
    rank, _ = compare_features(make_windows(20, 60, shift_col=COL, shift=2.0), AnalysisConfig())
    assert rank["p_adj"].is_monotonic_increasing
    assert (rank["p_adj"] >= rank["p_raw"] - 1e-12).all()


def test_null_has_no_significant():
    rank, _ = compare_features(make_windows(20, 60), AnalysisConfig())
    assert (rank["p_adj"] < 0.05).sum() == 0


def test_three_groups():
    rank, reliable = compare_features(make_windows(20, 60, n_mid=15, shift_col=COL, shift=2.0),
                                      AnalysisConfig())
    assert reliable
    assert (rank["n_mid"] == 15).all() and (rank["n_rej"] == 20).all() and (rank["n_no"] == 60).all()
    assert rank[["p_kw", "p_01", "p_12", "p_02"]].notna().all().all()
    top = rank.iloc[0]
    assert top["column"] == COL and top["p_kw"] < 0.05
    assert top["median_no"] < top["median_mid"] < top["median_rej"]


def test_two_groups_have_nan_mid_values():
    rank, _ = compare_features(make_windows(20, 60), AnalysisConfig())
    assert (rank["n_mid"] == 0).all()
    assert rank[["p_01", "p_12", "median_mid"]].isna().all().all()
    assert rank["p_kw"].notna().all()


def test_too_few_rejections_flagged():
    rank, reliable = compare_features(make_windows(2, 20), AnalysisConfig())
    assert not reliable and len(rank) == 132


def test_too_few_controls_flagged():
    _, reliable = compare_features(make_windows(20, 3), AnalysisConfig())
    assert not reliable


def test_no_rejections_does_not_crash():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rank, reliable = compare_features(make_windows(0, 10), AnalysisConfig())
    assert not reliable and len(rank) == 132
    assert rank["p_raw"].isna().all() and rank["p_adj"].isna().all()
    assert rank["cliffs_delta"].isna().all() and rank["median_rej"].isna().all()
    assert rank["median_no"].notna().all()


def test_only_rejections_does_not_crash():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rank, reliable = compare_features(make_windows(10, 0), AnalysisConfig())
    assert not reliable and rank["p_raw"].isna().all() and rank["p_kw"].isna().all()


def test_empty_table():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rank, reliable = compare_features(make_windows(0, 0), AnalysisConfig())
    assert len(rank) == 132 and not reliable
    assert list(rank.columns)[:len(EXPECTED_COLUMNS)] == EXPECTED_COLUMNS
    assert (rank[["n_no", "n_mid", "n_rej"]] == 0).all().all()


def test_constant_and_all_nan_columns():
    w = make_windows(20, 60, n_mid=10, shift_col=COL, shift=2.0)
    w["T_amplitude_mean__slope"] = 1.5
    w["P_duration_cv__zmaxabs"] = np.nan
    w.loc[w["grade"] == 2, "PR_interval_dev__zmean"] = np.nan
    w.loc[w.index[::5], "R_upslope_mean__zmean"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rank, reliable = compare_features(w, AnalysisConfig())
    assert reliable and len(rank) == 132
    by = rank.set_index("column")
    for col in ["T_amplitude_mean__slope", "P_duration_cv__zmaxabs"]:
        assert by.loc[col, ["p_kw", "p_01", "p_12", "p_02", "p_raw", "p_adj"]].isna().all()
    assert by.loc["P_duration_cv__zmaxabs", "n_rej"] == 0
    assert np.isnan(by.loc["P_duration_cv__zmaxabs", "cliffs_delta"])
    assert by.loc["T_amplitude_mean__slope", "cliffs_delta"] == 0.0
    # eine Gruppe ohne Werte: der Vergleich 0 gegen 1 bleibt moeglich
    assert np.isnan(by.loc["PR_interval_dev__zmean", "p_02"])
    assert not np.isnan(by.loc["PR_interval_dev__zmean", "p_01"])
    assert not np.isnan(by.loc["PR_interval_dev__zmean", "p_kw"])
    assert by.loc["R_upslope_mean__zmean", "n_no"] < 60
    assert not np.isnan(by.loc["R_upslope_mean__zmean", "p_02"])
    assert rank.iloc[0]["column"] == COL
    # NaN-Zeilen stehen am Ende
    assert rank["p_adj"].isna().iloc[-3:].all() and rank["p_adj"].notna().iloc[:129].all()


def test_unusable_rows_ignored():
    w = make_windows(20, 60, shift_col=COL, shift=2.0)
    extra = make_windows(30, 0, shift_col=COL, shift=-50.0, seed=1)
    extra["usable"] = False
    both = pd.concat([w, extra], ignore_index=True)
    rank_a, _ = compare_features(w, AnalysisConfig())
    rank_b, _ = compare_features(both, AnalysisConfig())
    pd.testing.assert_frame_equal(rank_a, rank_b)
    w2 = w.copy()
    w2.loc[w2["grade"] == 2, "usable"] = [True] * 3 + [False] * 17
    rank_c, reliable = compare_features(w2, AnalysisConfig())
    assert not reliable and (rank_c["n_rej"] == 3).all()


def test_negative_shift():
    rank, _ = compare_features(make_windows(20, 60, shift_col=COL, shift=-2.0), AnalysisConfig())
    top = rank.iloc[0]
    assert top["column"] == COL and top["cliffs_delta"] < -0.5 and top["auc"] > 0.8


def test_input_not_modified():
    w = make_windows(20, 60, n_mid=5)
    before = w.copy()
    compare_features(w, AnalysisConfig())
    pd.testing.assert_frame_equal(w, before)
