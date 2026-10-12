import numpy as np
import pandas as pd
import pytest

from kardio.config import AnalysisConfig
from kardio.constants import DEV_FEATURES, FEATURE_GROUPS
from kardio.features.baseline import compute_baselines, compute_deviations
from tests.conftest import empty_biopsies, make_biopsies, make_values

DEV_COLUMNS = ["patient", "date", "feature", "value", "z", "delta_prev", "trend_7d"]


def _row(b: pd.DataFrame, patient: str, feature: str) -> pd.Series:
    return b[(b.patient == patient) & (b.feature == feature)].iloc[0]


def _series(dev: pd.DataFrame, patient: str, feature: str) -> pd.DataFrame:
    return dev[(dev.patient == patient) & (dev.feature == feature)].sort_values("date").reset_index(drop=True)


def _hand_daily(means, sds, dates=None, patient="H1") -> pd.DataFrame:
    """One patient; R_rate_mean / R_rate_sd set by hand, the other columns constant."""
    rows = []
    for i, (m, s) in enumerate(zip(means, sds)):
        vals = make_values()
        vals["R_rate_mean"], vals["R_rate_sd"] = m, s
        date = pd.Timestamp(dates[i]) if dates else pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
        rows.append({"patient": patient, "date": date, "included_no": 30, "cc": 0.95, **vals})
    return pd.DataFrame(rows)


def _unit_baselines(patients) -> pd.DataFrame:
    """center 0, scale 1 for every feature, so z equals the value."""
    return pd.DataFrame([{"patient": p, "feature": f, "center": 0.0, "scale": 1.0, "n_days": 14, "weak": False}
                         for p in patients for f in DEV_FEATURES])


def test_baseline_per_patient(daily_df):
    cfg = AnalysisConfig(baseline_days=14, baseline_exclude_rejection=True)  # ausdruecklich: alte Regel
    b = compute_baselines(daily_df, empty_biopsies(), cfg)
    assert list(b.columns) == ["patient", "feature", "center", "scale", "n_days", "weak"]
    assert len(b) == 3 * 33
    assert set(b["patient"]) == {"P1", "P2", "P3"}
    assert set(b["feature"]) == set(DEV_FEATURES)
    p1 = _row(b, "P1", "R_rate_mean")
    assert abs(p1["center"] - 74) < 5 and p1["scale"] > 0 and not p1["weak"]
    assert p1["n_days"] == cfg.baseline_days
    expected = daily_df[daily_df.patient == "P1"].sort_values("date").head(14)["R_rate_mean"].median()
    assert p1["center"] == pytest.approx(expected)
    cv = _row(b, "P1", "R_rate_cv")
    assert cv["center"] == pytest.approx(0.1)
    assert b["weak"].dtype == bool


def test_weak_patient_uses_population(daily_df):
    cfg = AnalysisConfig()
    b = compute_baselines(daily_df, empty_biopsies(), cfg)
    p3 = b[b.patient == "P3"]
    assert p3["weak"].all()
    assert not b[b.patient == "P1"]["weak"].any()
    r = _row(b, "P3", "R_rate_mean")
    assert r["n_days"] == 2
    expected = daily_df.groupby("patient")["R_rate_mean"].median().median()
    assert r["center"] == pytest.approx(expected)
    assert np.isfinite(r["scale"]) and r["scale"] > 0
    dev = _row(b, "P3", "R_rate_dev")
    assert dev["weak"] and dev["center"] == 0 and dev["scale"] >= 0.02


def test_empty_population_gives_nan_center():
    d = _hand_daily([np.nan, np.nan], [np.nan, np.nan])
    value_cols = [c for c in d.columns if c not in ("patient", "date", "included_no", "cc")]
    d[value_cols] = np.nan
    cfg = AnalysisConfig()
    b = compute_baselines(d, empty_biopsies(), cfg)
    assert len(b) == 33 and b["weak"].all()
    assert b.loc[b.feature == "R_rate_mean", "center"].isna().all()
    dev = compute_deviations(d, b, cfg)
    assert len(dev) == 2 * 33
    assert dev["z"].isna().all()


def test_baseline_excludes_rejection_phase(daily_df):
    cfg = AnalysisConfig(baseline_days=14, baseline_exclude_rejection=True)  # ausdruecklich: alte Regel
    bx = make_biopsies([("P1", 1, "2025-01-10", 2)])
    d = daily_df.copy()
    mask = (d.patient == "P1") & (d.date <= "2025-01-10")
    d.loc[mask, "R_rate_mean"] = 200  # outliers before the grade-2 biopsy
    b = compute_baselines(d, bx, cfg)
    assert _row(b, "P1", "R_rate_mean")["center"] < 100
    # without the biopsy the outliers dominate
    b0 = compute_baselines(d, empty_biopsies(), cfg)
    assert _row(b0, "P1", "R_rate_mean")["center"] == 200


def test_grade_one_biopsy_does_not_exclude(daily_df):
    cfg = AnalysisConfig(baseline_days=14, baseline_exclude_rejection=True)  # ausdruecklich: alte Regel
    d = daily_df.copy()
    d.loc[(d.patient == "P1") & (d.date <= "2025-01-10"), "R_rate_mean"] = 200
    b = compute_baselines(d, make_biopsies([("P1", 1, "2025-01-10", 1)]), cfg)
    assert _row(b, "P1", "R_rate_mean")["center"] == 200


def test_deviations_shape_and_z(daily_df):
    cfg = AnalysisConfig()
    b = compute_baselines(daily_df, empty_biopsies(), cfg)
    dev = compute_deviations(daily_df, b, cfg)
    assert list(dev.columns) == DEV_COLUMNS
    assert dev["feature"].nunique() == 33
    assert len(dev) == len(daily_df) * 33
    assert dev["z"].abs().median() < 2
    assert not np.isinf(dev[["value", "z", "delta_prev", "trend_7d"]].to_numpy()).any()
    s = _series(dev, "P1", "R_rate_mean")
    assert np.isnan(s.loc[0, "delta_prev"])
    assert s.loc[1, "delta_prev"] == pytest.approx(s.loc[1, "value"] - s.loc[0, "value"])
    assert len(s) == 30 and s["date"].is_monotonic_increasing


def test_constant_column_gives_finite_z(daily_df):
    d = daily_df.copy()
    d["P_duration_mean"] = 0.123
    d["T_duration_mean"] = 0.0
    cfg = AnalysisConfig()
    b = compute_baselines(d, empty_biopsies(), cfg)
    assert (b.loc[b.center.notna(), "scale"] > 0).all()
    assert np.isfinite(b.loc[b.center.notna(), "scale"]).all()
    dev = compute_deviations(d, b, cfg)
    assert np.isfinite(dev.loc[dev.feature == "P_duration_mean", "z"]).all()
    assert np.isfinite(dev.loc[dev.feature == "T_duration_mean", "z"]).all()
    assert not np.isinf(dev["z"].to_numpy()).any()
    # mean 0 -> cv and relative deviation are undefined
    assert dev.loc[dev.feature == "T_duration_cv", "value"].isna().all()
    assert dev.loc[dev.feature == "T_duration_dev", "value"].isna().all()


def test_cv_and_dev_hand_made():
    cfg = AnalysisConfig(baseline_min_days=3)
    d = _hand_daily([100, 110, 90], [10, 22, 9])
    b = compute_baselines(d, empty_biopsies(), cfg)
    mean = _row(b, "H1", "R_rate_mean")
    assert mean["center"] == pytest.approx(100) and not mean["weak"] and mean["n_days"] == 3
    assert mean["scale"] == pytest.approx(1.4826 * 10)
    cv = _row(b, "H1", "R_rate_cv")
    assert cv["center"] == pytest.approx(0.1)
    assert cv["scale"] == pytest.approx(0.005)  # MAD 0 -> floor 5 % of center
    dv = _row(b, "H1", "R_rate_dev")
    assert dv["center"] == 0 and dv["scale"] == pytest.approx(0.02) and not dv["weak"]

    dev = compute_deviations(d, b, cfg)
    m = _series(dev, "H1", "R_rate_mean")
    assert m["value"].tolist() == [100, 110, 90]
    assert m["z"].to_numpy() == pytest.approx(np.array([0, 10, -10]) / 14.826)
    c = _series(dev, "H1", "R_rate_cv")
    assert c["value"].to_numpy() == pytest.approx([0.1, 0.2, 0.1])
    assert c["z"].to_numpy() == pytest.approx([0, 20, 0], abs=1e-9)
    r = _series(dev, "H1", "R_rate_dev")
    assert r["value"].to_numpy() == pytest.approx([0, 0.1, 0.1])
    assert r["z"].to_numpy() == pytest.approx([0, 5, 5])


def test_delta_prev_skips_missing_values():
    cfg = AnalysisConfig(baseline_min_days=2)
    d = _hand_daily([100, np.nan, 90], [10, np.nan, 9])
    dev = compute_deviations(d, compute_baselines(d, empty_biopsies(), cfg), cfg)
    s = _series(dev, "H1", "R_rate_mean")
    assert len(s) == 3  # NaN row is kept
    assert np.isnan(s.loc[1, "value"]) and np.isnan(s.loc[1, "z"]) and np.isnan(s.loc[1, "delta_prev"])
    assert s.loc[2, "delta_prev"] == pytest.approx(-10)


def test_trend_uses_calendar_days():
    cfg = AnalysisConfig()
    dates = ["2025-01-01", "2025-01-03", "2025-01-05", "2025-01-20"]
    d = _hand_daily([0.0, 2.0, 4.0, 9.0], [1, 1, 1, 1], dates=dates)
    dev = compute_deviations(d, _unit_baselines(["H1"]), cfg)
    s = _series(dev, "H1", "R_rate_mean")
    assert s["z"].tolist() == [0.0, 2.0, 4.0, 9.0]
    assert np.isnan(s.loc[0, "trend_7d"]) and np.isnan(s.loc[1, "trend_7d"])
    assert s.loc[2, "trend_7d"] == pytest.approx(1.0)  # per calendar day, not per row
    assert np.isnan(s.loc[3, "trend_7d"])  # only one point in the last 7 days


def test_single_day_patient_and_unsorted_input(daily_df):
    cfg = AnalysisConfig()
    d = pd.concat([daily_df, _hand_daily([70], [7], patient="S1")]).sample(frac=1, random_state=1)
    b = compute_baselines(d, empty_biopsies(), cfg)
    assert _row(b, "S1", "R_rate_mean")["weak"]
    ref = compute_baselines(daily_df, empty_biopsies(), cfg)
    assert _row(b, "P1", "R_rate_mean")["center"] == pytest.approx(_row(ref, "P1", "R_rate_mean")["center"])
    dev = compute_deviations(d, b, cfg)
    assert len(dev) == len(d) * 33
    assert len(dev[dev.patient == "S1"]) == 33
    ref_dev = compute_deviations(daily_df, ref, cfg)
    a = _series(dev, "P2", "T_amplitude_cv")
    e = _series(ref_dev, "P2", "T_amplitude_cv")
    np.testing.assert_allclose(a["trend_7d"].to_numpy(), e["trend_7d"].to_numpy(), equal_nan=True)
    np.testing.assert_allclose(a["delta_prev"].to_numpy(), e["delta_prev"].to_numpy(), equal_nan=True)


def test_empty_daily():
    cfg = AnalysisConfig()
    d = _hand_daily([70], [7]).iloc[0:0]
    b = compute_baselines(d, empty_biopsies(), cfg)
    assert len(b) == 0 and list(b.columns) == ["patient", "feature", "center", "scale", "n_days", "weak"]
    dev = compute_deviations(d, b, cfg)
    assert len(dev) == 0 and list(dev.columns) == DEV_COLUMNS


def test_all_groups_present(daily_df):
    cfg = AnalysisConfig()
    b = compute_baselines(daily_df, empty_biopsies(), cfg)
    for g in FEATURE_GROUPS:
        assert _row(b, "P1", f"{g}_dev")["center"] == 0
        assert _row(b, "P1", f"{g}_dev")["scale"] >= 0.02



def test_baseline_default_uses_all_stable_days(daily_df):
    """Standard (baseline_days=0): alle stabilen Tage zaehlen, nicht nur die ersten."""
    cfg = AnalysisConfig()
    assert cfg.baseline_days == 0
    b = compute_baselines(daily_df, empty_biopsies(), cfg)
    p1 = _row(b, "P1", "R_rate_mean")
    assert p1["n_days"] == 30
    assert p1["center"] == pytest.approx(daily_df[daily_df.patient == "P1"]["R_rate_mean"].median())


def test_baseline_default_ignores_biopsy_labels(daily_df):
    """Standard: die Normallinie ist mit und ohne Befunde identisch (kennt die Etiketten nicht)."""
    cfg = AnalysisConfig()
    with_labels = compute_baselines(daily_df, make_biopsies([("P1", 1, "2025-01-10", 2)]), cfg)
    without = compute_baselines(daily_df, empty_biopsies(), cfg)
    pd.testing.assert_frame_equal(with_labels, without)
