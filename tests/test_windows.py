import time

import numpy as np
import pandas as pd
import pytest

from kardio.config import AnalysisConfig
from kardio.constants import DEV_FEATURES, VALUE_COLUMNS, WINDOW_FEATURE_COLUMNS
from kardio.features.baseline import compute_baselines, compute_deviations
from kardio.features.windows import build_windows, rolling_windows
from tests.conftest import empty_biopsies, make_biopsies

META = ["patient", "biopsy_no", "date", "grade", "rejection", "n_data_days", "under_treatment", "usable"]


def _dev(daily, bx, cfg=None):
    cfg = cfg or AnalysisConfig()
    return compute_deviations(daily, compute_baselines(daily, bx, cfg), cfg), cfg


def _hand_dev(z_values, dates=None, patient="H1", feature="R_rate_mean") -> pd.DataFrame:
    """Deviations table by hand: `feature` gets z_values, every other feature z = 0."""
    if dates is None:
        dates = pd.date_range("2025-01-01", periods=len(z_values))
    rows = []
    for date, z in zip(pd.to_datetime(dates), z_values):
        for f in DEV_FEATURES:
            zz = z if f == feature else 0.0
            rows.append({"patient": patient, "date": date, "feature": f, "value": zz, "z": zz,
                         "delta_prev": np.nan, "trend_7d": np.nan})
    return pd.DataFrame(rows)


def test_window_basic(daily_df):
    bx = make_biopsies([("P1", 1, "2025-01-20", 0), ("P1", 2, "2025-01-28", 2)])
    dev, cfg = _dev(daily_df, bx)
    w = build_windows(dev, bx, cfg)
    assert len(WINDOW_FEATURE_COLUMNS) == 132
    assert list(w.columns) == META + WINDOW_FEATURE_COLUMNS
    assert len(w) == 2
    assert w.iloc[0]["n_data_days"] == 15 and w.iloc[0]["usable"]
    assert w["biopsy_no"].tolist() == [1, 2] and w["grade"].tolist() == [0, 2]
    assert w["rejection"].tolist() == [False, True]
    assert np.isfinite(w.iloc[0]["R_rate_mean__zmean"])
    assert w["usable"].dtype == bool and w["under_treatment"].dtype == bool


def test_window_aggregates_hand_made():
    cfg = AnalysisConfig()
    dev = _hand_dev(np.arange(15, dtype=float))
    w = build_windows(dev, make_biopsies([("H1", 1, "2025-01-15", 0)]), cfg).iloc[0]
    assert w["n_data_days"] == 15
    assert w["R_rate_mean__zmean"] == pytest.approx(7.0)
    assert w["R_rate_mean__zmaxabs"] == pytest.approx(14.0)
    assert w["R_rate_mean__slope"] == pytest.approx(1.0)
    assert w["R_rate_mean__alertfrac"] == pytest.approx(12 / 15)  # z = 3..14 above 2.0
    assert w["R_rate_cv__slope"] == pytest.approx(0.0, abs=1e-12)
    assert w["R_rate_cv__alertfrac"] == 0

    falling = build_windows(_hand_dev(-2 * np.arange(15, dtype=float)),
                            make_biopsies([("H1", 1, "2025-01-15", 0)]), cfg).iloc[0]
    assert falling["R_rate_mean__slope"] == pytest.approx(-2.0)
    assert falling["R_rate_mean__zmaxabs"] == pytest.approx(28.0)
    assert falling["R_rate_mean__zmean"] == pytest.approx(-14.0)


def test_window_bounds_are_inclusive():
    cfg = AnalysisConfig()
    z = np.zeros(20)
    z[0] = 50.0  # 2025-01-01, exactly window_days before the first biopsy -> inside
    dev = _hand_dev(z)
    w = build_windows(dev, make_biopsies([("H1", 1, "2025-01-15", 0), ("H1", 2, "2025-01-16", 0)]), cfg)
    assert w.iloc[0]["R_rate_mean__zmaxabs"] == 50.0
    assert w.iloc[1]["R_rate_mean__zmaxabs"] == 0.0


def test_slope_uses_day_offsets_and_needs_three_points():
    cfg = AnalysisConfig()
    dev = _hand_dev([0.0, 2.0, 4.0], dates=["2025-01-01", "2025-01-03", "2025-01-05"])
    w = build_windows(dev, make_biopsies([("H1", 1, "2025-01-05", 0)]), cfg).iloc[0]
    assert w["R_rate_mean__slope"] == pytest.approx(1.0)
    two = build_windows(dev, make_biopsies([("H1", 1, "2025-01-03", 0)]), cfg).iloc[0]
    assert np.isnan(two["R_rate_mean__slope"])
    assert two["R_rate_mean__zmean"] == pytest.approx(1.0) and two["n_data_days"] == 2


def test_nan_z_is_ignored_in_aggregates():
    cfg = AnalysisConfig()
    dev = _hand_dev([1.0, np.nan, 3.0, 5.0])
    dev.loc[dev.z.isna(), "value"] = np.nan
    w = build_windows(dev, make_biopsies([("H1", 1, "2025-01-04", 0)]), cfg).iloc[0]
    assert w["R_rate_mean__zmean"] == pytest.approx(3.0)
    assert w["R_rate_mean__alertfrac"] == pytest.approx(2 / 3)
    assert w["n_data_days"] == 4  # the other features still have data on day 2


def test_window_before_first_measurement(daily_df):
    bx = make_biopsies([("P1", 1, "2024-12-01", 0), ("ZZ", 1, "2025-01-15", 2)])
    dev, cfg = _dev(daily_df, bx)
    w = build_windows(dev, bx, cfg)
    assert len(w) == 2
    for i in range(2):
        assert w.iloc[i]["n_data_days"] == 0 and not w.iloc[i]["usable"]
        assert w.iloc[i][WINDOW_FEATURE_COLUMNS].isna().all()
    assert w["n_data_days"].dtype.kind == "i"


def test_biopsy_on_day_without_measurement(daily_df):
    # P2 has no measurement on 2025-01-12; the data of P1 end on 2025-01-30
    bx = make_biopsies([("P2", 1, "2025-01-12", 0), ("P1", 1, "2025-02-05", 0), ("P1", 2, "2025-03-30", 0)])
    dev, cfg = _dev(daily_df, bx)
    w = build_windows(dev, bx, cfg).set_index(["patient", "biopsy_no"])
    assert w.loc[("P2", 1), "n_data_days"] == 10  # window 2024-12-29..2025-01-12 holds Jan 1..10
    assert np.isfinite(w.loc[("P2", 1), "R_rate_mean__zmean"])
    assert w.loc[("P1", 1), "n_data_days"] == 9  # Jan 22..30
    assert w.loc[("P1", 2), "n_data_days"] == 0 and not w.loc[("P1", 2), "usable"]


def test_under_treatment_flag(daily_df):
    bx = make_biopsies([("P1", 1, "2025-01-10", 2), ("P1", 2, "2025-01-20", 0),
                        ("P1", 3, "2025-02-14", 0), ("P1", 4, "2025-02-15", 0),
                        ("P2", 1, "2025-01-20", 0)])
    dev, cfg = _dev(daily_df, bx)
    w = build_windows(dev, bx, cfg).set_index(["patient", "biopsy_no"])
    assert not w.loc[("P1", 1), "under_treatment"]
    assert w.loc[("P1", 2), "under_treatment"] and not w.loc[("P1", 2), "usable"]
    assert w.loc[("P1", 3), "under_treatment"]  # exactly 35 days later
    assert not w.loc[("P1", 4), "under_treatment"]  # 36 days later
    assert not w.loc[("P2", 1), "under_treatment"]  # other patient
    assert w.loc[("P1", 1), "usable"]


def test_gap_handling(daily_df):
    bx = make_biopsies([("P2", 1, "2025-01-20", 0), ("P3", 1, "2025-01-02", 0)])
    dev, cfg = _dev(daily_df, bx)
    w = build_windows(dev, bx, cfg).set_index("patient")
    assert w.loc["P2", "n_data_days"] == 12 and w.loc["P2", "usable"]
    assert w.loc["P3", "n_data_days"] == 2 and not w.loc["P3", "usable"]  # below min_window_data_days


def test_all_nan_day_is_not_a_data_day(daily_df):
    d = daily_df.copy()
    d.loc[(d.patient == "P1") & (d.date == "2025-01-15"), VALUE_COLUMNS] = np.nan
    bx = make_biopsies([("P1", 1, "2025-01-20", 0)])
    dev, cfg = _dev(d, bx)
    assert build_windows(dev, bx, cfg).iloc[0]["n_data_days"] == 14
    r = rolling_windows(dev, cfg)
    assert r[(r.patient == "P1") & (r.date == "2025-01-20")].iloc[0]["n_data_days"] == 14


def test_rolling(daily_df):
    dev, cfg = _dev(daily_df, empty_biopsies())
    r = rolling_windows(dev, cfg)
    assert list(r.columns) == ["patient", "date", "n_data_days"] + WINDOW_FEATURE_COLUMNS
    assert set(r.patient) == {"P1", "P2", "P3"}
    assert r[r.patient == "P1"]["n_data_days"].max() == 15
    assert r[r.patient == "P1"]["n_data_days"].tolist()[:3] == [1, 2, 3]
    # exactly the measured days, nothing from the filled calendar
    got = r[["patient", "date"]].reset_index(drop=True)
    want = daily_df[["patient", "date"]].sort_values(["patient", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
    assert len(r) == 30 + 27 + 2
    assert not np.isinf(r[WINDOW_FEATURE_COLUMNS].to_numpy()).any()


def test_build_matches_rolling(daily_df):
    bx = make_biopsies([("P1", 1, "2025-01-20", 0), ("P2", 1, "2025-01-25", 1), ("P3", 1, "2025-01-02", 0)])
    dev, cfg = _dev(daily_df, bx)
    w = build_windows(dev, bx, cfg)
    r = rolling_windows(dev, cfg)
    for _, b in w.iterrows():
        rr = r[(r.patient == b["patient"]) & (r.date == b["date"])]
        assert len(rr) == 1
        assert rr.iloc[0]["n_data_days"] == b["n_data_days"]
        np.testing.assert_allclose(rr.iloc[0][WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float),
                                   b[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float), equal_nan=True)


def test_rolling_matches_naive_reference(daily_df):
    """Vectorized result equals a plain per-window computation."""
    dev, cfg = _dev(daily_df, empty_biopsies())
    r = rolling_windows(dev, cfg)
    f = "T_amplitude_cv"
    for pid, day in [("P2", "2025-01-16"), ("P1", "2025-01-30"), ("P3", "2025-01-02")]:
        day = pd.Timestamp(day)
        win = dev[(dev.patient == pid) & (dev.feature == f) & (dev.date <= day)
                  & (dev.date >= day - pd.Timedelta(days=cfg.window_days))].dropna(subset=["z"])
        x = (win.date - day).dt.days.to_numpy(dtype=float)
        z = win.z.to_numpy()
        row = r[(r.patient == pid) & (r.date == day)].iloc[0]
        assert row[f"{f}__zmean"] == pytest.approx(z.mean())
        assert row[f"{f}__zmaxabs"] == pytest.approx(np.abs(z).max())
        assert row[f"{f}__alertfrac"] == pytest.approx((np.abs(z) > cfg.z_alert).mean())
        if len(z) >= 3:
            assert row[f"{f}__slope"] == pytest.approx(np.polyfit(x, z, 1)[0], abs=1e-9)
        else:
            assert np.isnan(row[f"{f}__slope"])


def test_empty_inputs(daily_df):
    dev, cfg = _dev(daily_df, empty_biopsies())
    w = build_windows(dev, empty_biopsies(), cfg)
    assert len(w) == 0 and list(w.columns) == META + WINDOW_FEATURE_COLUMNS
    empty_dev = dev.iloc[0:0]
    r = rolling_windows(empty_dev, cfg)
    assert len(r) == 0 and list(r.columns) == ["patient", "date", "n_data_days"] + WINDOW_FEATURE_COLUMNS
    w2 = build_windows(empty_dev, make_biopsies([("P1", 1, "2025-01-20", 2)]), cfg)
    assert len(w2) == 1 and w2.iloc[0]["n_data_days"] == 0 and not w2.iloc[0]["usable"]


def _synthetic(n_patients=45, n_days=365, seed=0):
    rng = np.random.default_rng(seed)
    n = n_patients * n_days
    base = rng.uniform(50, 500, size=len(VALUE_COLUMNS))
    vals = base * (1 + rng.normal(0, 0.05, size=(n, len(VALUE_COLUMNS))))
    vals[rng.random(n) < 0.1] = np.nan  # days without data
    d = pd.DataFrame(vals, columns=VALUE_COLUMNS)
    d.insert(0, "cc", 0.95)
    d.insert(0, "included_no", 30)
    d.insert(0, "date", np.tile(pd.date_range("2025-01-01", periods=n_days), n_patients))
    d.insert(0, "patient", np.repeat([f"S{i:02d}" for i in range(n_patients)], n_days))
    keep = rng.random(n) > 0.05  # days missing altogether
    d = d[keep].reset_index(drop=True)
    rows = []
    for i in range(n_patients):
        for k, off in enumerate(sorted(rng.choice(np.arange(5, n_days + 10), size=20, replace=False))):
            rows.append((f"S{i:02d}", k + 1, pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(off)),
                         int(rng.choice([0, 0, 1, 2]))))
    return d, make_biopsies(rows)


def test_performance_realistic_size():
    daily, bx = _synthetic()
    cfg = AnalysisConfig()
    t0 = time.perf_counter()
    base = compute_baselines(daily, bx, cfg)
    dev = compute_deviations(daily, base, cfg)
    roll = rolling_windows(dev, cfg)
    t1 = time.perf_counter()
    win = build_windows(dev, bx, cfg)
    t2 = time.perf_counter()
    print(f"\nperf 45x365: baselines+deviations+rolling {t1 - t0:.2f}s, build_windows {t2 - t1:.2f}s")
    assert len(dev) == len(daily) * 33
    assert len(roll) == len(daily)
    assert len(win) == len(bx) == 900
    assert not np.isinf(dev["z"].to_numpy()).any()
    assert t1 - t0 < 15
    assert t2 - t1 < 6
