import warnings

import numpy as np
import pandas as pd
import pytest

from kardio.config import AnalysisConfig
from kardio.constants import WINDOW_FEATURE_COLUMNS
from kardio.model.risk import ModelReport, fit_risk_model, score_timeline, traffic_light
from tests.conftest import make_windows

COL = "R_rate_mean__zmean"


@pytest.fixture(scope="module")
def fitted():
    cfg = AnalysisConfig()
    w = make_windows(25, 75, shift_col=COL, shift=2.5)
    rep, model = fit_risk_model(w, cfg)
    return w, rep, model, cfg


def _rolling(w: pd.DataFrame, n_data_days) -> pd.DataFrame:
    r = w[WINDOW_FEATURE_COLUMNS].copy()
    r.insert(0, "n_data_days", n_data_days)
    r.insert(0, "date", pd.Timestamp("2025-01-01"))
    r.insert(0, "patient", w["patient"].to_numpy())
    return r


def test_insufficient_few_windows():
    rep, model = fit_risk_model(make_windows(2, 12), AnalysisConfig())
    assert isinstance(rep, ModelReport)
    assert rep.status == "insufficient" and model is None
    assert rep.n_windows == 14 and rep.n_rejections == 2 and rep.n_patients == 14
    for part in ["14", "2", "20", "8", "Zu wenig Daten"]:
        assert part in rep.reason
    assert rep.auc_cv is None and rep.weights is None


def test_insufficient_few_positives():
    rep, model = fit_risk_model(make_windows(3, 40), AnalysisConfig())
    assert rep.status == "insufficient" and model is None
    assert "43" in rep.reason and "3 Abstoßungen" in rep.reason and "8" in rep.reason


def test_insufficient_few_negatives():
    rep, model = fit_risk_model(make_windows(40, 3), AnalysisConfig())
    assert rep.status == "insufficient" and model is None
    assert "3" in rep.reason and "8" in rep.reason and "ohne Abstoßung" in rep.reason


def test_insufficient_single_patient():
    w = make_windows(25, 75, shift_col=COL, shift=2.5)
    w["patient"] = "P0"
    rep, model = fit_risk_model(w, AnalysisConfig())
    assert rep.status == "insufficient" and model is None and rep.n_patients == 1
    assert "1 Patient" in rep.reason and "3" in rep.reason


def test_insufficient_empty_and_no_rejections():
    for w in [make_windows(0, 0), make_windows(0, 50), make_windows(50, 0)]:
        rep, model = fit_risk_model(w, AnalysisConfig())
        assert rep.status == "insufficient" and model is None and rep.reason


def test_all_positives_in_one_patient_is_insufficient():
    w = make_windows(25, 75, shift_col=COL, shift=2.5)
    w.loc[w["grade"] == 2, "patient"] = "PX"
    rep, model = fit_risk_model(w, AnalysisConfig())
    assert rep.status == "insufficient" and model is None and "Patient" in rep.reason


def test_learns_pattern(fitted):
    _, rep, model, cfg = fitted
    assert rep.status == "ok" and rep.reason == "" and model is not None
    assert rep.n_windows == 100 and rep.n_rejections == 25 and rep.n_patients == 100
    assert rep.auc_cv > 0.8
    assert rep.weights.iloc[0]["column"] == COL and rep.weights.iloc[0]["weight"] > 0
    assert list(rep.weights.columns) == ["column", "weight"] and len(rep.weights) == 132
    assert rep.weights["weight"].abs().is_monotonic_decreasing
    lo, hi = rep.auc_ci
    assert 0 <= lo <= rep.auc_cv <= hi <= 1 and lo < hi
    assert 0 <= rep.sensitivity <= 1 and 0 <= rep.specificity <= 1
    assert rep.threshold == cfg.red_threshold
    assert rep.auc_gb is not None and rep.auc_gb > 0.7


def test_null_auc_near_half():
    rep, _ = fit_risk_model(make_windows(25, 75), AnalysisConfig())
    assert rep.status == "ok" and 0.3 < rep.auc_cv < 0.7


def test_grade_1_excluded_from_training():
    cfg = AnalysisConfig()
    w = make_windows(25, 75, n_mid=30, shift_col=COL, shift=2.5)
    rep, model = fit_risk_model(w, cfg)
    assert rep.status == "ok" and rep.n_windows == 100 and rep.n_rejections == 25
    assert rep.n_patients == 100
    rep2, _ = fit_risk_model(w[w["grade"] != 1], cfg)
    assert rep2.auc_cv == rep.auc_cv
    pd.testing.assert_frame_equal(rep.weights, rep2.weights)
    # Zwischengruppe allein reicht nicht fuer ein Modell
    rep3, model3 = fit_risk_model(make_windows(3, 10, n_mid=40), cfg)
    assert rep3.status == "insufficient" and rep3.n_windows == 13 and model3 is None


def test_unusable_rows_excluded():
    w = make_windows(25, 75, shift_col=COL, shift=2.5)
    extra = make_windows(20, 20, seed=3)
    extra["usable"] = False
    extra["patient"] = "X" + extra["patient"]
    rep, _ = fit_risk_model(pd.concat([w, extra], ignore_index=True), AnalysisConfig())
    assert rep.n_windows == 100 and rep.n_rejections == 25


def test_missing_values_and_empty_columns_ok():
    w = make_windows(25, 75, shift_col=COL, shift=2.5)
    w.loc[w.index[::7], "T_amplitude_mean__slope"] = np.nan
    w["P_duration_cv__zmaxabs"] = np.nan
    w["PR_interval_dev__alertfrac"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rep, model = fit_risk_model(w, AnalysisConfig())
        t = score_timeline(model, _rolling(w, 10), AnalysisConfig())
    assert rep.status == "ok" and rep.auc_cv > 0.8 and len(rep.weights) == 132
    by = rep.weights.set_index("column")["weight"]
    assert by["P_duration_cv__zmaxabs"] == 0.0 and by["PR_interval_dev__alertfrac"] == 0.0
    assert t["risk"].between(0, 1).all()


def test_few_patients_grouped_cv():
    w = make_windows(25, 75, shift_col=COL, shift=2.5)
    w["patient"] = [f"P{i % 4}" for i in range(len(w))]
    rep, model = fit_risk_model(w, AnalysisConfig())
    assert rep.status == "ok" and rep.n_patients == 4 and rep.auc_cv > 0.7


def test_deterministic():
    cfg = AnalysisConfig()
    w = make_windows(25, 75, n_mid=5, shift_col=COL, shift=1.0)
    a, ma = fit_risk_model(w, cfg)
    b, mb = fit_risk_model(w.copy(), cfg)
    assert a.auc_cv == b.auc_cv and a.auc_ci == b.auc_ci and a.auc_gb == b.auc_gb
    assert a.sensitivity == b.sensitivity and a.specificity == b.specificity
    pd.testing.assert_frame_equal(a.weights, b.weights)
    r = _rolling(w, 10)
    pd.testing.assert_frame_equal(score_timeline(ma, r, cfg), score_timeline(mb, r, cfg))


def test_score_timeline(fitted):
    w, _, model, cfg = fitted
    rolling = _rolling(w, [1, 2] + [10] * (len(w) - 2))
    t = score_timeline(model, rolling, cfg)
    assert list(t.columns) == ["patient", "date", "risk"] and len(t) == len(rolling)
    assert t["risk"].iloc[:2].isna().all() and t["risk"].iloc[2:].notna().all()
    assert t["risk"].dropna().between(0, 1).all()
    assert list(t["patient"]) == list(rolling["patient"])
    # Abstossungsfenster bekommen im Mittel ein hoeheres Risiko
    rej = w["grade"].to_numpy() == 2
    assert t["risk"][rej].mean() > t["risk"][~rej].mean() + 0.3


def test_score_timeline_scores_grade_1_and_keeps_index(fitted):
    _, _, model, cfg = fitted
    w = make_windows(5, 5, n_mid=5, shift_col=COL, shift=2.5, seed=9)
    rolling = _rolling(w, 10)
    rolling.index = range(100, 100 + len(rolling))
    t = score_timeline(model, rolling, cfg)
    assert t["risk"].notna().all() and list(t.index) == list(rolling.index)


def test_score_timeline_without_model_and_empty(fitted):
    w, _, model, cfg = fitted
    t = score_timeline(None, _rolling(w, 10), cfg)
    assert len(t) == len(w) and t["risk"].isna().all()
    for m in [model, None]:
        e = score_timeline(m, _rolling(w, 10).iloc[0:0], cfg)
        assert list(e.columns) == ["patient", "date", "risk"] and len(e) == 0
    e = score_timeline(model, pd.DataFrame(), cfg)
    assert list(e.columns) == ["patient", "date", "risk"] and len(e) == 0


def test_score_timeline_all_below_minimum(fitted):
    w, _, model, cfg = fitted
    t = score_timeline(model, _rolling(w, 0), cfg)
    assert t["risk"].isna().all()


def test_traffic_light():
    cfg = AnalysisConfig()
    assert traffic_light(None, cfg) == "grey" and traffic_light(np.nan, cfg) == "grey"
    assert traffic_light(float("nan"), cfg) == "grey"
    assert traffic_light(0.0, cfg) == "green" and traffic_light(0.29, cfg) == "green"
    assert traffic_light(0.3, cfg) == "amber" and traffic_light(0.59, cfg) == "amber"
    assert traffic_light(0.6, cfg) == "red" and traffic_light(1.0, cfg) == "red"
    assert traffic_light(np.float64(0.4), cfg) == "amber"
    assert traffic_light(0.4, AnalysisConfig(amber_threshold=0.5)) == "green"


def test_score_timeline_cv_never_scores_a_patient_with_a_model_that_saw_them():
    """Je Patient ein Modell, das ihn nicht kennt: Ergebnis weicht vom Endmodell ab und bleibt in [0, 1]."""
    import numpy as np
    from kardio.config import AnalysisConfig
    from kardio.constants import WINDOW_FEATURE_COLUMNS
    from kardio.model.risk import fit_risk_model, score_timeline, score_timeline_cv
    from tests.conftest import make_windows
    cfg = AnalysisConfig()
    w = make_windows(25, 75, shift_col="R_rate_mean__zmean", shift=2.5)
    report, model = fit_risk_model(w, cfg)
    rolling = w[["patient", "date", "n_data_days"] + WINDOW_FEATURE_COLUMNS].copy()
    honest = score_timeline_cv(w, rolling, cfg, model)
    in_sample = score_timeline(model, rolling, cfg)
    assert honest["risk"].between(0, 1).all() and len(honest) == len(rolling)
    assert not np.allclose(honest["risk"], in_sample["risk"])
    # in den Lerndaten sieht das Endmodell besser aus als die ehrliche Bewertung
    from sklearn.metrics import roc_auc_score
    y = w["rejection"].astype(int)
    assert roc_auc_score(y, in_sample["risk"]) >= roc_auc_score(y, honest["risk"])
    assert abs(roc_auc_score(y, honest["risk"]) - report.auc_cv) < 1e-9  # identisch zur Kreuzvalidierung
    assert score_timeline_cv(w, rolling, cfg, None)["risk"].isna().all()
