"""Station 5: kleines, erklaerbares Risikomodell.

Lernt, welche Kombination von Abweichungen einer Abstossung vorausgeht.
Trainiert wird nur auf Stufe 0 (negativ) und Stufe 2 (positiv); die
Zwischengruppe (Stufe 1) bleibt beim Lernen aussen vor, wird aber bewertet.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kardio.config import AnalysisConfig
from kardio.constants import WINDOW_FEATURE_COLUMNS

MIN_PATIENTS = 3
N_BOOTSTRAP = 500
TIMELINE_COLUMNS = ["patient", "date", "risk"]


@dataclass
class ModelReport:
    status: str            # "ok" | "insufficient"
    reason: str            # German, empty when ok
    n_windows: int         # usable windows used for training (grade 0 and 2)
    n_rejections: int
    n_patients: int
    auc_cv: float | None = None
    auc_ci: tuple[float, float] | None = None   # 95% bootstrap CI of the cross-validated AUC
    sensitivity: float | None = None
    specificity: float | None = None
    threshold: float = 0.6
    weights: pd.DataFrame | None = None   # columns: column, weight (sorted by |weight| desc)
    auc_gb: float | None = None           # gradient boosting reference, cross-validated


def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


def _logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("imp", _imputer()),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", C=0.3, class_weight="balanced", max_iter=2000)),
    ])


def _boosting_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("imp", _imputer()),
        ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=seed)),
    ])


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    x = frame.reindex(columns=WINDOW_FEATURE_COLUMNS).apply(pd.to_numeric, errors="coerce")
    x = x.to_numpy(dtype=float)
    return np.where(np.isfinite(x), x, np.nan)


def _training_set(windows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = windows
    if "usable" in w.columns:
        w = w[w["usable"].eq(True).to_numpy()]
    if "grade" in w.columns:
        grade = pd.to_numeric(w["grade"], errors="coerce").to_numpy(dtype=float)
    elif "rejection" in w.columns:
        grade = np.where(w["rejection"].eq(True).to_numpy(), 2.0, 0.0)
    else:
        grade = np.full(len(w), np.nan)
    keep = (grade == 0) | (grade == 2)
    w = w[keep]
    y = (grade[keep] == 2).astype(int)
    groups = (w["patient"].astype(str).to_numpy() if "patient" in w.columns
              else np.array([], dtype=str))
    return _feature_matrix(w), y, groups


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _insufficient_reason(n: int, n_pos: int, n_neg: int, n_pat: int, cfg: AnalysisConfig) -> str:
    head = "Zu wenig Daten für ein Risikomodell: "
    if n < cfg.min_windows_total or n_pos < cfg.min_rejection_windows:
        return (f"{head}{n} Fenster, davon {_plural(n_pos, 'Abstoßung', 'Abstoßungen')} "
                f"(nötig: mindestens {cfg.min_windows_total} Fenster und "
                f"{cfg.min_rejection_windows} Abstoßungen).")
    if n_neg < cfg.min_rejection_windows:
        return (f"{head}{n} Fenster, davon nur {n_neg} ohne Abstoßung "
                f"(nötig: mindestens {cfg.min_rejection_windows} Fenster ohne Abstoßung).")
    if n_pat < MIN_PATIENTS:
        return (f"{head}{n} Fenster von nur {_plural(n_pat, 'Patient', 'Patienten')} "
                f"(nötig: mindestens {MIN_PATIENTS} Patienten).")
    return ""


def _out_of_fold(make_model, x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 n_splits: int, seed: int) -> np.ndarray | None:
    """Out-of-fold probabilities with patients kept together; None if a training fold has one class."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.full(len(y), np.nan)
    for train, test in cv.split(x, y, groups):
        if len(np.unique(y[train])) < 2:
            return None
        model = make_model().fit(x[train], y[train])
        proba[test] = model.predict_proba(x[test])[:, 1]
    return None if np.isnan(proba).any() else proba


def _bootstrap_ci(y: np.ndarray, proba: np.ndarray, groups: np.ndarray,
                  seed: int) -> tuple[float, float] | None:
    """95-%-Intervall der AUC. Gezogen werden ganze Patienten, nicht einzelne Fenster:
    Fenster desselben Patienten haengen zusammen, sonst wuerde das Intervall zu schmal."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if 0 < y[idx].sum() < len(idx):
            aucs.append(roc_auc_score(y[idx], proba[idx]))
    if not aucs:
        return None
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def _reference_auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int,
                   seed: int) -> float | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = _out_of_fold(lambda: _boosting_pipeline(seed), x, y, groups, n_splits, seed)
        return None if proba is None else float(roc_auc_score(y, proba))
    except Exception:  # reference only, must never take the main model down
        return None


def fit_risk_model(windows: pd.DataFrame, cfg: AnalysisConfig) -> tuple[ModelReport, Pipeline | None]:
    x, y, groups = _training_set(windows)
    n, n_pos = len(y), int(y.sum())
    n_neg, n_pat = n - n_pos, int(len(np.unique(groups)))

    def insufficient(reason: str) -> tuple[ModelReport, None]:
        return ModelReport("insufficient", reason, n, n_pos, n_pat,
                           threshold=cfg.red_threshold), None

    reason = _insufficient_reason(n, n_pos, n_neg, n_pat, cfg)
    if reason:
        return insufficient(reason)

    n_splits = min(5, n_pat, n_pos)
    proba = _out_of_fold(_logistic_pipeline, x, y, groups, n_splits, cfg.seed)
    if proba is None:
        n_pat_pos = int(len(np.unique(groups[y == 1])))
        n_pat_neg = int(len(np.unique(groups[y == 0])))
        return insufficient(
            "Zu wenig Daten für ein Risikomodell: Die Fenster verteilen sich auf zu wenige "
            f"Patienten für eine ehrliche Prüfung ({_plural(n_pat_pos, 'Patient', 'Patienten')} "
            f"mit Abstoßung, {n_pat_neg} ohne; nötig: jeweils mindestens 2).")

    pred = proba >= cfg.red_threshold
    model = _logistic_pipeline().fit(x, y)
    weights = pd.DataFrame({"column": WINDOW_FEATURE_COLUMNS,
                            "weight": model.named_steps["lr"].coef_[0]})
    order = weights["weight"].abs().sort_values(ascending=False, kind="stable").index
    weights = weights.loc[order].reset_index(drop=True)

    report = ModelReport(
        status="ok", reason="", n_windows=n, n_rejections=n_pos, n_patients=n_pat,
        auc_cv=float(roc_auc_score(y, proba)),
        auc_ci=_bootstrap_ci(y, proba, groups, cfg.seed),
        sensitivity=float(pred[y == 1].mean()),
        specificity=float((~pred[y == 0]).mean()),
        threshold=cfg.red_threshold,
        weights=weights,
        auc_gb=_reference_auc(x, y, groups, n_splits, cfg.seed),
    )
    return report, model


def score_timeline(model: Pipeline | None, rolling: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    """Risk per patient and day; NaN without a model or with too few data days in the window."""
    if len(rolling) == 0 or not {"patient", "date"} <= set(rolling.columns):
        out = rolling.reindex(columns=["patient", "date"]).iloc[0:0].copy()
        out["risk"] = pd.Series(dtype=float)
        return out[TIMELINE_COLUMNS]
    out = rolling[["patient", "date"]].copy()
    out["risk"] = np.nan
    if model is None or "n_data_days" not in rolling.columns:
        return out
    days = pd.to_numeric(rolling["n_data_days"], errors="coerce").to_numpy(dtype=float)
    ok = days >= cfg.min_window_data_days
    if ok.any():
        risk = np.full(len(rolling), np.nan)
        risk[ok] = model.predict_proba(_feature_matrix(rolling[ok]))[:, 1]
        out["risk"] = np.clip(risk, 0.0, 1.0)
    return out


def score_timeline_cv(windows: pd.DataFrame, rolling: pd.DataFrame, cfg: AnalysisConfig,
                      model: Pipeline | None) -> pd.DataFrame:
    """Ehrliche Risikokurve: jeder Patient wird von einem Modell bewertet, das ihn nie gesehen hat.

    Das Endmodell hat die Fenster aller Patienten gelernt; damit die eigenen Verlaeufe zu bewerten
    saehe zu gut aus. Deshalb dieselben Patienten-Gruppen wie in der Kreuzvalidierung: je Gruppe ein
    Modell aus den uebrigen Patienten. Patienten ohne Lernfenster bewertet das Endmodell.
    """
    out = score_timeline(model, rolling, cfg)
    if model is None or len(out) == 0:
        return out
    x, y, groups = _training_set(windows)
    n_splits = min(5, int(len(np.unique(groups))), int(y.sum()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    patients = rolling["patient"].to_numpy()
    days = pd.to_numeric(rolling["n_data_days"], errors="coerce").to_numpy(dtype=float)
    ok = days >= cfg.min_window_data_days
    risk = out["risk"].to_numpy(dtype=float).copy()
    for train, test in cv.split(x, y, groups):
        rows = ok & np.isin(patients, np.unique(groups[test]))
        if not rows.any() or len(np.unique(y[train])) < 2:
            continue
        fold_model = _logistic_pipeline().fit(x[train], y[train])
        risk[rows] = fold_model.predict_proba(_feature_matrix(rolling[rows]))[:, 1]
    out["risk"] = np.clip(risk, 0.0, 1.0)
    return out


def traffic_light(risk: float | None, cfg: AnalysisConfig) -> str:
    if risk is None or pd.isna(risk):
        return "grey"
    if risk >= cfg.red_threshold:
        return "red"
    if risk >= cfg.amber_threshold:
        return "amber"
    return "green"
