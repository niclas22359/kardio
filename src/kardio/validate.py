"""Zweiter Strang: ein bereits gelerntes Modell auf einen unabhaengigen Datensatz anwenden.

Im ersten Strang (pipeline.run_analysis) lernt das Modell seine Gewichte aus einem Datensatz.
Hier wird NICHTS neu gelernt: die Gewichte bleiben eingefroren. Der neue Datensatz durchlaeuft
nur Einlesen, persoenliche Normallinie und Fenster, dann bewertet das fertige Modell ihn.
Gibt es zum neuen Datensatz Biopsien, zeigt der Vergleich Vorhersage gegen Befund, ob das
Modell auch bei Patienten funktioniert, die es nie gesehen hat.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from kardio.config import AnalysisConfig
from kardio.constants import WINDOW_FEATURE_COLUMNS
from kardio.features.baseline import compute_baselines, compute_deviations
from kardio.features.windows import build_windows, rolling_windows
from kardio.ingest.ecg import select_daily
from kardio.ingest.report import QualityReport, build_report
from kardio.model.risk import N_BOOTSTRAP, _feature_matrix, score_timeline

MIN_PER_CLASS = 5   # so viele Fenster mit und ohne Abstossung braucht eine belastbare Guete


@dataclass
class ValidationMetrics:
    status: str                 # "ok" | "few" (zu wenige Befunde fuer eine Guete) | "unlabeled"
    note: str = ""
    n_windows: int = 0          # verwertbare Fenster mit Befund 0 oder 2
    n_rejections: int = 0
    n_patients: int = 0
    auc: float | None = None
    auc_ci: tuple[float, float] | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    threshold: float = 0.6
    tp: int = 0                 # Abstossung, Modell rot
    fn: int = 0                 # Abstossung, Modell nicht rot
    fp: int = 0                 # keine Abstossung, Modell rot
    tn: int = 0                 # keine Abstossung, Modell nicht rot
    roc: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["fpr", "tpr"]))
    train_auc: float | None = None   # zum Vergleich: Guete aus dem Training (Kreuzvalidierung)


@dataclass
class ValidationResult:
    daily: pd.DataFrame
    dropped: pd.DataFrame
    biopsies: pd.DataFrame
    report: QualityReport
    baselines: pd.DataFrame
    deviations: pd.DataFrame
    windows: pd.DataFrame        # eine Zeile pro Biopsie, mit Spalte "risk" aus dem eingefrorenen Modell
    rolling: pd.DataFrame
    timeline: pd.DataFrame       # patient, date, risk
    metrics: ValidationMetrics
    data_source_label: str
    trained_on: str              # Datenbasis, auf der das Modell gelernt wurde
    cfg: AnalysisConfig

    @property
    def is_demo(self) -> bool:
        return "demo" in self.data_source_label.lower()


def empty_biopsies() -> pd.DataFrame:
    return pd.DataFrame({"patient": pd.Series(dtype=str), "biopsy_no": pd.Series(dtype="int64"),
                         "date": pd.to_datetime([]), "grade": pd.Series(dtype="int64"),
                         "rejection": pd.Series(dtype=bool)})


def _patient_bootstrap_ci(y, proba, groups, seed) -> tuple[float, float] | None:
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


def evaluate_frozen(model, windows: pd.DataFrame, cfg: AnalysisConfig,
                    train_auc: float | None = None) -> tuple[pd.DataFrame, ValidationMetrics]:
    """Fenster mit dem eingefrorenen Modell bewerten und gegen die Befunde pruefen.

    Rueckgabe: (Fenster mit Spalte risk, Kennzahlen). Bewertet wird jedes Fenster mit genug Messtagen;
    in die Guete gehen nur verwertbare Fenster mit Befund 0 oder 2 ein (wie im Training).
    """
    w = windows.copy()
    w["risk"] = np.nan
    enough = pd.to_numeric(w["n_data_days"], errors="coerce").to_numpy(dtype=float) >= cfg.min_window_data_days
    if len(w) and enough.any():
        risk = np.full(len(w), np.nan)
        risk[enough] = model.predict_proba(_feature_matrix(w[enough]))[:, 1]
        w["risk"] = np.clip(risk, 0.0, 1.0)

    m = ValidationMetrics(status="unlabeled", threshold=cfg.red_threshold, train_auc=train_auc)
    if len(w) == 0:
        m.note = "Zu diesem Datensatz liegen keine Biopsien vor. Es gibt Risikowerte, aber keine Prüfung."
        return w, m
    scored = w[w["usable"].eq(True) & w["grade"].isin([0, 2]) & w["risk"].notna()]
    y = (scored["grade"] == 2).to_numpy(dtype=int)
    p = scored["risk"].to_numpy(dtype=float)
    groups = scored["patient"].astype(str).to_numpy()
    pred = p >= cfg.red_threshold
    m.n_windows, m.n_rejections, m.n_patients = len(y), int(y.sum()), int(len(np.unique(groups)))
    m.tp, m.fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    m.fp, m.tn = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
    n_pos, n_neg = m.n_rejections, m.n_windows - m.n_rejections
    if n_pos:
        m.sensitivity = m.tp / n_pos
    if n_neg:
        m.specificity = m.tn / n_neg
    if n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
        m.status = "few"
        m.note = (f"Für eine belastbare Prüfung sind es zu wenige Befunde: {n_pos} verwertbare Biopsien mit "
                  f"Abstoßung und {n_neg} ohne (nötig: jeweils mindestens {MIN_PER_CLASS}).")
        if n_pos and n_neg:
            m.auc = float(roc_auc_score(y, p))
        return w, m
    m.status = "ok"
    m.auc = float(roc_auc_score(y, p))
    m.auc_ci = _patient_bootstrap_ci(y, p, groups, cfg.seed)
    fpr, tpr, _ = roc_curve(y, p)
    m.roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    return w, m


def run_validation_frames(trained, ecg_raw: pd.DataFrame, biopsies: pd.DataFrame | None,
                          data_source_label: str = "Zweiter Datensatz") -> ValidationResult:
    """trained: AnalysisResult aus dem Training (mit Modell). ecg_raw: Ausgabe von load_ecg.
    biopsies: Ausgabe von load_biopsies oder None (dann nur Risikowerte, keine Pruefung)."""
    if getattr(trained, "model", None) is None:
        raise ValueError("Es gibt kein gelerntes Modell. Bitte zuerst einen Trainings-Datensatz analysieren, "
                         "der für ein Risikomodell ausreicht.")
    cfg = trained.cfg
    bx = biopsies if biopsies is not None else empty_biopsies()
    daily, dropped = select_daily(ecg_raw)
    report = build_report(daily, dropped, bx)
    baselines = compute_baselines(daily, bx, cfg)      # persoenliches Normal: je Patient aus seinen eigenen Tagen
    deviations = compute_deviations(daily, baselines, cfg)
    windows = build_windows(deviations, bx, cfg)
    rolling = rolling_windows(deviations, cfg)
    for col in WINDOW_FEATURE_COLUMNS:                 # leere Tabelle: Spalten sicherstellen
        if col not in windows.columns:
            windows[col] = np.nan
    windows, metrics = evaluate_frozen(trained.model, windows, cfg, trained.model_report.auc_cv)
    timeline = score_timeline(trained.model, rolling, cfg)   # eingefrorenes Modell, kein neues Lernen
    return ValidationResult(daily, dropped, bx, report, baselines, deviations, windows, rolling, timeline,
                            metrics, data_source_label, trained.data_source_label, cfg)


def lead_time_table(result: ValidationResult, lookback_days: int = 21) -> pd.DataFrame:
    """Je Abstossungs-Biopsie: wie viele Tage vorher stand die Ampel erstmals auf Rot?

    Spalten: patient, date, max_risk, first_red_days_before (NaN = nie rot im Rueckblick).
    """
    rows = []
    tl = result.timeline.dropna(subset=["risk"])
    for _, b in result.biopsies[result.biopsies["grade"] == 2].iterrows():
        seg = tl[(tl["patient"] == b["patient"]) & (tl["date"] <= b["date"])
                 & (tl["date"] >= b["date"] - pd.Timedelta(days=lookback_days))].sort_values("date")
        if len(seg) == 0:
            rows.append({"patient": b["patient"], "date": b["date"], "max_risk": np.nan,
                         "first_red_days_before": np.nan})
            continue
        # zusammenhaengende rote Phase, die bis zur Biopsie reicht: ihren Beginn suchen
        red = (seg["risk"] >= result.cfg.red_threshold).to_numpy()
        first = np.nan
        if red[-1]:
            i = len(red) - 1
            while i > 0 and red[i - 1]:
                i -= 1
            first = float((b["date"] - seg["date"].iloc[i]).days)
        rows.append({"patient": b["patient"], "date": b["date"], "max_risk": float(seg["risk"].max()),
                     "first_red_days_before": first})
    return pd.DataFrame(rows, columns=["patient", "date", "max_risk", "first_red_days_before"])
