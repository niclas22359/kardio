"""Verkettet alle Stationen: zwei Tabellen rein, ein Analyseergebnis raus."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from kardio.config import AnalysisConfig, load_analysis_config
from kardio.features.baseline import compute_baselines, compute_deviations
from kardio.features.windows import build_windows, rolling_windows
from kardio.ingest.biopsies import load_biopsies
from kardio.ingest.ecg import load_ecg, select_daily
from kardio.ingest.report import QualityReport, build_report
from kardio.model.risk import ModelReport, fit_risk_model, score_timeline_cv
from kardio.stats.compare import compare_features


@dataclass
class AnalysisResult:
    daily: pd.DataFrame          # eine Zeile pro Patient und Tag
    dropped: pd.DataFrame        # verworfene Mehrfachmessungen
    biopsies: pd.DataFrame       # patient, biopsy_no, date, grade, rejection
    report: QualityReport
    baselines: pd.DataFrame      # persoenliche Normallinie je Patient und Merkmal
    deviations: pd.DataFrame     # Langformat: patient, date, feature, value, z, ...
    windows: pd.DataFrame        # eine Zeile pro Biopsie (14 Tage davor verdichtet)
    rolling: pd.DataFrame        # dasselbe fuer jeden Messtag
    ranking: pd.DataFrame        # Merkmals-Rangliste (Station 4)
    ranking_reliable: bool
    model_report: ModelReport
    timeline: pd.DataFrame       # patient, date, risk
    data_source_label: str
    cfg: AnalysisConfig
    model: object = None         # das gelernte Modell (None bei zu wenig Daten); fuer die Pruefung an einem 2. Datensatz

    @property
    def is_demo(self) -> bool:
        return "demo" in self.data_source_label.lower()


def run_analysis_frames(ecg_raw: pd.DataFrame, biopsies: pd.DataFrame,
                        cfg: AnalysisConfig | None = None,
                        data_source_label: str = "Echte Daten") -> AnalysisResult:
    """ecg_raw: Ausgabe von load_ecg; biopsies: Ausgabe von load_biopsies."""
    cfg = cfg or load_analysis_config()
    daily, dropped = select_daily(ecg_raw)
    report = build_report(daily, dropped, biopsies)
    baselines = compute_baselines(daily, biopsies, cfg)
    deviations = compute_deviations(daily, baselines, cfg)
    windows = build_windows(deviations, biopsies, cfg)
    rolling = rolling_windows(deviations, cfg)
    ranking, reliable = compare_features(windows, cfg)
    model_report, model = fit_risk_model(windows, cfg)
    timeline = score_timeline_cv(windows, rolling, cfg, model)  # je Patient ein Modell, das ihn nicht kennt
    return AnalysisResult(daily, dropped, biopsies, report, baselines, deviations, windows, rolling,
                          ranking, reliable, model_report, timeline, data_source_label, cfg, model)


def run_analysis(ecg_src, biopsies_src, cfg: AnalysisConfig | None = None,
                 mapping: dict | None = None,
                 data_source_label: str = "Echte Daten") -> AnalysisResult:
    """Pfad oder Datei-Objekt fuer beide Tabellen."""
    return run_analysis_frames(load_ecg(ecg_src), load_biopsies(biopsies_src, mapping),
                               cfg, data_source_label)


def save_processed(result: AnalysisResult, out_dir) -> Path:
    """Alle Zwischenergebnisse als CSV ablegen, damit man sie anschauen kann."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {"daily": "ecg_daily", "biopsies": "biopsies", "baselines": "baselines",
             "deviations": "deviations", "windows": "windows", "ranking": "feature_ranking",
             "timeline": "risk_timeline"}
    for attr, fname in names.items():
        getattr(result, attr).to_csv(out / f"{fname}.csv", index=False)
    m = result.model_report
    report = {"status": m.status, "reason": m.reason, "n_windows": m.n_windows,
              "n_rejections": m.n_rejections, "n_patients": m.n_patients, "auc_cv": m.auc_cv,
              "auc_ci": list(m.auc_ci) if m.auc_ci else None, "sensitivity": m.sensitivity,
              "specificity": m.specificity, "threshold": m.threshold, "auc_gb": m.auc_gb,
              "weights": m.weights.to_dict("records") if m.weights is not None else None,
              "data_source": result.data_source_label}
    import json
    (out / "model_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return out
