from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AnalysisConfig:
    window_days: int = 14
    baseline_days: int = 0   # 0 = alle stabilen Tage; >0 = nur die ersten N
    baseline_exclude_rejection: bool = False  # True = Tage um Abstossungen aus der Normallinie nehmen
    baseline_min_days: int = 5
    min_window_data_days: int = 3
    treatment_days: int = 21
    min_windows_total: int = 20
    min_rejection_windows: int = 8
    min_rejection_windows_stats: int = 5
    amber_threshold: float = 0.3
    red_threshold: float = 0.6
    z_alert: float = 2.0
    seed: int = 42


@dataclass
class GeneratorConfig:
    n_patients: int = 40
    days: int = 365
    missing_day_rate: float = 0.1
    multi_measurement_rate: float = 0.15
    rejection_rate: float = 0.25
    borderline_rate: float = 0.15
    borderline_pattern_factor: float = 0.5
    pattern_enabled: bool = True
    pattern_days_before: int = 10
    recovery_days: int = 14
    pattern: dict = field(default_factory=dict)
    biopsy_schedule: list = field(default_factory=list)
    followup_biopsies_after_rejection: int = 3
    followup_interval_days: int = 7
    seed: int = 42


def _load(path, default_name: str) -> dict:
    p = Path(path) if path else ROOT / "config" / default_name
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_analysis_config(path=None) -> AnalysisConfig:
    return AnalysisConfig(**_load(path, "analysis.yaml"))


def load_generator_config(path=None) -> GeneratorConfig:
    return GeneratorConfig(**_load(path, "generator.yaml"))
