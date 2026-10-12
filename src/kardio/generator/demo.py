"""Demo data generator: files in exactly the real format with a switchable built-in pattern.

The clinic has a time course for a single patient only (MW) and no biopsy table yet. To prove
that the analysis works, this module produces demo patients whose value ranges come from MW's
real measurements and, optionally, a deliberately built-in ECG drift before rejections that the
analysis has to rediscover. With the pattern switched off the analysis must find nothing.

The pattern is applied as a pure multiplication after all random numbers have been drawn, so
switching it on or off never changes patients, dates, biopsies or any non-pattern value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kardio.config import ROOT, GeneratorConfig
from kardio.constants import FEATURE_GROUPS, VALUE_COLUMNS
from kardio.ingest.ecg import load_ecg

RAW_HEADER = ["Unnamed: 0", "Name", "Date", "included_no", "cc", "hr", "rr", "rr_int"] + VALUE_COLUMNS
BIOPSY_HEADER = ["Patient", "Biopsie_Nr", "Datum", "Befund"]
INT_COLUMNS = ["Unnamed: 0", "included_no", "hr", "rr", "rr_int"]
GRADE_TEXT = {0: "nein", 1: "Zwischengruppe", 2: "ja"}
T_BLOCK_GROUPS = ("T_amplitude", "T_duration", "RT_interval", "qRT_interval")

REFERENCE_PATIENT = "MW"          # only real patient with a time course
FIRST_TRANSPLANT = "2025-01-01"
TRANSPLANT_SPREAD_DAYS = 60
FIRST_BIOPSY_DAY = 7
AR_PHI = 0.4                      # day-to-day autocorrelation of the personal series
BETWEEN_PATIENT_SD = 1.5          # personal level offset, in units of the day-to-day sd
SAME_DAY_JITTER = 0.3             # difference between measurements of one day, same units
MIN_LEVEL_RATIO = 0.3             # a value never drops below 30 % of the personal level (keeps the sign)
EMPTY_ROW_RATE = 0.02
MISSING_T_BLOCK_RATE = 0.03
INCLUDED_RANGE = (14, 43)
CC_RANGE = (0.87, 0.97)


def _real_path(real_ecg_path) -> Path:
    return Path(real_ecg_path) if real_ecg_path else ROOT / "data" / "raw" / "ecg_real.csv"


def _consecutive_block(df: pd.DataFrame) -> pd.DataFrame:
    """Longest stretch of (almost) daily measurements; isolated dates years apart are left out."""
    df = df.sort_values("date")
    run_id = (df["date"].diff().dt.days > 2).cumsum()
    return df[run_id == run_id.value_counts().idxmax()]


REFERENCE_STATS_PATH = ROOT / "config" / "reference_stats.json"


def _load_reference_stats() -> dict[str, np.ndarray]:
    import json
    with open(REFERENCE_STATS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("feature_groups") != FEATURE_GROUPS:
        raise ValueError("config/reference_stats.json passt nicht zu den Merkmalsgruppen")
    out = {k: np.asarray(v, dtype=float) for k, v in raw["stats"].items()}
    out["positive_min"] = out["positive_min"].astype(bool)
    return out


def save_reference_stats(real_ecg_path=None) -> Path:
    """Aggregierte Kennwerte aus der echten Datei ableiten und ablegen (einmalig, lokal)."""
    import json
    stats = _population_stats(real_ecg_path)
    payload = {"comment": "Aggregierte Kennwerte (Mittel, Streuung) eines Referenzverlaufs. "
                          "Keine Einzelmessungen, keine Patientendaten.",
               "feature_groups": FEATURE_GROUPS,
               "stats": {k: [float(x) for x in np.asarray(v, dtype=float)] for k, v in stats.items()}}
    REFERENCE_STATS_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return REFERENCE_STATS_PATH


def _population_stats(real_ecg_path) -> dict[str, np.ndarray]:
    """Typical level and day-to-day spread per feature group from the reference patient.

    Only the consecutive block of days is used: the few isolated measurements from other years
    are no day-to-day variation and would inflate the spread.
    """
    path = _real_path(real_ecg_path)
    if not path.exists():
        # Online-Betrieb / frischer Checkout: echte Patientendaten liegen nur lokal vor.
        # Dann die mitgelieferten, rein aggregierten Kennwerte nutzen (keine Einzelmessungen).
        return _load_reference_stats()
    df = load_ecg(path)
    df = df[df["patient"] == REFERENCE_PATIENT].dropna(subset=VALUE_COLUMNS)
    block = _consecutive_block(df) if len(df) else df
    df = block if len(block) >= 5 else df
    if len(df) < 5:
        raise ValueError(f"Real ECG file has too few complete rows of patient {REFERENCE_PATIENT} "
                         "to derive value ranges")
    v = df[VALUE_COLUMNS].to_numpy(float).reshape(len(df), len(FEATURE_GROUPS), 4)
    mean, sd, lo, hi = v[:, :, 0], v[:, :, 1], v[:, :, 2], v[:, :, 3]
    mu = mean.mean(axis=0)
    day_sd = mean.std(axis=0, ddof=1)
    return {
        "mu": mu,
        "day_sd": np.where(day_sd > 0, day_sd, np.abs(mu) * 0.05),
        "sd_level": np.median(sd, axis=0),
        "sd_log_spread": np.clip(np.log(sd).std(axis=0, ddof=1), 0.15, 0.40),
        "k_min": np.clip(np.median((mean - lo) / sd, axis=0), 0.5, None),
        "k_max": np.clip(np.median((hi - mean) / sd, axis=0), 0.5, None),
        "positive_min": np.median(lo, axis=0) > 0,
    }


def _interval_at(cfg: GeneratorConfig, day: int) -> int:
    schedule = cfg.biopsy_schedule or [{"until_day": 0, "every_days": 30}]
    for seg in schedule:
        if day < seg["until_day"]:
            return max(1, int(seg["every_days"]))
    return max(1, int(schedule[-1]["every_days"]))


def _simulate_biopsies(cfg: GeneratorConfig, rng: np.random.Generator) -> list[tuple[int, int]]:
    """(day after transplant, grade) per biopsy. Dense early, sparser later; after a rejection
    follow-up controls with grade 0, during which no new rejection can start."""
    last_day = cfg.days - 1
    out: list[tuple[int, int]] = []
    day = FIRST_BIOPSY_DAY
    while day <= last_day:
        u = rng.random()
        grade = 2 if u < cfg.rejection_rate else 1 if u < cfg.rejection_rate + cfg.borderline_rate else 0
        out.append((day, grade))
        if grade == 2:
            for k in range(1, cfg.followup_biopsies_after_rejection + 1):
                follow = day + k * cfg.followup_interval_days
                if follow <= last_day:
                    out.append((follow, 0))
            day += cfg.followup_biopsies_after_rejection * cfg.followup_interval_days
        day += _interval_at(cfg, day)
    return out


def _pattern_strength(cfg: GeneratorConfig, biopsies: list[tuple[int, int]]) -> np.ndarray:
    """Per day 0..1: linear ramp up before a graded biopsy, linear return to normal afterwards."""
    days = np.arange(cfg.days)
    strength = np.zeros(cfg.days)
    before_len, after_len = max(cfg.pattern_days_before, 1), max(cfg.recovery_days, 1)
    for day, grade in biopsies:
        scale = 1.0 if grade == 2 else cfg.borderline_pattern_factor if grade == 1 else 0.0
        if scale == 0.0:
            continue
        before = day - days
        ramp = np.where((before >= 0) & (before <= cfg.pattern_days_before), 1 - before / before_len, 0.0)
        ramp = np.where((before < 0) & (-before <= cfg.recovery_days), 1 + before / after_len, ramp)
        strength = np.maximum(strength, scale * ramp)
    return strength


def _ar1(rng: np.random.Generator, n_days: int, n_series: int) -> np.ndarray:
    """Stationary AR(1) noise with unit variance, shape (n_days, n_series)."""
    noise = rng.standard_normal((n_days, n_series))
    out = np.empty_like(noise)
    out[0] = noise[0]
    innovation = np.sqrt(1 - AR_PHI ** 2)
    for t in range(1, n_days):
        out[t] = AR_PHI * out[t - 1] + innovation * noise[t]
    return out


def _patient_measurements(cfg: GeneratorConfig, rng: np.random.Generator, pop: dict,
                          strength: np.ndarray, pattern_idx: dict[int, float]) -> dict:
    n_groups = len(FEATURE_GROUPS)
    # personal normal
    level = pop["mu"] + pop["day_sd"] * BETWEEN_PATIENT_SD * np.clip(rng.standard_normal(n_groups), -2, 2)
    level = pop["mu"] * np.maximum(level / pop["mu"], 0.4)
    sd_level = pop["sd_level"] * np.exp(0.25 * np.clip(rng.standard_normal(n_groups), -2, 2))
    # which days are measured, and how often
    measured = rng.random(cfg.days) >= cfg.missing_day_rate
    multi = rng.random(cfg.days) < cfg.multi_measurement_rate
    counts = np.where(measured, np.where(multi, rng.integers(2, 5, cfg.days), 1), 0)
    day_idx = np.repeat(np.arange(cfg.days), counts)
    n = len(day_idx)
    within_day = np.arange(n) - np.repeat(np.cumsum(counts) - counts, counts)
    # values: personal level + autocorrelated day effect + small per-measurement jitter
    day_effect = _ar1(rng, cfg.days, n_groups)
    z = day_effect[day_idx] + SAME_DAY_JITTER * rng.standard_normal((n, n_groups))
    z /= np.sqrt(1 + SAME_DAY_JITTER ** 2)  # total spread = real day-to-day spread
    mean = level * np.maximum((level + pop["day_sd"] * z) / level, MIN_LEVEL_RATIO)
    sd = sd_level * np.exp(pop["sd_log_spread"] * rng.standard_normal((n, n_groups)))
    lo = mean - pop["k_min"] * sd * rng.uniform(0.8, 1.2, (n, n_groups))
    hi = mean + pop["k_max"] * sd * rng.uniform(0.8, 1.2, (n, n_groups))
    lo = np.where(pop["positive_min"], np.maximum(lo, 0.1 * mean), lo)
    included = rng.integers(INCLUDED_RANGE[0], INCLUDED_RANGE[1] + 1, n)
    cc = np.round(rng.uniform(CC_RANGE[0], CC_RANGE[1], n), 2)
    empty = rng.random(n) < EMPTY_ROW_RATE
    no_t_block = rng.random(n) < MISSING_T_BLOCK_RATE
    # ---- no random numbers below this line ----
    for g, rel in pattern_idx.items():
        factor = 1 + rel * strength[day_idx]  # > 0, so min < mean < max and the sign are kept
        mean[:, g] *= factor
        lo[:, g] *= factor
        hi[:, g] *= factor
    values = np.stack([mean, sd, lo, hi], axis=2)
    t_groups = [i for i, g in enumerate(FEATURE_GROUPS) if g in T_BLOCK_GROUPS]
    values[np.ix_(no_t_block, t_groups)] = np.nan
    values[empty] = np.nan
    return {"day_idx": day_idx, "within_day": within_day, "n_that_day": counts[day_idx],
            "values": values.reshape(n, n_groups * 4), "empty": empty,
            "included": np.where(empty, 0, included), "cc": np.where(empty, np.nan, cc)}


def _pattern_index(cfg: GeneratorConfig) -> dict[int, float]:
    out = {}
    for col, rel in (cfg.pattern or {}).items():
        group = col[:-5] if col.endswith("_mean") else None
        if group not in FEATURE_GROUPS:
            raise ValueError(f"generator pattern: '{col}' is not a known _mean column")
        if rel <= -1:
            raise ValueError(f"generator pattern: relative change of '{col}' must be above -1")
        out[FEATURE_GROUPS.index(group)] = float(rel)
    return out


def _integer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flags and counts as whole numbers that may be empty, so files show '1' and not '1.0'."""
    for c in INT_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    return df


def _generate_demo_patients(cfg: GeneratorConfig, pop: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(cfg.seed)
    pattern_idx = _pattern_index(cfg) if cfg.pattern_enabled else {}
    t0 = np.datetime64(FIRST_TRANSPLANT, "D")
    names, dates, included, cc, empty, values = [], [], [], [], [], []
    bx_rows, rejection_dates, borderline_dates = [], {}, {}
    for i in range(cfg.n_patients):
        pid = f"D{i + 1:03d}"
        transplant = t0 + int(rng.integers(0, TRANSPLANT_SPREAD_DAYS))
        biopsies = _simulate_biopsies(cfg, rng)
        strength = _pattern_strength(cfg, biopsies) if pattern_idx else np.zeros(cfg.days)
        m = _patient_measurements(cfg, rng, pop, strength, pattern_idx)

        day_text = np.datetime_as_string(transplant + m["day_idx"], unit="D")
        dates += [f"ecg_{d}_{k + 1}.csv" if c > 1 else f"ecg_{d}.csv"
                  for d, k, c in zip(day_text, m["within_day"], m["n_that_day"])]
        names += [pid] * len(day_text)
        included.append(m["included"]); cc.append(m["cc"])
        empty.append(m["empty"]); values.append(m["values"])

        as_date = lambda d: (transplant + d).astype(object)  # noqa: E731  (datetime.date)
        rejection_dates[pid] = [as_date(d) for d, g in biopsies if g == 2]
        borderline_dates[pid] = [as_date(d) for d, g in biopsies if g == 1]
        bx_rows += [(pid, no, str(transplant + d), GRADE_TEXT[g]) for no, (d, g) in enumerate(biopsies, 1)]

    empty_all = np.concatenate(empty) if empty else np.zeros(0, bool)
    flag = np.where(empty_all, np.nan, 1.0)
    ecg = pd.DataFrame(np.concatenate(values) if values else np.zeros((0, len(VALUE_COLUMNS))),
                       columns=VALUE_COLUMNS)
    ecg.insert(0, "rr_int", flag)
    ecg.insert(0, "rr", flag)
    ecg.insert(0, "hr", flag)
    ecg.insert(0, "cc", np.concatenate(cc) if cc else [])
    ecg.insert(0, "included_no", np.concatenate(included) if included else [])
    ecg.insert(0, "Date", pd.Series(dates, dtype=object))
    ecg.insert(0, "Name", pd.Series(names, dtype=object))
    ecg.insert(0, "Unnamed: 0", 0)
    ecg = _integer_columns(ecg[RAW_HEADER])
    bx = pd.DataFrame(bx_rows, columns=BIOPSY_HEADER)
    truth = {"pattern_columns": [f"{FEATURE_GROUPS[g]}_mean" for g in pattern_idx],
             "rejection_dates": rejection_dates, "borderline_dates": borderline_dates,
             "pattern_enabled": bool(cfg.pattern_enabled)}
    return ecg, bx, truth


def _read_real_rows(real_ecg_path) -> pd.DataFrame:
    """Rows of the real file exactly as they are (numbers read without rounding loss)."""
    real = pd.read_csv(_real_path(real_ecg_path), sep=";", encoding="utf-8-sig",
                       float_precision="round_trip", dtype={"Name": str, "Date": str})
    real.columns = [str(c).strip() for c in real.columns]
    if list(real.columns) != RAW_HEADER:
        raise ValueError("Real ECG file does not have the expected columns; cannot append real patients")
    return _integer_columns(real)


def _real_patient_biopsies(real: pd.DataFrame) -> pd.DataFrame:
    """INVENTED biopsy labels for the real patients.

    The clinic has not delivered a biopsy table yet. These rows exist only so that the real
    patients run through the whole pipeline; they say nothing about the real patients and must
    be replaced as soon as real findings are available. Biopsies are placed on measured dates
    only: a patient with a consecutive block of days gets three biopsies inside the block (the
    one near its end is "ja") plus one on a later single date, a patient measured on a single
    day gets one, everyone else gets two on the last dates that carry values.
    """
    rows = []
    date = pd.to_datetime(real["Date"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0])
    has_values = real[VALUE_COLUMNS].notna().any(axis=1)
    for name in pd.unique(real["Name"]):
        mine = real["Name"] == name
        days = sorted(set(date[mine & has_values].dropna())) or sorted(set(date[mine].dropna()))
        if not days:
            continue
        runs, run = [], [days[0]]
        for d in days[1:]:
            if (d - run[-1]).days <= 2:
                run.append(d)
            else:
                runs.append(run)
                run = [d]
        runs.append(run)
        block = max(runs, key=len)
        if len(block) >= 10:
            n = len(block)
            picks = [(block[round(0.2 * (n - 1))], 0), (block[round(0.55 * (n - 1))], 0), (block[n - 2], 2)]
            later = [d for d in days if d > block[-1]]
            if later:
                picks.append((later[-1], 0))
        elif len(days) == 1:
            picks = [(days[0], 0)]
        else:
            picks = [(d, 0) for d in days[-2:]]
        rows += [(name, no, d.strftime("%Y-%m-%d"), GRADE_TEXT[g]) for no, (d, g) in enumerate(picks, 1)]
    return pd.DataFrame(rows, columns=BIOPSY_HEADER)


def generate_demo(cfg: GeneratorConfig, real_ecg_path=None,
                  include_real: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (ecg_raw, biopsies_raw, truth) in the format of the real files.

    truth describes the demo patients only: pattern_columns (empty when the pattern is off),
    rejection_dates and borderline_dates per patient, pattern_enabled.
    """
    ecg, bx, truth = _generate_demo_patients(cfg, _population_stats(real_ecg_path))
    if include_real and _real_path(real_ecg_path).exists():  # echte Daten gibt es nur lokal
        real = _read_real_rows(real_ecg_path)  # appended unchanged, never touched by the pattern
        ecg = pd.concat([ecg, real], ignore_index=True)
        bx = pd.concat([bx, _real_patient_biopsies(real)], ignore_index=True)
    return ecg, bx, truth


def write_demo(cfg: GeneratorConfig, out_dir, real_ecg_path=None,
               include_real: bool = True) -> tuple[Path, Path]:
    """Writes ecg_demo.csv and biopsies_demo.csv (semicolon, UTF-8) into out_dir."""
    out_dir = Path(out_dir)
    if out_dir.resolve() == (ROOT / "data" / "raw").resolve():
        raise ValueError("Demo files must not be written into data/raw (real data only)")
    out_dir.mkdir(parents=True, exist_ok=True)
    ecg, bx, _ = generate_demo(cfg, real_ecg_path, include_real)
    ecg_path, bx_path = out_dir / "ecg_demo.csv", out_dir / "biopsies_demo.csv"
    ecg.to_csv(ecg_path, sep=";", index=False, encoding="utf-8")
    bx.to_csv(bx_path, sep=";", index=False, encoding="utf-8")
    return ecg_path, bx_path
