from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from kardio.constants import FEATURE_GROUPS, VALUE_COLUMNS

DATA = Path(__file__).parent / "data"
ROOT = Path(__file__).resolve().parents[1]

BASE = {"R_amplitude": 580, "R_upslope": 28000, "P_amplitude": -50, "T_amplitude": 270,
        "R_duration": 0.107, "P_duration": 0.123, "T_duration": 0.185,
        "RT_interval": 0.373, "qRT_interval": 0.41, "PR_interval": 0.16, "R_rate": 74}


def make_values(scale: float = 1.0, drop_t: bool = False) -> dict:
    out = {}
    for g in FEATURE_GROUPS:
        m = BASE[g] * scale
        out[f"{g}_mean"] = m
        out[f"{g}_sd"] = abs(m) * 0.1
        out[f"{g}_min"] = m - abs(m) * 0.2
        out[f"{g}_max"] = m + abs(m) * 0.2
    if drop_t:
        for c in list(out):
            if c.startswith(("T_amplitude", "T_duration", "RT_interval", "qRT_interval")):
                out[c] = ""
    return out


def raw_row(name, fname, included, cc, **kw):
    r = {"Unnamed: 0": 0, "Name": name, "Date": fname, "included_no": included,
         "cc": cc, "hr": 1, "rr": 1, "rr_int": 1}
    r.update(make_values(**kw))
    return r


@pytest.fixture(scope="session")
def ecg_small_path() -> Path:
    """AA: 3 Messungen am 01.01. (eine unvollstaendig), 1 am 02.01., 1 leere am 03.01.; ' bb ': 2 Tage."""
    rows = [
        raw_row("AA", "ecg_2025-01-01_1.csv", 30, 0.95, drop_t=True),
        raw_row("AA", "ecg_2025-01-01_2.csv", 30, 0.96),
        raw_row("AA", "ecg_2025-01-01_3.csv", 35, 0.94),
        raw_row("AA", "ecg_2025-01-02.csv", 33, 0.96, scale=1.05),
        {"Unnamed: 0": 0, "Name": "AA", "Date": "ecg_2025-01-03.csv", "included_no": 0,
         "cc": "", "hr": "", "rr": "", "rr_int": "", **{c: "" for c in VALUE_COLUMNS}},
        raw_row(" bb ", "ecg_2025-01-01.csv", 20, 0.9),
        raw_row(" bb ", "ecg_2025-01-05.csv", 25, 0.92, scale=0.9),
    ]
    DATA.mkdir(exist_ok=True)
    p = DATA / "ecg_small.csv"
    pd.DataFrame(rows).to_csv(p, sep=";", index=False)
    return p


@pytest.fixture
def biopsies_small_path() -> Path:
    return DATA / "biopsies_small.csv"


@pytest.fixture
def biopsies_variant_path() -> Path:
    return DATA / "biopsies_variant.csv"


@pytest.fixture
def real_ecg_path() -> Path:
    """Echte Patientendaten liegen nur lokal (nicht im Repo). Fehlen sie, werden diese Tests uebersprungen."""
    p = ROOT / "data" / "raw" / "ecg_real.csv"
    if not p.exists():
        pytest.skip("data/raw/ecg_real.csv nicht vorhanden (echte Daten liegen nur lokal)")
    return p


@pytest.fixture
def daily_df() -> pd.DataFrame:
    """Bereits bereinigt: 3 Patienten, 30 Tage; P2 mit Luecke an Tag 10-12, P3 nur 2 Tage."""
    rng = np.random.default_rng(0)
    rows = []
    for pid, ndays in [("P1", 30), ("P2", 30), ("P3", 2)]:
        for d in range(ndays):
            if pid == "P2" and 10 <= d <= 12:
                continue
            vals = make_values(scale=1 + rng.normal(0, 0.02))
            rows.append({"patient": pid, "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=d),
                         "included_no": 30, "cc": 0.95, **vals})
    return pd.DataFrame(rows)


def empty_biopsies() -> pd.DataFrame:
    return pd.DataFrame({"patient": pd.Series(dtype=str), "biopsy_no": pd.Series(dtype=int),
                         "date": pd.to_datetime([]), "grade": pd.Series(dtype=int),
                         "rejection": pd.Series(dtype=bool)})


def make_biopsies(rows: list[tuple]) -> pd.DataFrame:
    """rows: (patient, biopsy_no, 'YYYY-MM-DD', grade) -> Tabelle wie load_biopsies sie liefert."""
    df = pd.DataFrame(rows, columns=["patient", "biopsy_no", "date", "grade"])
    df["date"] = pd.to_datetime(df["date"])
    df["rejection"] = df["grade"] == 2
    return df


def make_windows(n_rej: int, n_no: int, n_mid: int = 0, shift_col: str | None = None,
                 shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Kuenstliche Fenstertabelle: Zufallswerte, optional ein Merkmal in der Abstossungsgruppe verschoben.
    Je Fenster ein eigener Patient."""
    from kardio.constants import WINDOW_FEATURE_COLUMNS
    rng = np.random.default_rng(seed)
    n = n_rej + n_mid + n_no
    df = pd.DataFrame(rng.normal(size=(n, len(WINDOW_FEATURE_COLUMNS))), columns=WINDOW_FEATURE_COLUMNS)
    grade = [2] * n_rej + [1] * n_mid + [0] * n_no
    df.insert(0, "usable", True)
    df.insert(0, "under_treatment", False)
    df.insert(0, "n_data_days", 10)
    df.insert(0, "rejection", [g == 2 for g in grade])
    df.insert(0, "grade", grade)
    df.insert(0, "date", pd.Timestamp("2025-03-01"))
    df.insert(0, "biopsy_no", 1)
    df.insert(0, "patient", [f"P{i}" for i in range(n)])
    if shift_col:
        df.loc[df["grade"] == 2, shift_col] += shift
        df.loc[df["grade"] == 1, shift_col] += shift / 2
    return df
