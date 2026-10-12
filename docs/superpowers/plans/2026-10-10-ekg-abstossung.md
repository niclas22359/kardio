# EKG-Abstoßungserkennung Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit-Dashboard, das EKG- und Biopsietabellen einliest, bereinigt, statistisch vergleicht, ein erklärbares Risikomodell lernt und einen Arzttext erzeugt; läuft lokal und auf Render.

**Architecture:** Sieben Stationen als reine Funktionen "DataFrame rein, DataFrame raus" in `src/kardio/`. Ein Generator erzeugt Demo-Daten im Realformat vor Station 1. Das Dashboard in `app/` ruft nur die Pipeline-Funktion auf und rendert Ergebnisse.

**Tech Stack:** Python 3.13, pandas, numpy, scipy, scikit-learn, statsmodels (BH-Korrektur), streamlit, plotly, pyyaml, requests, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-ekg-abstossung-design.md`

## Global Constraints

- Python 3.13; alle Abhängigkeiten in `requirements.txt`, Umgebung `.venv`.
- Dateien unter 500 Zeilen.
- Keine Schlüssel im Repo. Vor jedem Push: `git ls-files -z | xargs -0 grep -lE "mga-[A-Za-z0-9]{8}" | wc -l` muss 0 sein.
- Jede Station ist eine Funktion mit typisierten Parametern und DataFrame-Rückgabe; keine Station liest Dateien selbst außer Station 1 und der Generator.
- Merkmalsgruppen (11): `R_amplitude, R_upslope, P_amplitude, T_amplitude, R_duration, P_duration, T_duration, RT_interval, qRT_interval, PR_interval, R_rate`; Kennzahlen je Gruppe: `_mean, _sd, _min, _max`.
- Zufall immer mit festem Seed (Standard 42).
- Alle Standardwerte aus `config/analysis.yaml` und `config/generator.yaml`, nie fest im Code.
- UI-Texte Deutsch, Code und Bezeichner Englisch.

## Review Focus

1. Biopsietabelle mit nur einem Patienten und null Abstoßungen: Station 4 und 5 dürfen nicht abstürzen (Division durch 0, leere Gruppen), sondern Status "zu wenig Daten" liefern. Test in Task 8 und 9.
2. EKG-Datei mit Komma statt Semikolon oder Dezimalkomma: Einleser muss das Trennzeichen erkennen und eine verständliche Fehlermeldung liefern, wenn nichts passt. Test in Task 3.
3. Biopsiedatum vor der ersten EKG-Messung: Fenster hat 0 Datentage, wird markiert, kein Absturz. Test in Task 7.
4. Patientenkürzel mit Leerzeichen oder unterschiedlicher Schreibweise (`mc `, `MC`): Normalisierung vor dem Abgleich. Test in Task 3.
5. Upload einer Datei, die gar keine EKG-Tabelle ist (z.B. die Biopsietabelle im EKG-Feld): klare Fehlermeldung mit fehlenden Spalten. Test in Task 3 und 11.

## Ergänzung aus der Prism-Vorlage (gilt vor allen Tasks)

Die Klinik wertet heute mit drei Befundgruppen und drei Kennzahlen je Merkmal
aus (`docs/referenz/prism_vorlage/`). Folgende Abweichungen von den
Task-Texten unten sind verbindlich; wo ein Task-Code dem widerspricht, gilt
dieser Abschnitt.

**A. Befund als Stufe (Task 4).** `parse_rejection` wird zu
`parse_grade(v) -> int | None` mit 0/1/2. Zuordnung:
```python
GRADE_WORDS = {
    0: {"0", "nein", "no", "n", "false", "negativ", "negative", "neg", "keine", "keine abstossung",
        "keine abstoßung", "none", "no rejection", "0r"},
    1: {"1", "zwischen", "zwischengruppe", "borderline", "grenzwertig", "mild", "1r", "leicht"},
    2: {"2", "3", "ja", "yes", "y", "j", "true", "positiv", "positive", "pos", "abstossung",
        "abstoßung", "rejection", "2r", "3r"},
}
```
Reihenfolge der Prüfung: exakter Treffer in 0, dann 1, dann 2; danach
Teilstring-Regeln (`startswith("kein")`/`"no "` → 0, `"zwischen"`/`"border"` → 1,
`"abst"`/`"reject"`/`"pos"` → 2). Ausgabe von `load_biopsies`: Spalten
`patient, biopsy_no, date, grade (int), rejection (bool = grade == 2)`.
Test-Parametrisierung ergänzen: `("Zwischengruppe", 1), ("1R", 1), ("2R", 2), ("0R", 0), ("borderline", 1)`.
`tests/data/biopsies_small.csv` bekommt eine vierte Zeile `BB;2;2025-01-08;Zwischengruppe`.
Report (Task 5) ergänzt `n_borderline` und Zeile "Davon Zwischengruppe".

**B. Drei Kennzahlen je Merkmal (Task 6).** `DEV_FEATURES` wird zu 33
Einträgen: je Gruppe g `g_mean` (Absolutwert), `g_cv` (= `g_sd / |g_mean|`,
vor dem Melt als Spalte berechnet) und `g_dev` (relative Abweichung
`|g_mean - center| / |center|`, berechnet nach dem Merge mit der
Normallinie; für `g_dev` ist `center` die Normallinie von `g_mean`). Die
Normallinie wird nur für `g_mean` und `g_cv` berechnet; `g_dev` bekommt
`center = 0`, `scale = MAD der g_dev-Werte in der Basisphase`
(oder 0.05 als Minimum). Test `test_deviations_shape_and_z` erwartet
`nunique() == 33`. `WINDOW_FEATURE_COLUMNS` (Task 7) wird damit 33 x 4 = 132;
Tests in Task 7/8 passen die Zahl 88 → 132 an.

**C. Drei-Gruppen-Statistik (Task 8).** `compare_features` liefert je
Spalte zusätzlich `p_kw` (Kruskal-Wallis über Gruppen 0/1/2, nur wenn
jede vorhandene Gruppe >= 2 Fenster hat), `p_01`, `p_12`, `p_02`
(paarweise Mann-Whitney, `p_02` ist das bisherige `p_raw`). BH-Korrektur
weiterhin auf `p_02`. `median_1` ergänzen. Die Fenster-Tabelle (Task 7)
führt `grade` mit. Testhelfer `_windows` bekommt Parameter `n_mid=0` für
Gruppe 1.

**D. Risikomodell (Task 9).** Unverändert binär auf `rejection`
(Stufe 2). Fenster mit `grade == 1` werden beim Lernen ausgeschlossen
(weder positiv noch negativ), beim Scoring aber bewertet.

**E. Generator (Task 10).** `rejection_rate: 0.25` bleibt Stufe 2,
neu `borderline_rate: 0.15` (Stufe 1, Muster mit Faktor 0,5, keine
Behandlungs-Folgebiopsien). Spalte `Befund` enthält
`nein | Zwischengruppe | ja`. `truth` bekommt zusätzlich
`borderline_dates`.

**F. Merkmalsseite im Prism-Look (Task 13, `ui_features.py`).** Oberhalb
der Rangliste ein Raster aus Boxplots: Auswahl der Kennzahl
(Absolutwert / CV / Abweichung) und der Fenster-Aggregation
(Standard `zmean`), dann je Merkmalsgruppe ein Boxplot mit den drei Gruppen
in grün (`#2ecc40`), gelb (`#ffdc00`), rot (`#ff4136`), Titel
`"<Merkmal> <Kennzahl>"`, darüber die drei p-Werte als Text-Annotationen
(`0 vs 1`, `1 vs 2`, `0 vs 2`), Format `<0.0001` bzw. 4 Nachkommastellen,
Titel grün gefärbt wenn `p_02 < 0.05`. Umsetzung mit `plotly.subplots.make_subplots`,
4 Spalten, `go.Box(boxpoints="outliers")`. Zusätzlich ein Schalter
"alle Messungen statt Tagesauswahl" ist NICHT Teil von heute (Spec 3.3
erwähnt die Prism-Variante "alle Werte"; bewusst weggelassen).

**G. Übersichtsseite (Task 13).** Spalte "Zwischengruppe" ergänzen.

---

## Dateistruktur

```
kardio/
  requirements.txt
  render.yaml
  start.bat                      # Doppelklick-Start lokal
  config/analysis.yaml
  config/generator.yaml
  src/kardio/__init__.py
  src/kardio/constants.py        # FEATURE_GROUPS, STATS, Spaltennamen
  src/kardio/config.py           # YAML laden -> dataclasses
  src/kardio/ingest/__init__.py
  src/kardio/ingest/ecg.py       # Station 1a: load_ecg, select_daily
  src/kardio/ingest/biopsies.py  # Station 1b: load_biopsies, Spaltenabgleich
  src/kardio/ingest/report.py    # Qualitätsbericht
  src/kardio/features/baseline.py   # Station 2
  src/kardio/features/windows.py    # Station 3
  src/kardio/stats/compare.py       # Station 4
  src/kardio/model/risk.py          # Station 5
  src/kardio/explain/text.py        # Station 6 (MGA + Vorlage)
  src/kardio/generator/demo.py      # Generator
  src/kardio/pipeline.py            # alles verketten, AnalysisResult
  src/kardio/run.py                 # CLI
  app/main.py                       # Streamlit Einstieg + Passwort
  app/pages_upload.py, pages_overview.py, pages_features.py,
  app/pages_patient.py, pages_model.py
  tests/conftest.py                 # Fixtures: kleine Tabellen
  tests/test_*.py
```

---

### Task 1: Projektgerüst, Konstanten, Konfiguration

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `config/analysis.yaml`, `config/generator.yaml`, `src/kardio/__init__.py`, `src/kardio/constants.py`, `src/kardio/config.py`, `tests/conftest.py`, `tests/test_config.py`, `start.bat`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `FEATURE_GROUPS: list[str]` (11), `STATS = ["mean","sd","min","max"]`, `VALUE_COLUMNS: list[str]` (44 = 11x4), `load_analysis_config(path=None) -> AnalysisConfig`, `load_generator_config(path=None) -> GeneratorConfig`.

- [ ] **Step 1: requirements und pyproject**

`requirements.txt`:
```
pandas==2.2.3
numpy==2.1.3
scipy==1.14.1
scikit-learn==1.5.2
statsmodels==0.14.4
streamlit==1.40.1
plotly==5.24.1
pyyaml==6.0.2
requests==2.32.3
pytest==8.3.3
```

`pyproject.toml`:
```toml
[project]
name = "kardio"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Umgebung anlegen**

Run: `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt`
Expected: ohne Fehler.

- [ ] **Step 3: Konfigurationsdateien**

`config/analysis.yaml`:
```yaml
window_days: 14
baseline_days: 14
baseline_min_days: 5
min_window_data_days: 3
treatment_days: 21
min_windows_total: 20
min_rejection_windows: 8
min_rejection_windows_stats: 5
amber_threshold: 0.30
red_threshold: 0.60
z_alert: 2.0
seed: 42
```

`config/generator.yaml`:
```yaml
n_patients: 40
days: 365
missing_day_rate: 0.10
multi_measurement_rate: 0.15
rejection_rate: 0.25
pattern_enabled: true
pattern_days_before: 10
recovery_days: 14
pattern:
  R_amplitude_mean: -0.15
  qRT_interval_mean: 0.08
  R_rate_mean: 0.10
  T_amplitude_mean: -0.20
biopsy_schedule:
  - {until_day: 28, every_days: 7}
  - {until_day: 90, every_days: 14}
  - {until_day: 180, every_days: 30}
  - {until_day: 365, every_days: 90}
followup_biopsies_after_rejection: 3
followup_interval_days: 7
seed: 42
```

- [ ] **Step 4: Test für Konstanten und Config**

`tests/test_config.py`:
```python
from kardio.constants import FEATURE_GROUPS, STATS, VALUE_COLUMNS
from kardio.config import load_analysis_config, load_generator_config

def test_value_columns_are_44():
    assert len(FEATURE_GROUPS) == 11
    assert STATS == ["mean", "sd", "min", "max"]
    assert len(VALUE_COLUMNS) == 44
    assert "qRT_interval_mean" in VALUE_COLUMNS

def test_analysis_config_defaults():
    cfg = load_analysis_config()
    assert cfg.window_days == 14
    assert cfg.red_threshold == 0.6

def test_generator_config_pattern():
    cfg = load_generator_config()
    assert cfg.rejection_rate == 0.25
    assert cfg.pattern["R_amplitude_mean"] == -0.15
```

- [ ] **Step 5: Test laufen lassen, muss fehlschlagen**

Run: `.venv/Scripts/pytest tests/test_config.py -v`
Expected: FAIL, ModuleNotFoundError kardio.

- [ ] **Step 6: Implementierung**

`src/kardio/constants.py`:
```python
FEATURE_GROUPS = [
    "R_amplitude", "R_upslope", "P_amplitude", "T_amplitude",
    "R_duration", "P_duration", "T_duration",
    "RT_interval", "qRT_interval", "PR_interval", "R_rate",
]
STATS = ["mean", "sd", "min", "max"]
VALUE_COLUMNS = [f"{g}_{s}" for g in FEATURE_GROUPS for s in STATS]
MEAN_COLUMNS = [f"{g}_mean" for g in FEATURE_GROUPS]

COL_PATIENT = "patient"
COL_DATE = "date"
COL_INCLUDED = "included_no"
COL_CC = "cc"
COL_BIOPSY_NO = "biopsy_no"
COL_REJECTION = "rejection"
```

`src/kardio/config.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

@dataclass
class AnalysisConfig:
    window_days: int = 14
    baseline_days: int = 14
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
    pattern_enabled: bool = True
    pattern_days_before: int = 10
    recovery_days: int = 14
    pattern: dict = field(default_factory=dict)
    biopsy_schedule: list = field(default_factory=list)
    followup_biopsies_after_rejection: int = 3
    followup_interval_days: int = 7
    seed: int = 42

def _load(path: Path | None, default_name: str) -> dict:
    p = Path(path) if path else ROOT / "config" / default_name
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_analysis_config(path: str | Path | None = None) -> AnalysisConfig:
    return AnalysisConfig(**_load(path, "analysis.yaml"))

def load_generator_config(path: str | Path | None = None) -> GeneratorConfig:
    return GeneratorConfig(**_load(path, "generator.yaml"))
```

`src/kardio/__init__.py`: leer. `tests/conftest.py`: vorerst leer (wird in Task 3 gefüllt).

`start.bat`:
```bat
@echo off
cd /d "%~dp0"
if not exist .venv (python -m venv .venv && .venv\Scripts\pip install -r requirements.txt)
.venv\Scripts\streamlit run app\main.py
```

`.gitignore` ergänzen: `.venv/`, `data/processed/`, `data/demo/`, `*.jsonl`, `.env`, `__pycache__/`, `.pytest_cache/`.

- [ ] **Step 7: Tests grün**

Run: `.venv/Scripts/pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: Projektgeruest, Konstanten, Konfiguration"
```

---

### Task 2: Testdaten-Fixtures

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/data/ecg_small.csv`, `tests/data/biopsies_small.csv`, `tests/data/biopsies_variant.csv`

**Interfaces:**
- Produces: Fixtures `ecg_small_path`, `biopsies_small_path`, `biopsies_variant_path`, `real_ecg_path`, `daily_df` (bereits bereinigte 3-Patienten-Tabelle, 30 Tage).

- [ ] **Step 1: Kleine EKG-Datei im Realformat**

`tests/data/ecg_small.csv` (Semikolon, gleiche Kopfzeile wie `data/raw/ecg_real.csv`). Inhalt: Patient `AA` mit 3 Messungen am 2025-01-01 (eine mit leerem T-Block, eine vollständig mit included_no 30, eine vollständig mit included_no 35), 1 Messung am 2025-01-02, eine komplett leere am 2025-01-03; Patient ` bb ` (mit Leerzeichen, klein) mit 2 Messungen an 2025-01-01 und 2025-01-05. Werte für die 44 Spalten: realistische Zahlen aus den Bereichen der echten Datei (R_amplitude ~580, R_upslope ~28000, P_amplitude ~-50, T_amplitude ~270, R_duration ~0.107, P_duration ~0.123, T_duration ~0.185, RT ~0.373, qRT ~0.41, PR ~0.16, R_rate ~74; sd ein Zehntel, min/max plus/minus ein Fünftel).

Die Datei wird in Step 3 durch ein Skript erzeugt, damit die Zahlen konsistent sind.

- [ ] **Step 2: Biopsiedateien**

`tests/data/biopsies_small.csv`:
```
Patient;Biopsie_Nr;Datum;Befund
AA;1;2025-01-02;nein
AA;2;2025-01-05;ja
BB;1;2025-01-05;nein
```

`tests/data/biopsies_variant.csv` (englisch, Komma, andere Reihenfolge, 1/0):
```
result,biopsy number,id,date
0,1,AA,02.01.2025
1,2,AA,05/01/2025
```

- [ ] **Step 3: conftest mit Generator-Skript für ecg_small**

`tests/conftest.py`:
```python
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
    return ROOT / "data" / "raw" / "ecg_real.csv"

@pytest.fixture
def daily_df() -> pd.DataFrame:
    """3 Patienten, 30 Tage, Patient P2 mit Luecke an Tag 10-12, P3 nur 2 Tage."""
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
```

- [ ] **Step 4: Smoke-Test**

Run: `.venv/Scripts/pytest --collect-only -q`
Expected: keine Fehler beim Import.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test: Fixtures fuer kleine EKG- und Biopsietabellen"
```

---

### Task 3: Station 1a, EKG einlesen und Tagesregel

**Files:**
- Create: `src/kardio/ingest/__init__.py`, `src/kardio/ingest/ecg.py`, `tests/test_ingest_ecg.py`

**Interfaces:**
- Produces: `load_ecg(path_or_buffer) -> pd.DataFrame` (Spalten `patient, date (datetime64), included_no, cc, *VALUE_COLUMNS`, eine Zeile pro Rohmessung; leere Werte als NaN); `select_daily(df) -> tuple[pd.DataFrame, pd.DataFrame]` (daily, dropped); `normalize_patient(s: str) -> str`; `class IngestError(ValueError)`.

- [ ] **Step 1: Tests**

`tests/test_ingest_ecg.py`:
```python
import io
import numpy as np
import pandas as pd
import pytest
from kardio.ingest.ecg import load_ecg, select_daily, normalize_patient, IngestError
from kardio.constants import VALUE_COLUMNS

def test_normalize_patient():
    assert normalize_patient(" bb ") == "BB"
    assert normalize_patient("Mc") == "MC"

def test_load_small(ecg_small_path):
    df = load_ecg(ecg_small_path)
    assert len(df) == 7
    assert set(df["patient"]) == {"AA", "BB"}
    assert df["date"].dtype.kind == "M"
    assert df.loc[df["patient"] == "AA", "date"].min() == pd.Timestamp("2025-01-01")
    empty = df[(df["patient"] == "AA") & (df["date"] == "2025-01-03")]
    assert empty[VALUE_COLUMNS].isna().all(axis=None)

def test_load_real(real_ecg_path):
    df = load_ecg(real_ecg_path)
    assert len(df) == 70
    assert set(df["patient"]) == {"MC", "MW", "JM239"}
    assert df[VALUE_COLUMNS].isna().all(axis=1).sum() == 4

def test_select_daily_rule(ecg_small_path):
    daily, dropped = select_daily(load_ecg(ecg_small_path))
    aa1 = daily[(daily["patient"] == "AA") & (daily["date"] == "2025-01-01")]
    assert len(aa1) == 1
    assert aa1["included_no"].iloc[0] == 35          # vollstaendig + meiste Schlaege
    assert len(dropped) == 2
    # leere Messung bleibt als Tag ohne Daten erhalten
    aa3 = daily[(daily["patient"] == "AA") & (daily["date"] == "2025-01-03")]
    assert len(aa3) == 1 and aa3[VALUE_COLUMNS].isna().all(axis=None)

def test_select_daily_real_mc(real_ecg_path):
    daily, _ = select_daily(load_ecg(real_ecg_path))
    assert (daily["patient"] == "MC").sum() == 1

def test_comma_separator_detected():
    txt = "Name,Date,included_no,cc," + ",".join(VALUE_COLUMNS) + "\n" \
          "AA,ecg_2025-01-01.csv,30,0.95," + ",".join(["1"] * 44) + "\n"
    df = load_ecg(io.StringIO(txt))
    assert len(df) == 1

def test_wrong_file_raises(biopsies_small_path):
    with pytest.raises(IngestError) as e:
        load_ecg(biopsies_small_path)
    assert "Name" in str(e.value) or "R_amplitude_mean" in str(e.value)
```

- [ ] **Step 2: Fehlschlag prüfen**

Run: `.venv/Scripts/pytest tests/test_ingest_ecg.py -v`
Expected: FAIL, Import.

- [ ] **Step 3: Implementierung**

`src/kardio/ingest/ecg.py`:
```python
from __future__ import annotations
import re
import pandas as pd
from kardio.constants import VALUE_COLUMNS, COL_PATIENT, COL_DATE, COL_INCLUDED, COL_CC

class IngestError(ValueError):
    pass

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

def normalize_patient(s) -> str:
    return str(s).strip().upper()

def _read_any(path_or_buffer) -> pd.DataFrame:
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    try:
        df = pd.read_csv(path_or_buffer, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as e:
        raise IngestError(f"Datei konnte nicht gelesen werden: {e}") from e
    if df.shape[1] < 5:
        raise IngestError("Datei hat zu wenige Spalten, ist das eine EKG-Tabelle?")
    return df

def load_ecg(path_or_buffer) -> pd.DataFrame:
    raw = _read_any(path_or_buffer)
    raw.columns = [c.strip() for c in raw.columns]
    missing = [c for c in ["Name", "Date", "included_no"] + VALUE_COLUMNS if c not in raw.columns]
    if missing:
        raise IngestError("EKG-Tabelle: fehlende Spalten: " + ", ".join(missing[:6]))
    out = pd.DataFrame()
    out[COL_PATIENT] = raw["Name"].map(normalize_patient)
    dates = raw["Date"].astype(str).str.extract(DATE_RE)[0]
    if dates.isna().any():
        raise IngestError("EKG-Tabelle: Datum konnte nicht aus Spalte 'Date' gelesen werden")
    out[COL_DATE] = pd.to_datetime(dates)
    out[COL_INCLUDED] = pd.to_numeric(raw["included_no"], errors="coerce").fillna(0).astype(int)
    out[COL_CC] = pd.to_numeric(raw.get("cc"), errors="coerce")
    for c in VALUE_COLUMNS:
        out[c] = pd.to_numeric(raw[c], errors="coerce")
    return out

def select_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pro Patient und Tag eine Messung: wenigste NaN, dann included_no, dann cc."""
    d = df.copy()
    d["_nan"] = d[VALUE_COLUMNS].isna().sum(axis=1)
    d = d.sort_values([COL_PATIENT, COL_DATE, "_nan", COL_INCLUDED, COL_CC],
                      ascending=[True, True, True, False, False])
    keep = ~d.duplicated([COL_PATIENT, COL_DATE], keep="first")
    daily = d[keep].drop(columns="_nan").reset_index(drop=True)
    dropped = d[~keep].drop(columns="_nan").reset_index(drop=True)
    return daily, dropped
```

`src/kardio/ingest/__init__.py`: leer.

- [ ] **Step 4: Tests grün**

Run: `.venv/Scripts/pytest tests/test_ingest_ecg.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: EKG einlesen und Tagesregel"
```

---

### Task 4: Station 1b, Biopsien einlesen mit tolerantem Spaltenabgleich

**Files:**
- Create: `src/kardio/ingest/biopsies.py`, `tests/test_ingest_biopsies.py`

**Interfaces:**
- Produces: `guess_columns(columns: list[str]) -> dict[str, str | None]` (Schlüssel `patient, biopsy_no, date, rejection`); `load_biopsies(path_or_buffer, mapping: dict | None = None) -> pd.DataFrame` (Spalten `patient, biopsy_no (int), date (datetime64), rejection (bool)`, sortiert); `parse_rejection(v) -> bool | None`.

- [ ] **Step 1: Tests**

`tests/test_ingest_biopsies.py`:
```python
import pandas as pd
import pytest
from kardio.ingest.biopsies import load_biopsies, guess_columns, parse_rejection
from kardio.ingest.ecg import IngestError

def test_guess_german():
    m = guess_columns(["Patient", "Biopsie_Nr", "Datum", "Befund"])
    assert m == {"patient": "Patient", "biopsy_no": "Biopsie_Nr", "date": "Datum", "rejection": "Befund"}

def test_guess_english_unordered():
    m = guess_columns(["result", "biopsy number", "id", "date"])
    assert m["patient"] == "id" and m["rejection"] == "result" and m["biopsy_no"] == "biopsy number"

def test_guess_missing_returns_none():
    assert guess_columns(["foo", "bar"])["patient"] is None

@pytest.mark.parametrize("v,exp", [("ja", True), ("Nein", False), ("1", True), (0, False),
                                   ("positiv", True), ("yes", True), ("NO", False),
                                   ("Abstossung", True), ("keine Abstossung", False), ("", None)])
def test_parse_rejection(v, exp):
    assert parse_rejection(v) is exp

def test_load_small(biopsies_small_path):
    b = load_biopsies(biopsies_small_path)
    assert list(b.columns) == ["patient", "biopsy_no", "date", "rejection"]
    assert b["rejection"].tolist() == [False, True, False]
    assert b["date"].dtype.kind == "M"

def test_load_variant(biopsies_variant_path):
    b = load_biopsies(biopsies_variant_path)
    assert b["patient"].tolist() == ["AA", "AA"]
    assert b["date"].tolist() == [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-05")]
    assert b["rejection"].tolist() == [False, True]

def test_missing_column_raises(tmp_path):
    p = tmp_path / "b.csv"; p.write_text("foo;bar\n1;2\n")
    with pytest.raises(IngestError) as e:
        load_biopsies(p)
    assert "Patient" in str(e.value)

def test_manual_mapping(tmp_path):
    p = tmp_path / "b.csv"; p.write_text("x;y;z\nAA;2025-01-02;ja\n")
    b = load_biopsies(p, mapping={"patient": "x", "date": "y", "rejection": "z", "biopsy_no": None})
    assert b["biopsy_no"].tolist() == [1]
```

- [ ] **Step 2: Fehlschlag prüfen**

Run: `.venv/Scripts/pytest tests/test_ingest_biopsies.py -v`
Expected: FAIL.

- [ ] **Step 3: Implementierung**

`src/kardio/ingest/biopsies.py`:
```python
from __future__ import annotations
import re
import pandas as pd
from kardio.ingest.ecg import IngestError, _read_any, normalize_patient
from kardio.constants import COL_PATIENT, COL_DATE, COL_BIOPSY_NO, COL_REJECTION

ALIASES = {
    "patient": ["patient", "patientenkuerzel", "patientenkürzel", "kuerzel", "kürzel", "name", "id",
                "patient_id", "patientid", "pseudonym", "code"],
    "biopsy_no": ["biopsie_nr", "biopsienummer", "biopsie", "biopsy_no", "biopsy number", "biopsy",
                  "nr", "nummer", "number", "no"],
    "date": ["datum", "biopsiedatum", "date", "biopsy_date", "biopsy date", "tag"],
    "rejection": ["befund", "abstossung", "abstoßung", "rejection", "result", "ergebnis",
                  "label", "outcome", "acr"],
}
TRUE_WORDS = {"ja", "yes", "y", "j", "true", "1", "positiv", "positive", "pos", "abstossung", "abstoßung", "rejection"}
FALSE_WORDS = {"nein", "no", "n", "false", "0", "negativ", "negative", "neg", "keine abstossung",
               "keine abstoßung", "no rejection", "none"}

def _norm(c: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(c).strip().lower())

def guess_columns(columns: list[str]) -> dict[str, str | None]:
    normed = {_norm(c): c for c in columns}
    out: dict[str, str | None] = {}
    used: set[str] = set()
    for key, aliases in ALIASES.items():
        hit = None
        for a in aliases:
            a = _norm(a)
            for n, orig in normed.items():
                if orig in used:
                    continue
                if n == a or n.replace(" ", "") == a.replace(" ", ""):
                    hit = orig; break
            if hit: break
        if hit is None:
            for a in aliases:
                for n, orig in normed.items():
                    if orig not in used and _norm(a) in n:
                        hit = orig; break
                if hit: break
        out[key] = hit
        if hit: used.add(hit)
    return out

def parse_rejection(v) -> bool | None:
    s = str(v).strip().lower()
    if s in ("", "nan", "none"):
        return None
    if s in FALSE_WORDS:
        return False
    if s in TRUE_WORDS:
        return True
    if s.startswith("kein") or s.startswith("no "):
        return False
    if "abst" in s or "reject" in s or "pos" in s:
        return True
    return None

def load_biopsies(path_or_buffer, mapping: dict | None = None) -> pd.DataFrame:
    raw = _read_any(path_or_buffer) if not isinstance(path_or_buffer, pd.DataFrame) else path_or_buffer
    raw.columns = [str(c).strip() for c in raw.columns]
    m = mapping or guess_columns(list(raw.columns))
    missing = [k for k in ("patient", "date", "rejection") if not m.get(k)]
    if missing:
        names = {"patient": "Patient", "date": "Datum", "rejection": "Befund"}
        raise IngestError("Biopsie-Tabelle: Spalte nicht gefunden für: " +
                          ", ".join(names[k] for k in missing))
    out = pd.DataFrame()
    out[COL_PATIENT] = raw[m["patient"]].map(normalize_patient)
    out[COL_DATE] = pd.to_datetime(raw[m["date"]], dayfirst=True, errors="coerce",
                                   format="mixed")
    if out[COL_DATE].isna().any():
        raise IngestError("Biopsie-Tabelle: Datum nicht lesbar in Spalte " + m["date"])
    rej = raw[m["rejection"]].map(parse_rejection)
    if rej.isna().any():
        bad = raw.loc[rej.isna(), m["rejection"]].astype(str).unique()[:3]
        raise IngestError("Biopsie-Tabelle: Befund nicht verstanden: " + ", ".join(bad))
    out[COL_REJECTION] = rej.astype(bool)
    if m.get("biopsy_no"):
        out[COL_BIOPSY_NO] = pd.to_numeric(raw[m["biopsy_no"]], errors="coerce")
    out = out.sort_values([COL_PATIENT, COL_DATE]).reset_index(drop=True)
    if COL_BIOPSY_NO not in out or out[COL_BIOPSY_NO].isna().any():
        out[COL_BIOPSY_NO] = out.groupby(COL_PATIENT).cumcount() + 1
    out[COL_BIOPSY_NO] = out[COL_BIOPSY_NO].astype(int)
    return out[[COL_PATIENT, COL_BIOPSY_NO, COL_DATE, COL_REJECTION]]
```

Hinweis: `pd.read_csv` mit `sep=None` erkennt Komma und Semikolon. `format="mixed"` mit `dayfirst=True` liest `02.01.2025`, `05/01/2025` und `2025-01-02`.

- [ ] **Step 4: Tests grün**

Run: `.venv/Scripts/pytest tests/test_ingest_biopsies.py -v`
Expected: alle passed.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Biopsien einlesen mit tolerantem Spaltenabgleich"
```

---

### Task 5: Qualitätsbericht

**Files:**
- Create: `src/kardio/ingest/report.py`, `tests/test_report.py`

**Interfaces:**
- Produces: `@dataclass QualityReport` (Felder `n_patients_ecg, n_patients_biopsy, n_patients_matched, patients_only_ecg: list, patients_only_biopsy: list, n_days, n_empty_days, n_dropped_duplicates, n_biopsies, n_rejections, date_min, date_max`); `build_report(daily, dropped, biopsies) -> QualityReport`; `QualityReport.as_rows() -> list[tuple[str, str]]` für die Anzeige.

- [ ] **Step 1: Test**

`tests/test_report.py`:
```python
from kardio.ingest.ecg import load_ecg, select_daily
from kardio.ingest.biopsies import load_biopsies
from kardio.ingest.report import build_report

def test_report_small(ecg_small_path, biopsies_small_path):
    daily, dropped = select_daily(load_ecg(ecg_small_path))
    b = load_biopsies(biopsies_small_path)
    r = build_report(daily, dropped, b)
    assert r.n_patients_ecg == 2 and r.n_patients_biopsy == 2 and r.n_patients_matched == 2
    assert r.n_days == 5 and r.n_empty_days == 1 and r.n_dropped_duplicates == 2
    assert r.n_biopsies == 3 and r.n_rejections == 1
    assert any("Patienten" in k for k, _ in r.as_rows())
```

- [ ] **Step 2: Fehlschlag, dann Implementierung**

`src/kardio/ingest/report.py`:
```python
from dataclasses import dataclass, field
import pandas as pd
from kardio.constants import VALUE_COLUMNS, COL_PATIENT, COL_DATE, COL_REJECTION

@dataclass
class QualityReport:
    n_patients_ecg: int
    n_patients_biopsy: int
    n_patients_matched: int
    patients_only_ecg: list
    patients_only_biopsy: list
    n_days: int
    n_empty_days: int
    n_dropped_duplicates: int
    n_biopsies: int
    n_rejections: int
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None

    def as_rows(self) -> list[tuple[str, str]]:
        f = lambda d: d.strftime("%d.%m.%Y") if d is not None and not pd.isna(d) else "-"
        return [
            ("Patienten mit EKG", str(self.n_patients_ecg)),
            ("Patienten mit Biopsien", str(self.n_patients_biopsy)),
            ("Patienten in beiden Tabellen", str(self.n_patients_matched)),
            ("Nur in EKG-Tabelle", ", ".join(self.patients_only_ecg) or "-"),
            ("Nur in Biopsie-Tabelle", ", ".join(self.patients_only_biopsy) or "-"),
            ("Messtage gesamt", str(self.n_days)),
            ("Davon Tage ohne Werte", str(self.n_empty_days)),
            ("Verworfene Mehrfachmessungen", str(self.n_dropped_duplicates)),
            ("Biopsien", str(self.n_biopsies)),
            ("Davon Abstoßung", str(self.n_rejections)),
            ("Zeitraum", f"{f(self.date_min)} bis {f(self.date_max)}"),
        ]

def build_report(daily: pd.DataFrame, dropped: pd.DataFrame, biopsies: pd.DataFrame) -> QualityReport:
    pe, pb = set(daily[COL_PATIENT]), set(biopsies[COL_PATIENT])
    return QualityReport(
        n_patients_ecg=len(pe), n_patients_biopsy=len(pb), n_patients_matched=len(pe & pb),
        patients_only_ecg=sorted(pe - pb), patients_only_biopsy=sorted(pb - pe),
        n_days=len(daily), n_empty_days=int(daily[VALUE_COLUMNS].isna().all(axis=1).sum()),
        n_dropped_duplicates=len(dropped), n_biopsies=len(biopsies),
        n_rejections=int(biopsies[COL_REJECTION].sum()) if len(biopsies) else 0,
        date_min=daily[COL_DATE].min() if len(daily) else None,
        date_max=daily[COL_DATE].max() if len(daily) else None,
    )
```

- [ ] **Step 3: Test grün, Commit**

Run: `.venv/Scripts/pytest tests/test_report.py -v`
```bash
git add -A && git commit -m "feat: Qualitaetsbericht nach dem Einlesen"
```

---

### Task 6: Station 2, Normallinie und Abweichungen

**Files:**
- Create: `src/kardio/features/__init__.py`, `src/kardio/features/baseline.py`, `tests/test_baseline.py`

**Interfaces:**
- Consumes: `daily` aus Task 3, `biopsies` aus Task 4, `AnalysisConfig`.
- Produces: `compute_baselines(daily, biopsies, cfg) -> pd.DataFrame` (Spalten `patient, feature, center, scale, n_days, weak: bool`); `compute_deviations(daily, baselines, cfg) -> pd.DataFrame` Langformat (`patient, date, feature, value, z, delta_prev, trend_7d`). `feature` sind die 11 `_mean`-Spalten plus 11 `_sd`-Spalten (22).

- [ ] **Step 1: Tests**

`tests/test_baseline.py`:
```python
import numpy as np
import pandas as pd
from kardio.config import AnalysisConfig
from kardio.features.baseline import compute_baselines, compute_deviations

def _bx(): return pd.DataFrame({"patient": [], "biopsy_no": [], "date": pd.to_datetime([]), "rejection": []})

def test_baseline_per_patient(daily_df):
    cfg = AnalysisConfig()
    b = compute_baselines(daily_df, _bx(), cfg)
    assert set(b["patient"]) == {"P1", "P2", "P3"}
    p1 = b[(b.patient == "P1") & (b.feature == "R_rate_mean")].iloc[0]
    assert abs(p1["center"] - 74) < 5 and p1["scale"] > 0 and not p1["weak"]
    p3 = b[(b.patient == "P3") & (b.feature == "R_rate_mean")].iloc[0]
    assert p3["weak"]  # nur 2 Tage < baseline_min_days -> Population

def test_baseline_excludes_rejection_window(daily_df):
    cfg = AnalysisConfig()
    bx = pd.DataFrame({"patient": ["P1"], "biopsy_no": [1],
                       "date": [pd.Timestamp("2025-01-10")], "rejection": [True]})
    d = daily_df.copy()
    mask = (d.patient == "P1") & (d.date <= "2025-01-10")
    d.loc[mask, "R_rate_mean"] = 200  # Ausreisser im Abstossungsfenster
    b = compute_baselines(d, bx, cfg)
    p1 = b[(b.patient == "P1") & (b.feature == "R_rate_mean")].iloc[0]
    assert p1["center"] < 100

def test_deviations_shape_and_z(daily_df):
    cfg = AnalysisConfig()
    b = compute_baselines(daily_df, _bx(), cfg)
    dev = compute_deviations(daily_df, b, cfg)
    assert set(dev.columns) == {"patient", "date", "feature", "value", "z", "delta_prev", "trend_7d"}
    assert dev["feature"].nunique() == 22
    assert dev["z"].abs().median() < 2
    first = dev[(dev.patient == "P1") & (dev.feature == "R_rate_mean")].sort_values("date").iloc[0]
    assert np.isnan(first["delta_prev"])

def test_zero_scale_does_not_crash(daily_df):
    d = daily_df.copy(); d["P_duration_mean"] = 0.123
    cfg = AnalysisConfig()
    dev = compute_deviations(d, compute_baselines(d, _bx(), cfg), cfg)
    assert np.isfinite(dev.loc[dev.feature == "P_duration_mean", "z"]).all()
```

- [ ] **Step 2: Fehlschlag prüfen, dann Implementierung**

`src/kardio/features/baseline.py`:
```python
from __future__ import annotations
import numpy as np
import pandas as pd
from kardio.config import AnalysisConfig
from kardio.constants import FEATURE_GROUPS, COL_PATIENT, COL_DATE, COL_REJECTION

DEV_FEATURES = [f"{g}_mean" for g in FEATURE_GROUPS] + [f"{g}_sd" for g in FEATURE_GROUPS]

def _mad(x: pd.Series) -> float:
    med = x.median()
    return float(1.4826 * (x - med).abs().median())

def _exclusion_mask(daily: pd.DataFrame, biopsies: pd.DataFrame, cfg: AnalysisConfig) -> pd.Series:
    """True fuer Tage, die in einem Abstossungsfenster (davor) oder einer Behandlungsphase (danach) liegen."""
    mask = pd.Series(False, index=daily.index)
    rej = biopsies[biopsies[COL_REJECTION]] if len(biopsies) else biopsies
    for _, r in rej.iterrows():
        lo = r[COL_DATE] - pd.Timedelta(days=cfg.window_days)
        hi = r[COL_DATE] + pd.Timedelta(days=cfg.treatment_days)
        mask |= (daily[COL_PATIENT] == r[COL_PATIENT]) & (daily[COL_DATE] >= lo) & (daily[COL_DATE] <= hi)
    return mask

def compute_baselines(daily: pd.DataFrame, biopsies: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    excl = _exclusion_mask(daily, biopsies, cfg)
    clean = daily[~excl].dropna(subset=DEV_FEATURES, how="all")
    rows = []
    pop = {f: (float(clean[f].median()), _mad(clean[f].dropna())) for f in DEV_FEATURES}
    for pid, g in daily.groupby(COL_PATIENT):
        gc = clean[clean[COL_PATIENT] == pid].sort_values(COL_DATE).head(cfg.baseline_days)
        for f in DEV_FEATURES:
            vals = gc[f].dropna()
            weak = len(vals) < cfg.baseline_min_days
            if weak:
                center, scale = pop[f]
            else:
                center, scale = float(vals.median()), _mad(vals)
            if not np.isfinite(scale) or scale <= 0:
                scale = max(abs(center) * 0.05, 1e-9)
            rows.append({"patient": pid, "feature": f, "center": center, "scale": scale,
                         "n_days": int(len(vals)), "weak": bool(weak)})
    return pd.DataFrame(rows)

def compute_deviations(daily: pd.DataFrame, baselines: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    long = daily.melt(id_vars=[COL_PATIENT, COL_DATE], value_vars=DEV_FEATURES,
                      var_name="feature", value_name="value")
    long = long.merge(baselines[["patient", "feature", "center", "scale"]],
                      on=[COL_PATIENT, "feature"], how="left")
    long["z"] = (long["value"] - long["center"]) / long["scale"]
    long = long.sort_values([COL_PATIENT, "feature", COL_DATE])
    grp = long.groupby([COL_PATIENT, "feature"], sort=False)
    long["delta_prev"] = grp["value"].diff()
    def _trend(s: pd.Series) -> pd.Series:
        return s.rolling(7, min_periods=3).apply(
            lambda w: np.polyfit(np.arange(len(w)), w, 1)[0] if np.isfinite(w).sum() >= 3 else np.nan, raw=True)
    long["trend_7d"] = grp["z"].transform(_trend)
    return long[["patient", "date", "feature", "value", "z", "delta_prev", "trend_7d"]].reset_index(drop=True)
```

`src/kardio/features/__init__.py`: leer.

- [ ] **Step 3: Tests grün, Commit**

Run: `.venv/Scripts/pytest tests/test_baseline.py -v`
```bash
git add -A && git commit -m "feat: persoenliche Normallinie und Abweichungen"
```

---

### Task 7: Station 3, Fenster um Biopsien

**Files:**
- Create: `src/kardio/features/windows.py`, `tests/test_windows.py`

**Interfaces:**
- Consumes: `deviations` (Task 6), `biopsies`, `cfg`.
- Produces: `build_windows(deviations, biopsies, cfg) -> pd.DataFrame`: eine Zeile pro Biopsie mit `patient, biopsy_no, date, rejection, n_data_days, under_treatment: bool, usable: bool` und je Feature f in `DEV_FEATURES` die Spalten `f__zmean, f__zmaxabs, f__slope, f__alertfrac`. `WINDOW_FEATURE_COLUMNS: list[str]` (88). Außerdem `rolling_windows(deviations, cfg) -> pd.DataFrame` mit denselben Feature-Spalten für jeden Patient/Tag (für die Risikokurve), Spalten `patient, date, n_data_days, *WINDOW_FEATURE_COLUMNS`.

- [ ] **Step 1: Tests**

`tests/test_windows.py`:
```python
import numpy as np
import pandas as pd
from kardio.config import AnalysisConfig
from kardio.features.baseline import compute_baselines, compute_deviations
from kardio.features.windows import build_windows, rolling_windows, WINDOW_FEATURE_COLUMNS

def _dev(daily_df, bx):
    cfg = AnalysisConfig()
    return compute_deviations(daily_df, compute_baselines(daily_df, bx, cfg), cfg), cfg

def test_window_basic(daily_df):
    bx = pd.DataFrame({"patient": ["P1", "P1"], "biopsy_no": [1, 2],
                       "date": pd.to_datetime(["2025-01-20", "2025-01-28"]), "rejection": [False, True]})
    dev, cfg = _dev(daily_df, bx)
    w = build_windows(dev, bx, cfg)
    assert len(w) == 2 and len(WINDOW_FEATURE_COLUMNS) == 88
    assert w.iloc[0]["n_data_days"] == 15 and w.iloc[0]["usable"]
    assert "R_rate_mean__zmean" in w.columns

def test_window_before_first_measurement(daily_df):
    bx = pd.DataFrame({"patient": ["P1"], "biopsy_no": [1],
                       "date": pd.to_datetime(["2024-12-01"]), "rejection": [False]})
    dev, cfg = _dev(daily_df, bx)
    w = build_windows(dev, bx, cfg)
    assert w.iloc[0]["n_data_days"] == 0 and not w.iloc[0]["usable"]
    assert np.isnan(w.iloc[0]["R_rate_mean__zmean"])

def test_under_treatment_flag(daily_df):
    bx = pd.DataFrame({"patient": ["P1", "P1"], "biopsy_no": [1, 2],
                       "date": pd.to_datetime(["2025-01-10", "2025-01-20"]), "rejection": [True, False]})
    dev, cfg = _dev(daily_df, bx)
    w = build_windows(dev, bx, cfg)
    assert not w.iloc[0]["under_treatment"] and w.iloc[1]["under_treatment"]

def test_rolling(daily_df):
    dev, cfg = _dev(daily_df, pd.DataFrame({"patient": [], "biopsy_no": [], "date": pd.to_datetime([]), "rejection": []}))
    r = rolling_windows(dev, cfg)
    assert set(r.patient) == {"P1", "P2", "P3"}
    assert (r[r.patient == "P1"]["n_data_days"].max()) == 15
```

- [ ] **Step 2: Fehlschlag, dann Implementierung**

`src/kardio/features/windows.py`:
```python
from __future__ import annotations
import numpy as np
import pandas as pd
from kardio.config import AnalysisConfig
from kardio.constants import COL_PATIENT, COL_DATE, COL_REJECTION, COL_BIOPSY_NO
from kardio.features.baseline import DEV_FEATURES

AGGS = ["zmean", "zmaxabs", "slope", "alertfrac"]
WINDOW_FEATURE_COLUMNS = [f"{f}__{a}" for f in DEV_FEATURES for a in AGGS]

def _summarize(win: pd.DataFrame, cfg: AnalysisConfig) -> dict:
    out = {c: np.nan for c in WINDOW_FEATURE_COLUMNS}
    n_days = win[COL_DATE].nunique() if len(win) else 0
    for f, g in win.groupby("feature"):
        z = g.sort_values(COL_DATE)["z"].dropna()
        if len(z) == 0:
            continue
        out[f"{f}__zmean"] = float(z.mean())
        out[f"{f}__zmaxabs"] = float(z.abs().max())
        out[f"{f}__alertfrac"] = float((z.abs() > cfg.z_alert).mean())
        out[f"{f}__slope"] = float(np.polyfit(np.arange(len(z)), z.values, 1)[0]) if len(z) >= 3 else np.nan
    out["n_data_days"] = int(n_days)
    return out

def build_windows(dev: pd.DataFrame, biopsies: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    rows = []
    bx = biopsies.sort_values([COL_PATIENT, COL_DATE])
    for _, b in bx.iterrows():
        lo = b[COL_DATE] - pd.Timedelta(days=cfg.window_days)
        win = dev[(dev[COL_PATIENT] == b[COL_PATIENT]) & (dev[COL_DATE] >= lo) & (dev[COL_DATE] <= b[COL_DATE])]
        prev_rej = bx[(bx[COL_PATIENT] == b[COL_PATIENT]) & bx[COL_REJECTION] & (bx[COL_DATE] < b[COL_DATE])]
        under = bool(len(prev_rej) and (b[COL_DATE] - prev_rej[COL_DATE].max()).days <= cfg.window_days + cfg.treatment_days)
        row = {"patient": b[COL_PATIENT], "biopsy_no": int(b[COL_BIOPSY_NO]), "date": b[COL_DATE],
               "rejection": bool(b[COL_REJECTION]), "under_treatment": under}
        row.update(_summarize(win, cfg))
        row["usable"] = row["n_data_days"] >= cfg.min_window_data_days and not under
        rows.append(row)
    cols = ["patient", "biopsy_no", "date", "rejection", "n_data_days", "under_treatment", "usable"] + WINDOW_FEATURE_COLUMNS
    return pd.DataFrame(rows, columns=cols)

def rolling_windows(dev: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    rows = []
    for pid, g in dev.groupby(COL_PATIENT):
        for day in sorted(g[COL_DATE].unique()):
            lo = day - pd.Timedelta(days=cfg.window_days)
            win = g[(g[COL_DATE] >= lo) & (g[COL_DATE] <= day)]
            row = {"patient": pid, "date": day}
            row.update(_summarize(win, cfg))
            rows.append(row)
    return pd.DataFrame(rows, columns=["patient", "date", "n_data_days"] + WINDOW_FEATURE_COLUMNS)
```

- [ ] **Step 3: Tests grün, Commit**

```bash
git add -A && git commit -m "feat: Fenster um Biopsien und rollierende Fenster"
```

---

### Task 8: Station 4, statistischer Vergleich

**Files:**
- Create: `src/kardio/stats/__init__.py`, `src/kardio/stats/compare.py`, `tests/test_compare.py`

**Interfaces:**
- Consumes: `windows` (Task 7).
- Produces: `compare_features(windows, cfg) -> tuple[pd.DataFrame, bool]`: Rangliste mit Spalten `feature, agg, column, n_rej, n_no, p_raw, p_adj, cliffs_delta, auc, median_rej, median_no`, sortiert nach `p_adj` aufsteigend; zweites Element `reliable: bool` (True wenn `n_rej >= cfg.min_rejection_windows_stats`).

- [ ] **Step 1: Tests**

`tests/test_compare.py`:
```python
import numpy as np
import pandas as pd
from kardio.config import AnalysisConfig
from kardio.features.windows import WINDOW_FEATURE_COLUMNS
from kardio.stats.compare import compare_features

def _windows(n_rej, n_no, shift_col=None, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = n_rej + n_no
    df = pd.DataFrame(rng.normal(size=(n, len(WINDOW_FEATURE_COLUMNS))), columns=WINDOW_FEATURE_COLUMNS)
    df.insert(0, "rejection", [True] * n_rej + [False] * n_no)
    df.insert(0, "usable", True); df.insert(0, "patient", [f"P{i}" for i in range(n)])
    if shift_col:
        df.loc[df.rejection, shift_col] += shift
    return df

def test_detects_shifted_feature():
    w = _windows(20, 60, "R_rate_mean__zmean", 2.0)
    rank, reliable = compare_features(w, AnalysisConfig())
    assert reliable
    assert rank.iloc[0]["column"] == "R_rate_mean__zmean"
    assert rank.iloc[0]["p_adj"] < 0.05 and rank.iloc[0]["auc"] > 0.8 and rank.iloc[0]["cliffs_delta"] > 0.5

def test_null_has_no_significant():
    rank, _ = compare_features(_windows(20, 60), AnalysisConfig())
    assert (rank["p_adj"] < 0.05).sum() == 0

def test_too_few_rejections_flagged():
    rank, reliable = compare_features(_windows(2, 20), AnalysisConfig())
    assert not reliable and len(rank) == 88

def test_no_rejections_does_not_crash():
    rank, reliable = compare_features(_windows(0, 10), AnalysisConfig())
    assert not reliable and rank["p_raw"].isna().all()
```

- [ ] **Step 2: Fehlschlag, dann Implementierung**

`src/kardio/stats/compare.py`:
```python
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from kardio.config import AnalysisConfig
from kardio.features.windows import WINDOW_FEATURE_COLUMNS

def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() - (diff < 0).mean())

def compare_features(windows: pd.DataFrame, cfg: AnalysisConfig) -> tuple[pd.DataFrame, bool]:
    w = windows[windows["usable"]] if "usable" in windows else windows
    rows = []
    for col in WINDOW_FEATURE_COLUMNS:
        feat, agg = col.split("__")
        a = w.loc[w["rejection"], col].dropna().to_numpy()
        b = w.loc[~w["rejection"], col].dropna().to_numpy()
        p = auc = np.nan
        if len(a) >= 2 and len(b) >= 2 and (np.std(np.r_[a, b]) > 0):
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            y = np.r_[np.ones(len(a)), np.zeros(len(b))]
            auc = float(roc_auc_score(y, np.r_[a, b]))
            auc = max(auc, 1 - auc)
        rows.append({"feature": feat, "agg": agg, "column": col, "n_rej": len(a), "n_no": len(b),
                     "p_raw": p, "cliffs_delta": _cliffs_delta(a, b), "auc": auc,
                     "median_rej": float(np.median(a)) if len(a) else np.nan,
                     "median_no": float(np.median(b)) if len(b) else np.nan})
    rank = pd.DataFrame(rows)
    rank["p_adj"] = np.nan
    ok = rank["p_raw"].notna()
    if ok.any():
        rank.loc[ok, "p_adj"] = multipletests(rank.loc[ok, "p_raw"], method="fdr_bh")[1]
    rank = rank.sort_values(["p_adj", "p_raw"], na_position="last").reset_index(drop=True)
    n_rej = int(w["rejection"].sum()) if len(w) else 0
    return rank, n_rej >= cfg.min_rejection_windows_stats
```

`src/kardio/stats/__init__.py`: leer.

- [ ] **Step 3: Tests grün, Commit**

```bash
git add -A && git commit -m "feat: statistischer Merkmalsvergleich mit BH-Korrektur"
```

---

### Task 9: Station 5, Risikomodell

**Files:**
- Create: `src/kardio/model/__init__.py`, `src/kardio/model/risk.py`, `tests/test_risk.py`

**Interfaces:**
- Consumes: `windows`, `rolling` (Task 7), `cfg`.
- Produces: `@dataclass ModelReport` (`status: str` in `{"ok","insufficient"}`, `reason: str`, `n_windows, n_rejections, auc_cv, sensitivity, specificity, threshold, weights: pd.DataFrame (column, weight)`, `auc_gb: float | None`); `fit_risk_model(windows, cfg) -> tuple[ModelReport, Pipeline | None]`; `score_timeline(model, rolling, cfg) -> pd.DataFrame` (`patient, date, risk` mit NaN wenn `n_data_days < cfg.min_window_data_days` oder model None); `traffic_light(risk, cfg) -> str` in `{"grey","green","amber","red"}`.

- [ ] **Step 1: Tests**

`tests/test_risk.py`:
```python
import numpy as np
import pandas as pd
from kardio.config import AnalysisConfig
from kardio.features.windows import WINDOW_FEATURE_COLUMNS
from kardio.model.risk import fit_risk_model, score_timeline, traffic_light
from tests.test_compare import _windows

def test_insufficient_data():
    rep, model = fit_risk_model(_windows(3, 10), AnalysisConfig())
    assert rep.status == "insufficient" and model is None and "8" in rep.reason

def test_learns_pattern():
    w = _windows(25, 75, "R_rate_mean__zmean", 2.5)
    rep, model = fit_risk_model(w, AnalysisConfig())
    assert rep.status == "ok" and rep.auc_cv > 0.8 and model is not None
    assert rep.weights.iloc[0]["column"] == "R_rate_mean__zmean"

def test_null_auc_near_half():
    rep, _ = fit_risk_model(_windows(25, 75), AnalysisConfig())
    assert 0.3 < rep.auc_cv < 0.7

def test_score_timeline_and_lights():
    cfg = AnalysisConfig()
    w = _windows(25, 75, "R_rate_mean__zmean", 2.5)
    rep, model = fit_risk_model(w, cfg)
    rolling = w[WINDOW_FEATURE_COLUMNS].copy()
    rolling.insert(0, "n_data_days", [1] + [10] * (len(w) - 1))
    rolling.insert(0, "date", pd.Timestamp("2025-01-01")); rolling.insert(0, "patient", w["patient"])
    t = score_timeline(model, rolling, cfg)
    assert np.isnan(t.iloc[0]["risk"]) and t["risk"].dropna().between(0, 1).all()
    assert traffic_light(np.nan, cfg) == "grey" and traffic_light(0.1, cfg) == "green"
    assert traffic_light(0.4, cfg) == "amber" and traffic_light(0.9, cfg) == "red"

def test_missing_values_in_windows_ok():
    w = _windows(25, 75, "R_rate_mean__zmean", 2.5)
    w.loc[::7, "T_amplitude_mean__slope"] = np.nan
    rep, model = fit_risk_model(w, AnalysisConfig())
    assert rep.status == "ok"
```

- [ ] **Step 2: Fehlschlag, dann Implementierung**

`src/kardio/model/risk.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from kardio.config import AnalysisConfig
from kardio.features.windows import WINDOW_FEATURE_COLUMNS

@dataclass
class ModelReport:
    status: str
    reason: str
    n_windows: int
    n_rejections: int
    auc_cv: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    threshold: float = 0.5
    weights: pd.DataFrame | None = None
    auc_gb: float | None = None

def _pipe() -> Pipeline:
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))])

def fit_risk_model(windows: pd.DataFrame, cfg: AnalysisConfig) -> tuple[ModelReport, Pipeline | None]:
    w = windows[windows["usable"]] if "usable" in windows else windows
    X = w[WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = w["rejection"].to_numpy(dtype=int)
    n, n_rej = len(w), int(y.sum())
    if n < cfg.min_windows_total or n_rej < cfg.min_rejection_windows or n_rej == n:
        return ModelReport("insufficient", f"Zu wenig Daten: {n} Fenster, {n_rej} Abstoßungen "
                           f"(mindestens {cfg.min_windows_total} / {cfg.min_rejection_windows} nötig)", n, n_rej), None
    groups = w["patient"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    proba = cross_val_predict(_pipe(), X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, proba))
    pred = proba >= cfg.red_threshold
    sens = float(pred[y == 1].mean()); spec = float((~pred[y == 0]).mean())
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=cfg.seed)
    gb_proba = cross_val_predict(Pipeline([("imp", SimpleImputer(strategy="median")), ("gb", gb)]),
                                 X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    model = _pipe().fit(X, y)
    coef = model.named_steps["lr"].coef_[0]
    weights = pd.DataFrame({"column": WINDOW_FEATURE_COLUMNS, "weight": coef})
    weights = weights.reindex(weights["weight"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    rep = ModelReport("ok", "", n, n_rej, auc_cv=auc, sensitivity=sens, specificity=spec,
                      threshold=cfg.red_threshold, weights=weights, auc_gb=float(roc_auc_score(y, gb_proba)))
    return rep, model

def score_timeline(model: Pipeline | None, rolling: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    out = rolling[["patient", "date"]].copy()
    out["risk"] = np.nan
    if model is None or len(rolling) == 0:
        return out
    ok = rolling["n_data_days"] >= cfg.min_window_data_days
    if ok.any():
        out.loc[ok, "risk"] = model.predict_proba(rolling.loc[ok, WINDOW_FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]
    return out

def traffic_light(risk: float, cfg: AnalysisConfig) -> str:
    if risk is None or np.isnan(risk):
        return "grey"
    if risk >= cfg.red_threshold:
        return "red"
    if risk >= cfg.amber_threshold:
        return "amber"
    return "green"
```

`src/kardio/model/__init__.py`: leer. `tests/__init__.py` anlegen (leer), damit `from tests.test_compare import _windows` geht.

- [ ] **Step 3: Tests grün, Commit**

```bash
git add -A && git commit -m "feat: erklaerbares Risikomodell mit Patienten-Kreuzvalidierung"
```

---

### Task 10: Demo-Generator

**Files:**
- Create: `src/kardio/generator/__init__.py`, `src/kardio/generator/demo.py`, `tests/test_generator.py`

**Interfaces:**
- Consumes: `GeneratorConfig`, echte Datei zur Ableitung der Wertebereiche (`data/raw/ecg_real.csv`).
- Produces: `generate_demo(cfg, real_ecg_path=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]`: Roh-EKG im Realformat (gleiche Spalten wie echte Datei inkl. `Name`, `Date` als `ecg_YYYY-MM-DD_n.csv`), Biopsietabelle (`Patient;Biopsie_Nr;Datum;Befund`), und `truth: dict` (`pattern_columns: list[str]`, `rejection_dates: dict[patient, list[date]]`). `write_demo(cfg, out_dir) -> tuple[Path, Path]` schreibt `ecg_demo.csv` und `biopsies_demo.csv` (Semikolon).

- [ ] **Step 1: Tests**

`tests/test_generator.py`:
```python
import pandas as pd
from kardio.config import load_generator_config
from kardio.generator.demo import generate_demo, write_demo
from kardio.ingest.ecg import load_ecg, select_daily
from kardio.ingest.biopsies import load_biopsies

def test_generate_shapes(real_ecg_path):
    cfg = load_generator_config(); cfg.n_patients = 5; cfg.days = 120
    ecg, bx, truth = generate_demo(cfg, real_ecg_path)
    assert set(["Name", "Date", "included_no", "cc"]).issubset(ecg.columns)
    assert ecg["Name"].nunique() == 5
    assert bx["Befund"].isin(["ja", "nein"]).all()
    assert 0.1 < (bx["Befund"] == "ja").mean() < 0.45
    assert truth["pattern_columns"] == ["R_amplitude_mean", "qRT_interval_mean", "R_rate_mean", "T_amplitude_mean"]

def test_roundtrip_through_ingest(tmp_path, real_ecg_path):
    cfg = load_generator_config(); cfg.n_patients = 3; cfg.days = 60
    e, b = write_demo(cfg, tmp_path, real_ecg_path)
    daily, dropped = select_daily(load_ecg(e))
    assert len(dropped) > 0  # Mehrfachmessungen vorhanden
    assert daily.groupby("patient").size().min() > 40  # ~10% Luecken
    assert len(load_biopsies(b)) > 0

def test_pattern_shifts_values(real_ecg_path):
    cfg = load_generator_config(); cfg.n_patients = 10; cfg.days = 200
    ecg_on, bx_on, truth = generate_demo(cfg, real_ecg_path)
    cfg.pattern_enabled = False
    ecg_off, _, _ = generate_demo(cfg, real_ecg_path)
    # gleicher Seed -> identische Patienten, nur die Musterwerte unterscheiden sich
    assert not ecg_on["R_amplitude_mean"].equals(ecg_off["R_amplitude_mean"])
    assert ecg_on["P_duration_mean"].equals(ecg_off["P_duration_mean"])

def test_reproducible(real_ecg_path):
    cfg = load_generator_config(); cfg.n_patients = 3; cfg.days = 30
    a = generate_demo(cfg, real_ecg_path)[0]; b = generate_demo(cfg, real_ecg_path)[0]
    assert a.equals(b)
```

- [ ] **Step 2: Fehlschlag, dann Implementierung**

`src/kardio/generator/demo.py`:
```python
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from kardio.config import GeneratorConfig, ROOT
from kardio.constants import FEATURE_GROUPS, VALUE_COLUMNS
from kardio.ingest.ecg import load_ecg

RAW_HEADER = ["Unnamed: 0", "Name", "Date", "included_no", "cc", "hr", "rr", "rr_int"] + VALUE_COLUMNS

def _population_stats(real_ecg_path: Path | None) -> dict[str, tuple[float, float]]:
    p = Path(real_ecg_path) if real_ecg_path else ROOT / "data" / "raw" / "ecg_real.csv"
    df = load_ecg(p)
    df = df[df["patient"] == "MW"].dropna(subset=VALUE_COLUMNS)
    return {c: (float(df[c].mean()), float(df[c].std() or abs(df[c].mean()) * 0.05)) for c in VALUE_COLUMNS}

def _biopsy_days(cfg: GeneratorConfig) -> list[int]:
    days, start = [], 7
    for seg in cfg.biopsy_schedule:
        d = start
        while d <= seg["until_day"] and d <= cfg.days:
            days.append(d); d += seg["every_days"]
        start = d
    return sorted(set(days))

def generate_demo(cfg: GeneratorConfig, real_ecg_path=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(cfg.seed)
    pop = _population_stats(real_ecg_path)
    pattern_cols = list(cfg.pattern.keys())
    t0 = pd.Timestamp("2025-01-01")
    ecg_rows, bx_rows, truth_rej = [], [], {}
    for i in range(cfg.n_patients):
        pid = f"D{i+1:03d}"
        tx = t0 + pd.Timedelta(days=int(rng.integers(0, 60)))
        center = {c: m + rng.normal(0, s * 0.5) for c, (m, s) in pop.items()}
        # Biopsieplan + Befunde
        sched = _biopsy_days(cfg)
        biopsies: list[tuple[int, bool]] = []
        d_idx = 0
        while d_idx < len(sched):
            day = sched[d_idx]
            rej = bool(rng.random() < cfg.rejection_rate)
            biopsies.append((day, rej))
            if rej:
                for k in range(1, cfg.followup_biopsies_after_rejection + 1):
                    fd = day + k * cfg.followup_interval_days
                    if fd <= cfg.days:
                        biopsies.append((fd, False))
                sched = [s for s in sched if s > day + cfg.followup_biopsies_after_rejection * cfg.followup_interval_days]
                d_idx = 0
            else:
                d_idx += 1
        biopsies.sort()
        rej_days = [d for d, r in biopsies if r]
        truth_rej[pid] = [(tx + pd.Timedelta(days=d)).date() for d in rej_days]
        for n, (d, r) in enumerate(biopsies, 1):
            bx_rows.append({"Patient": pid, "Biopsie_Nr": n,
                            "Datum": (tx + pd.Timedelta(days=d)).strftime("%Y-%m-%d"), "Befund": "ja" if r else "nein"})
        # Tagesmessungen
        for day in range(cfg.days):
            if rng.random() < cfg.missing_day_rate:
                continue
            n_meas = 1 + (int(rng.integers(1, 4)) if rng.random() < cfg.multi_measurement_rate else 0)
            # Musterfaktor
            factor = {c: 0.0 for c in pattern_cols}
            if cfg.pattern_enabled:
                for rd in rej_days:
                    before = rd - day
                    if 0 <= before <= cfg.pattern_days_before:
                        ramp = 1 - before / cfg.pattern_days_before
                        for c in pattern_cols: factor[c] = max(factor[c], ramp)
                    elif 0 < day - rd <= cfg.recovery_days:
                        ramp = 1 - (day - rd) / cfg.recovery_days
                        for c in pattern_cols: factor[c] = max(factor[c], ramp)
            for k in range(n_meas):
                row = {"Unnamed: 0": 0, "Name": pid, "hr": 1, "rr": 1, "rr_int": 1,
                       "Date": f"ecg_{(tx + pd.Timedelta(days=day)).strftime('%Y-%m-%d')}" + (f"_{k+1}.csv" if n_meas > 1 else ".csv"),
                       "included_no": int(rng.integers(14, 44)), "cc": round(float(rng.uniform(0.87, 0.97)), 2)}
                for g in FEATURE_GROUPS:
                    m_col = f"{g}_mean"
                    m = center[m_col] + rng.normal(0, pop[m_col][1] * 0.6)
                    if m_col in factor:
                        m *= 1 + cfg.pattern[m_col] * factor[m_col]
                    sd = abs(center[f"{g}_sd"]) * rng.uniform(0.7, 1.3)
                    row[m_col] = m; row[f"{g}_sd"] = sd
                    row[f"{g}_min"] = m - 2 * sd; row[f"{g}_max"] = m + 2 * sd
                ecg_rows.append(row)
    ecg = pd.DataFrame(ecg_rows, columns=RAW_HEADER)
    bx = pd.DataFrame(bx_rows, columns=["Patient", "Biopsie_Nr", "Datum", "Befund"])
    return ecg, bx, {"pattern_columns": pattern_cols if cfg.pattern_enabled else [], "rejection_dates": truth_rej}

def write_demo(cfg: GeneratorConfig, out_dir: Path, real_ecg_path=None) -> tuple[Path, Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ecg, bx, _ = generate_demo(cfg, real_ecg_path)
    e, b = out_dir / "ecg_demo.csv", out_dir / "biopsies_demo.csv"
    ecg.to_csv(e, sep=";", index=False); bx.to_csv(b, sep=";", index=False)
    return e, b
```

Hinweis zu `test_pattern_shifts_values`: Der Zufallsstrom muss bei an/aus identisch bleiben. Das ist gegeben, weil das Muster nur multipliziert und keine Zufallszahlen zieht.

- [ ] **Step 3: Tests grün, Commit**

```bash
git add -A && git commit -m "feat: Demo-Generator im Realformat mit schaltbarem Muster"
```

---

### Task 11: Pipeline, CLI und End-to-End-Nachweis

**Files:**
- Create: `src/kardio/pipeline.py`, `src/kardio/run.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `@dataclass AnalysisResult` (`daily, dropped, biopsies, report: QualityReport, baselines, deviations, windows, rolling, ranking, ranking_reliable, model_report, timeline, data_source_label: str`); `run_analysis(ecg_src, biopsies_src, cfg=None, mapping=None, data_source_label="Echte Daten") -> AnalysisResult`; `run_analysis_frames(daily_raw_df, biopsies_df, ...)` für bereits geladene Tabellen; `save_processed(result, out_dir)`.

- [ ] **Step 1: Tests (das sind die Nachweise 3, 4, 5 aus der Spec)**

`tests/test_pipeline.py`:
```python
import pandas as pd
import pytest
from kardio.config import load_generator_config, AnalysisConfig
from kardio.generator.demo import write_demo
from kardio.pipeline import run_analysis
from kardio.ingest.ecg import IngestError

@pytest.fixture(scope="module")
def demo_on(tmp_path_factory, ):
    cfg = load_generator_config(); cfg.n_patients = 40
    return write_demo(cfg, tmp_path_factory.mktemp("on"))

@pytest.fixture(scope="module")
def demo_off(tmp_path_factory):
    cfg = load_generator_config(); cfg.n_patients = 40; cfg.pattern_enabled = False
    return write_demo(cfg, tmp_path_factory.mktemp("off"))

def test_pattern_is_recovered(demo_on):
    r = run_analysis(*demo_on, data_source_label="Demo")
    top = r.ranking.head(12)["feature"].tolist()
    for f in ["R_amplitude_mean", "qRT_interval_mean", "R_rate_mean", "T_amplitude_mean"]:
        assert f in top, f
    assert r.ranking.iloc[0]["p_adj"] < 0.05
    assert r.model_report.status == "ok" and r.model_report.auc_cv > 0.8
    assert r.timeline["risk"].notna().sum() > 0

def test_null_finds_nothing(demo_off):
    r = run_analysis(*demo_off)
    assert (r.ranking["p_adj"] < 0.05).sum() == 0
    assert r.model_report.status == "ok" and 0.35 < r.model_report.auc_cv < 0.65

def test_real_file_with_sparse_biopsies(real_ecg_path, tmp_path):
    b = tmp_path / "b.csv"
    b.write_text("Patient;Biopsie_Nr;Datum;Befund\nMW;1;2024-04-20;nein\nJM239;1;2026-02-01;ja\n")
    r = run_analysis(real_ecg_path, b)
    assert r.model_report.status == "insufficient"
    assert not r.ranking_reliable
    assert len(r.timeline) > 0 and r.timeline["risk"].isna().all()

def test_wrong_file_in_ecg_slot(real_ecg_path, tmp_path):
    b = tmp_path / "b.csv"; b.write_text("Patient;Datum;Befund\nMW;2024-04-20;nein\n")
    with pytest.raises(IngestError):
        run_analysis(b, b)
```

- [ ] **Step 2: Fehlschlag, dann Implementierung**

`src/kardio/pipeline.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from kardio.config import AnalysisConfig, load_analysis_config
from kardio.ingest.ecg import load_ecg, select_daily
from kardio.ingest.biopsies import load_biopsies
from kardio.ingest.report import QualityReport, build_report
from kardio.features.baseline import compute_baselines, compute_deviations
from kardio.features.windows import build_windows, rolling_windows
from kardio.stats.compare import compare_features
from kardio.model.risk import ModelReport, fit_risk_model, score_timeline

@dataclass
class AnalysisResult:
    daily: pd.DataFrame
    dropped: pd.DataFrame
    biopsies: pd.DataFrame
    report: QualityReport
    baselines: pd.DataFrame
    deviations: pd.DataFrame
    windows: pd.DataFrame
    rolling: pd.DataFrame
    ranking: pd.DataFrame
    ranking_reliable: bool
    model_report: ModelReport
    timeline: pd.DataFrame
    data_source_label: str
    cfg: AnalysisConfig

def run_analysis_frames(ecg_raw: pd.DataFrame, biopsies: pd.DataFrame, cfg: AnalysisConfig | None = None,
                        data_source_label: str = "Echte Daten") -> AnalysisResult:
    cfg = cfg or load_analysis_config()
    daily, dropped = select_daily(ecg_raw)
    report = build_report(daily, dropped, biopsies)
    baselines = compute_baselines(daily, biopsies, cfg)
    dev = compute_deviations(daily, baselines, cfg)
    windows = build_windows(dev, biopsies, cfg)
    rolling = rolling_windows(dev, cfg)
    ranking, reliable = compare_features(windows, cfg)
    model_report, model = fit_risk_model(windows, cfg)
    timeline = score_timeline(model, rolling, cfg)
    return AnalysisResult(daily, dropped, biopsies, report, baselines, dev, windows, rolling,
                          ranking, reliable, model_report, timeline, data_source_label, cfg)

def run_analysis(ecg_src, biopsies_src, cfg: AnalysisConfig | None = None, mapping: dict | None = None,
                 data_source_label: str = "Echte Daten") -> AnalysisResult:
    return run_analysis_frames(load_ecg(ecg_src), load_biopsies(biopsies_src, mapping), cfg, data_source_label)

def save_processed(r: AnalysisResult, out_dir: Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["daily", "biopsies", "baselines", "deviations", "windows", "rolling", "ranking", "timeline"]:
        getattr(r, name).to_csv(out_dir / f"{name}.csv", index=False)
```

`src/kardio/run.py`:
```python
import argparse
from pathlib import Path
from kardio.config import load_generator_config
from kardio.generator.demo import write_demo
from kardio.pipeline import run_analysis, save_processed

def main():
    ap = argparse.ArgumentParser(description="Kardio-Analyse ohne Dashboard")
    ap.add_argument("--ecg"); ap.add_argument("--biopsies")
    ap.add_argument("--demo", action="store_true", help="Demo-Daten erzeugen und analysieren")
    ap.add_argument("--no-pattern", action="store_true")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    if a.demo:
        cfg = load_generator_config(); cfg.pattern_enabled = not a.no_pattern
        a.ecg, a.biopsies = write_demo(cfg, Path("data/demo"))
    r = run_analysis(a.ecg, a.biopsies)
    save_processed(r, Path(a.out))
    for k, v in r.report.as_rows(): print(f"{k:32} {v}")
    print("\nTop-Merkmale:"); print(r.ranking.head(10)[["column", "p_adj", "auc", "cliffs_delta"]].to_string(index=False))
    print(f"\nModell: {r.model_report.status} {r.model_report.reason} AUC={r.model_report.auc_cv}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Tests grün**

Run: `.venv/Scripts/pytest -v`
Expected: alle passed. Falls `test_pattern_is_recovered` scheitert, weil das Muster zu schwach ist, Musterstärken in `config/generator.yaml` erhöhen (bis zu ±0,25), nicht die Tests lockern.

- [ ] **Step 4: CLI-Lauf und Commit**

Run: `.venv/Scripts/python -m kardio.run --demo`
```bash
git add -A && git commit -m "feat: Pipeline, CLI und End-to-End-Nachweis"
```

---

### Task 12: Station 6, Erklärtext über MGA mit Vorlagen-Fallback

**Files:**
- Create: `src/kardio/explain/__init__.py`, `src/kardio/explain/text.py`, `tests/test_explain.py`

**Interfaces:**
- Produces: `build_facts(result, patient: str | None) -> dict` (verdichtete Zahlen: Datenlage, Top-5-Merkmale mit p_adj/auc/Richtung, Modellstatus, Risiko des Patienten, Quelle); `template_text(facts) -> str`; `llm_text(facts, client=None) -> str | None` (None bei Fehler/kein Schlüssel/Prüfung verletzt); `explain(result, patient=None) -> tuple[str, str]` (Text, Quelle in `{"llm","template"}`); `verify_numbers(text, facts) -> bool`.

- [ ] **Step 1: Tests**

`tests/test_explain.py`:
```python
import os
import pytest
from kardio.explain.text import template_text, verify_numbers, llm_text, explain
from kardio.pipeline import run_analysis
from kardio.generator.demo import write_demo
from kardio.config import load_generator_config

FACTS = {"source": "Demo", "n_patients": 40, "n_biopsies": 300, "n_rejections": 70,
         "top_features": [{"name": "R_rate_mean", "agg": "zmean", "p_adj": 0.001, "auc": 0.88, "direction": "höher"}],
         "model_status": "ok", "auc_cv": 0.86, "patient": "D001", "risk": 0.72, "light": "red",
         "allowed_numbers": ["40", "300", "70", "0.001", "0.88", "0.86", "72", "0.72"]}

def test_template_mentions_key_facts():
    t = template_text(FACTS)
    assert "40" in t and "R_rate_mean" in t and "72" in t

def test_verify_rejects_invented_number():
    assert verify_numbers("Risiko 72 %, AUC 0.86", FACTS)
    assert not verify_numbers("Risiko 93 %", FACTS)

def test_llm_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("MGA_API_KEY", raising=False)
    assert llm_text(FACTS) is None

def test_llm_with_fake_client_verified():
    class Fake:
        def chat(self, messages): return "Risiko 72 %, AUC 0.86."
    assert llm_text(FACTS, client=Fake()) == "Risiko 72 %, AUC 0.86."
    class Liar:
        def chat(self, messages): return "Risiko 99 %."
    assert llm_text(FACTS, client=Liar()) is None

def test_explain_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv("MGA_API_KEY", raising=False)
    cfg = load_generator_config(); cfg.n_patients = 10; cfg.days = 120
    r = run_analysis(*write_demo(cfg, tmp_path), data_source_label="Demo")
    text, src = explain(r, patient="D001")
    assert src == "template" and len(text) > 100

@pytest.mark.skipif(not os.environ.get("MGA_API_KEY"), reason="kein Schluessel")
def test_live_mga():
    t = llm_text(FACTS)
    assert t is None or len(t) > 50
```

- [ ] **Step 2: Fehlschlag, dann Implementierung**

`src/kardio/explain/text.py`:
```python
from __future__ import annotations
import os, re
import requests
from kardio.model.risk import traffic_light

SYSTEM = ("Du bist ein medizinischer Assistent und schreibst für Kardiologen einen kurzen Befundtext auf Deutsch "
          "(150 bis 250 Wörter) mit den Abschnitten Datenlage, Auffällige Merkmale, Risiko, Einschränkungen. "
          "Nutze ausschließlich die gelieferten Zahlen und Merkmalsnamen. Erfinde keine Werte. Keine Therapieempfehlung.")

class MgaClient:
    def __init__(self, api_key: str, base_url: str | None = None, model: str | None = None):
        self.key = api_key
        self.base = (base_url or os.environ.get("MGA_BASE_URL") or "https://chat.int.bayer.com/api/v2").rstrip("/")
        self.model = model or os.environ.get("MGA_MODEL") or "claude-sonnet-5"
    def chat(self, messages: list[dict]) -> str:
        r = requests.post(f"{self.base}/chat/completions", timeout=60,
                          headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                          json={"model": self.model, "messages": messages, "max_tokens": 700, "temperature": 0.2})
        r.raise_for_status()
        data = r.json()
        if data.get("model") and data["model"] != self.model:
            raise RuntimeError(f"Fallback-Modell {data['model']} statt {self.model}")
        return data["choices"][0]["message"]["content"]

def _fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

def build_facts(result, patient: str | None) -> dict:
    rep, mr = result.report, result.model_report
    top = []
    for _, r in result.ranking.head(5).iterrows():
        if r["p_adj"] != r["p_adj"]:
            continue
        direction = "höher" if r["median_rej"] > r["median_no"] else "niedriger"
        top.append({"name": r["feature"], "agg": r["agg"], "p_adj": round(float(r["p_adj"]), 3),
                    "auc": round(float(r["auc"]), 2), "direction": direction})
    risk = None
    if patient is not None:
        t = result.timeline[result.timeline["patient"] == patient].dropna(subset=["risk"])
        if len(t):
            risk = round(float(t.sort_values("date")["risk"].iloc[-1]), 2)
    facts = {"source": result.data_source_label, "n_patients": rep.n_patients_matched,
             "n_biopsies": rep.n_biopsies, "n_rejections": rep.n_rejections,
             "ranking_reliable": result.ranking_reliable, "top_features": top,
             "model_status": mr.status, "model_reason": mr.reason,
             "auc_cv": round(mr.auc_cv, 2) if mr.auc_cv is not None else None,
             "patient": patient, "risk": risk, "light": traffic_light(risk if risk is not None else float("nan"), result.cfg)}
    nums = [str(facts[k]) for k in ("n_patients", "n_biopsies", "n_rejections") ]
    for f in top:
        nums += [str(f["p_adj"]), str(f["auc"])]
    if facts["auc_cv"] is not None: nums.append(str(facts["auc_cv"]))
    if risk is not None: nums += [str(risk), str(int(round(risk * 100)))]
    facts["allowed_numbers"] = nums
    return facts

def template_text(f: dict) -> str:
    feats = "; ".join(f"{t['name']} ({t['agg']}) vor Abstoßung {t['direction']}, AUC {t['auc']}, p={t['p_adj']}"
                      for t in f["top_features"]) or "keine belastbaren Unterschiede"
    risk = f"{int(round(f['risk']*100))} %" if f.get("risk") is not None else "nicht berechenbar (zu wenig Daten)"
    model = f"AUC {f['auc_cv']} in der Kreuzvalidierung" if f["model_status"] == "ok" else f["model_reason"]
    return (f"Datenlage: {f['source']}, {f['n_patients']} Patienten, {f['n_biopsies']} Biopsien, davon "
            f"{f['n_rejections']} mit Abstoßung.\n\nAuffällige Merkmale: {feats}.\n\n"
            f"Risiko für Patient {f.get('patient') or '-'}: {risk}. Modell: {model}.\n\n"
            f"Einschränkungen: {'Demo-Daten, keine medizinische Aussage. ' if 'Demo' in f['source'] else ''}"
            f"{'Statistik nicht belastbar (zu wenige Abstoßungen). ' if not f.get('ranking_reliable', True) else ''}"
            "Die Analyse ersetzt keine Biopsie und keine ärztliche Beurteilung.")

def verify_numbers(text: str, f: dict) -> bool:
    found = re.findall(r"\d+(?:[.,]\d+)?", text)
    allowed = set(f["allowed_numbers"]) | {str(int(x)) for x in [0, 1, 5, 7, 14, 100, 150, 250]}
    return all(n.replace(",", ".") in allowed or n in allowed for n in found)

def llm_text(f: dict, client=None) -> str | None:
    if client is None:
        key = os.environ.get("MGA_API_KEY")
        if not key:
            return None
        client = MgaClient(key)
    try:
        text = client.chat([{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": "Fakten (JSON):\n" + str({k: v for k, v in f.items() if k != "allowed_numbers"})}])
    except Exception:
        return None
    return text if verify_numbers(text, f) else None

def explain(result, patient: str | None = None) -> tuple[str, str]:
    f = build_facts(result, patient)
    t = llm_text(f)
    return (t, "llm") if t else (template_text(f), "template")
```

`src/kardio/explain/__init__.py`: leer.

- [ ] **Step 3: Tests grün, Commit**

```bash
git add -A && git commit -m "feat: Erklaertext ueber MGA mit Zahlenpruefung und Vorlage"
```

---

### Task 13: Dashboard

**Files:**
- Create: `app/main.py`, `app/ui_upload.py`, `app/ui_overview.py`, `app/ui_features.py`, `app/ui_patient.py`, `app/ui_model.py`, `app/auth.py`, `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `run_analysis`, `run_analysis_frames`, `write_demo`, `explain`, `guess_columns`, `traffic_light`.
- Produces: Streamlit-App; `st.session_state["result"]: AnalysisResult | None`.

- [ ] **Step 1: Smoke-Test mit Streamlit AppTest**

`tests/test_app_smoke.py`:
```python
from streamlit.testing.v1 import AppTest

def test_app_starts_without_password(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    at = AppTest.from_file("app/main.py", default_timeout=60).run()
    assert not at.exception
    assert any("Kardio" in t.value for t in at.title)

def test_password_gate(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "geheim")
    at = AppTest.from_file("app/main.py", default_timeout=60).run()
    assert at.text_input  # Kennwortfeld sichtbar
    at.text_input[0].input("falsch").run()
    assert any("falsch" in e.value.lower() or "kennwort" in e.value.lower() for e in at.error)

def test_demo_button_runs_analysis(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("MGA_API_KEY", raising=False)
    at = AppTest.from_file("app/main.py", default_timeout=300).run()
    demo = [b for b in at.button if "Demo" in b.label][0]
    demo.click().run()
    assert not at.exception
    assert at.session_state["result"] is not None
```

- [ ] **Step 2: Implementierung**

`app/auth.py`:
```python
import os
import streamlit as st

def require_password() -> bool:
    pw = os.environ.get("APP_PASSWORD")
    if not pw:
        return True
    if st.session_state.get("authed"):
        return True
    st.title("Kardio – Anmeldung")
    entered = st.text_input("Kennwort", type="password")
    if entered:
        if entered == pw:
            st.session_state["authed"] = True
            st.rerun()
        st.error("Kennwort falsch.")
    return False
```

`app/main.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import streamlit as st
from app_paths import *  # noqa  (siehe unten, optional)
```

Statt eines Hilfsmoduls direkt:

```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "app"))
import streamlit as st
from auth import require_password
import ui_upload, ui_overview, ui_features, ui_patient, ui_model

st.set_page_config(page_title="Kardio", page_icon="❤️", layout="wide")
if not require_password():
    st.stop()
st.session_state.setdefault("result", None)
st.title("Kardio – EKG-Früherkennung von Abstoßungen")

r = st.session_state["result"]
if r is not None:
    label = r.data_source_label
    kind = "warning" if "Demo" in label else "info"
    getattr(st, kind)(f"Datenbasis: {label} · {r.report.n_patients_matched} Patienten · {r.report.n_biopsies} Biopsien")

tabs = st.tabs(["Upload", "Übersicht", "Merkmale", "Patient", "Modell"])
with tabs[0]: ui_upload.render()
with tabs[1]: ui_overview.render()
with tabs[2]: ui_features.render()
with tabs[3]: ui_patient.render()
with tabs[4]: ui_model.render()
```

`app/ui_upload.py`:
```python
import io
from pathlib import Path
import pandas as pd
import streamlit as st
from kardio.config import load_generator_config, ROOT
from kardio.generator.demo import write_demo
from kardio.ingest.biopsies import guess_columns, load_biopsies
from kardio.ingest.ecg import IngestError, load_ecg
from kardio.pipeline import run_analysis_frames

def _analyze(ecg_df, bx_df, label):
    with st.spinner("Analyse läuft …"):
        st.session_state["result"] = run_analysis_frames(ecg_df, bx_df, data_source_label=label)
    st.success("Analyse fertig. Ergebnisse in den anderen Reitern.")

def render():
    st.subheader("1. Daten laden")
    c1, c2 = st.columns(2)
    ecg_file = c1.file_uploader("EKG-Tabelle (CSV)", type=["csv"])
    bx_file = c2.file_uploader("Biopsie-Tabelle (CSV)", type=["csv"])
    st.markdown("**Oder Demo-Daten verwenden**")
    pattern = st.toggle("Muster eingebaut (Abstoßung kündigt sich im EKG an)", value=True)
    if st.button("Demo-Daten laden und analysieren"):
        cfg = load_generator_config(); cfg.pattern_enabled = pattern
        e, b = write_demo(cfg, ROOT / "data" / "demo")
        label = "DEMO-DATEN mit eingebautem Muster" if pattern else "DEMO-DATEN ohne Muster"
        try:
            _analyze(load_ecg(e), load_biopsies(b), label)
        except IngestError as ex:
            st.error(str(ex))
    if ecg_file and bx_file:
        try:
            ecg_df = load_ecg(io.StringIO(ecg_file.getvalue().decode("utf-8-sig")))
        except IngestError as ex:
            st.error(f"EKG-Tabelle: {ex}"); return
        raw_bx = pd.read_csv(io.StringIO(bx_file.getvalue().decode("utf-8-sig")), sep=None, engine="python")
        raw_bx.columns = [str(c).strip() for c in raw_bx.columns]
        guess = guess_columns(list(raw_bx.columns))
        mapping = guess
        if any(guess[k] is None for k in ("patient", "date", "rejection")):
            st.warning("Spalten der Biopsie-Tabelle konnten nicht alle erkannt werden. Bitte zuordnen:")
            opts = [None] + list(raw_bx.columns)
            mapping = {k: st.selectbox(lbl, opts, index=opts.index(guess[k]) if guess[k] in opts else 0, key=f"map_{k}")
                       for k, lbl in [("patient", "Patientenkürzel"), ("biopsy_no", "Biopsie-Nummer (optional)"),
                                      ("date", "Biopsiedatum"), ("rejection", "Befund")]}
        if st.button("🔴 Analysieren", type="primary"):
            try:
                bx_df = load_biopsies(raw_bx, mapping)
            except IngestError as ex:
                st.error(str(ex)); return
            _analyze(ecg_df, bx_df, f"Echte Daten ({ecg_df['patient'].nunique()} Patienten)")
    r = st.session_state.get("result")
    if r is not None:
        st.subheader("Qualitätsbericht")
        st.table(pd.DataFrame(r.report.as_rows(), columns=["Kennzahl", "Wert"]))
```

`app/ui_overview.py`:
```python
import pandas as pd
import streamlit as st
from kardio.model.risk import traffic_light

COLORS = {"red": "🔴", "amber": "🟡", "green": "🟢", "grey": "⚪"}

def render():
    r = st.session_state.get("result")
    if r is None:
        st.info("Noch keine Analyse. Bitte zuerst Daten laden."); return
    if r.model_report.status != "ok":
        st.warning("Risikomodell: " + r.model_report.reason + " – Ampeln bleiben grau.")
    last = r.timeline.sort_values("date").groupby("patient").tail(1).set_index("patient")
    rows = []
    for pid in sorted(r.daily["patient"].unique()):
        risk = last["risk"].get(pid, float("nan"))
        n_bx = int((r.biopsies["patient"] == pid).sum()); n_rej = int(r.biopsies[(r.biopsies["patient"] == pid)]["rejection"].sum())
        rows.append({"Ampel": COLORS[traffic_light(risk, r.cfg)], "Patient": pid,
                     "Risiko": "-" if pd.isna(risk) else f"{risk:.0%}",
                     "Messtage": int((r.daily["patient"] == pid).sum()), "Biopsien": n_bx, "Abstoßungen": n_rej})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
```

`app/ui_features.py`:
```python
import plotly.express as px
import streamlit as st

def render():
    r = st.session_state.get("result")
    if r is None:
        st.info("Noch keine Analyse."); return
    if not r.ranking_reliable:
        st.warning("Zu wenige Abstoßungen für eine belastbare Statistik. Rangliste nur zur Orientierung.")
    rk = r.ranking.dropna(subset=["p_adj"]).head(20)
    if rk.empty:
        st.info("Keine vergleichbaren Fenster vorhanden."); return
    st.dataframe(rk[["feature", "agg", "p_adj", "auc", "cliffs_delta", "n_rej", "n_no"]].round(3),
                 hide_index=True, use_container_width=True)
    fig = px.bar(rk, x="auc", y="column", orientation="h", title="Trennschärfe (AUC) je Merkmal")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
    st.plotly_chart(fig, use_container_width=True)
    col = st.selectbox("Merkmal für Boxplot", rk["column"].tolist())
    w = r.windows[r.windows["usable"]]
    st.plotly_chart(px.box(w, x="rejection", y=col, points="all",
                           labels={"rejection": "Abstoßung", col: col}), use_container_width=True)
```

`app/ui_patient.py`:
```python
import plotly.graph_objects as go
import streamlit as st
from kardio.explain.text import explain
from kardio.features.baseline import DEV_FEATURES

def render():
    r = st.session_state.get("result")
    if r is None:
        st.info("Noch keine Analyse."); return
    pid = st.selectbox("Patient", sorted(r.daily["patient"].unique()))
    feat = st.selectbox("Merkmal", [f for f in DEV_FEATURES if f.endswith("_mean")])
    d = r.deviations[(r.deviations.patient == pid) & (r.deviations.feature == feat)].sort_values("date")
    b = r.baselines[(r.baselines.patient == pid) & (r.baselines.feature == feat)].iloc[0]
    fig = go.Figure()
    fig.add_hrect(y0=b.center - 2 * b.scale, y1=b.center + 2 * b.scale, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(x=d.date, y=d.value, mode="lines+markers", name=feat))
    for _, bx in r.biopsies[r.biopsies.patient == pid].iterrows():
        fig.add_vline(x=bx.date, line_color="red" if bx.rejection else "gray", line_dash="dash")
    fig.update_layout(title=f"{feat} – {pid} (grün = Normalband, rot = Abstoßung, grau = Biopsie o.B.)")
    st.plotly_chart(fig, use_container_width=True)
    t = r.timeline[r.timeline.patient == pid].sort_values("date")
    if t["risk"].notna().any():
        st.plotly_chart(go.Figure(go.Scatter(x=t.date, y=t.risk, name="Risiko")).update_layout(
            title="Risikoverlauf", yaxis=dict(range=[0, 1])), use_container_width=True)
    else:
        st.info("Kein Risikoverlauf: Modell hat zu wenig Daten.")
    if st.button("Befundtext erzeugen"):
        text, src = explain(r, pid)
        st.caption("Quelle: " + ("KI-Text (geprüft)" if src == "llm" else "Vorlagentext"))
        st.write(text)
```

`app/ui_model.py`:
```python
import streamlit as st

def render():
    r = st.session_state.get("result")
    if r is None:
        st.info("Noch keine Analyse."); return
    m = r.model_report
    if m.status != "ok":
        st.warning(m.reason); return
    c = st.columns(4)
    c[0].metric("AUC (Kreuzvalidierung)", f"{m.auc_cv:.2f}")
    c[1].metric("Sensitivität", f"{m.sensitivity:.0%}")
    c[2].metric("Spezifität", f"{m.specificity:.0%}")
    c[3].metric("Vergleich Boosting AUC", f"{m.auc_gb:.2f}")
    st.caption(f"{m.n_windows} Fenster, davon {m.n_rejections} Abstoßungen. Schwelle rot ab {m.threshold:.0%}.")
    st.dataframe(m.weights.head(15).round(3), hide_index=True, use_container_width=True)
```

- [ ] **Step 3: Tests und manueller Start**

Run: `.venv/Scripts/pytest tests/test_app_smoke.py -v`, dann `start.bat` und im Browser alle fünf Reiter mit Demo-Daten durchklicken.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: Streamlit-Dashboard mit Upload, Ampel, Merkmalen, Patient, Modell"
```

---

### Task 14: Veröffentlichung (GitHub privat, Render)

**Files:**
- Create: `render.yaml`, `README.md`, `.streamlit/config.toml`

- [ ] **Step 1: Render-Blueprint**

`render.yaml`:
```yaml
services:
  - type: web
    name: kardio-ekg
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
    envVars:
      - key: PYTHON_VERSION
        value: "3.12.7"
      - key: MGA_MODEL
        value: claude-sonnet-5
      - key: MGA_API_KEY
        sync: false
      - key: APP_PASSWORD
        sync: false
```

`.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200
[theme]
primaryColor = "#c62828"
```

- [ ] **Step 2: README (kurz, deutsch)**

Inhalt: Zweck in drei Sätzen, Start lokal (`start.bat`), Start Kommandozeile (`python -m kardio.run --demo`), Online-Betrieb (Render, Variablen `MGA_API_KEY`, `APP_PASSWORD`), Hinweis Demo-Daten vs. echte Daten, Verweis auf Spec.

- [ ] **Step 3: Schlüssel-Kontrolle und Push**

```bash
git ls-files -z | xargs -0 grep -lE "mga-[A-Za-z0-9]{8}" | wc -l   # muss 0 sein
export MSYS_NO_PATHCONV=1
gh repo create niclas22359/kardio --private --source=. --push
```

- [ ] **Step 4: Render einrichten (Niclas)**

Dashboard → New → Blueprint → Repo `kardio` → `MGA_API_KEY` und `APP_PASSWORD` eintragen → Apply. Adresse auf der Service-Seite ablesen.

- [ ] **Step 5: Commit und Abschluss**

```bash
git add -A && git commit -m "chore: Render-Blueprint und README" && git push
```

---

## Self-Review

**Spec coverage:** Station 1 (Task 3, 4, 5), Station 2 (6), Station 3 (7), Station 4 (8), Station 5 (9), Station 6 (12), Station 7 (13), Generator (10), CLI und Nachweise 3-5 (11), Nachweis 1-2 (3, 4), Nachweis 6 (12), Render und Repo (14). Mobil-Tauglichkeit: Streamlit-Standard, kein eigener Task. Spaltenzuordnung manuell: Task 13 Upload.

**Placeholder scan:** README in Task 14 ist beschrieben, nicht ausformuliert; akzeptabel, da reiner Text ohne Code. Sonst keine.

**Type consistency:** `DEV_FEATURES` (baseline) wird in windows, patient-UI genutzt; `WINDOW_FEATURE_COLUMNS` in compare, risk, tests; `AnalysisResult.cfg` wird in Task 11 definiert und in 12/13 genutzt. `load_biopsies` akzeptiert DataFrame oder Pfad (Task 4) und wird in Task 13 mit DataFrame gerufen.

**Review Focus:** 1 → Task 8 `test_no_rejections_does_not_crash`, Task 9 `test_insufficient_data`, Task 11 `test_real_file_with_sparse_biopsies`. 2 → Task 3 `test_comma_separator_detected`. 3 → Task 7 `test_window_before_first_measurement`. 4 → Task 3 `test_normalize_patient`, `test_load_small`. 5 → Task 3 `test_wrong_file_raises`, Task 11 `test_wrong_file_in_ecg_slot`.
