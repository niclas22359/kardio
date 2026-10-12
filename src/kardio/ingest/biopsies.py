"""Station 1b: Biopsie-Tabelle einlesen. Spaltennamen, Reihenfolge, Sprache und Schreibweise duerfen variieren."""
from __future__ import annotations

import datetime as dt
import numbers
import re
import warnings

import pandas as pd

from kardio.constants import COL_BIOPSY_NO, COL_DATE, COL_GRADE, COL_PATIENT, COL_REJECTION
from kardio.ingest.ecg import IngestError, normalize_patient, read_table

OUTPUT_COLUMNS = [COL_PATIENT, COL_BIOPSY_NO, COL_DATE, COL_GRADE, COL_REJECTION]
REQUIRED_NAMES = {"patient": "Patient", "date": "Datum", "rejection": "Befund"}

# Reihenfolge innerhalb einer Liste = Vorrang. Schreibweise wie nach _norm (klein, ae/oe/ue/ss).
ALIASES = {
    "patient": ["patient", "patienten", "patientenkuerzel", "kuerzel", "patient id", "patientid", "pat id",
                "name", "id", "pseudonym", "code", "initialen", "proband"],
    "biopsy_no": ["biopsie nr", "biopsienummer", "biopsy no", "biopsy number", "biopsie", "biopsy",
                  "nr", "nummer", "number", "no", "lfd nr"],
    "date": ["datum", "biopsiedatum", "date", "biopsy date", "entnahmedatum", "tag"],
    "rejection": ["befund", "abstossung", "rejection", "result", "ergebnis", "grade", "grad", "ishlt",
                  "acr", "label", "outcome", "diagnose", "abstoss", "reject"],
}
# Exakte Treffer in dieser Reihenfolge; Teiltreffer mit der Biopsienummer zuletzt, damit
# "Datum der Biopsie" oder "Biopsiebefund" nicht als Nummer durchgehen.
EXACT_ORDER = ["patient", "biopsy_no", "date", "rejection"]
PARTIAL_ORDER = ["patient", "date", "rejection", "biopsy_no"]
MIN_SUBSTRING_LEN = 4  # kuerzere Kuerzel (id, nr, no, tag, acr) muessen als ganzes Wort vorkommen

GRADE_WORDS = {
    0: {"nein", "no", "n", "false", "negativ", "negative", "neg", "keine", "keine abstossung",
        "none", "no rejection", "0r"},
    1: {"zwischen", "zwischengruppe", "borderline", "grenzwertig", "mild", "leicht", "1r"},
    2: {"ja", "yes", "y", "j", "true", "positiv", "positive", "pos", "abstossung", "rejection", "2r", "3r"},
}
NUMBER_TO_GRADE = {0: 0, 1: 1, 2: 2, 3: 2}
ISHLT_RE = re.compile(r"\b([0-3])\s*r\b")

ISO_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
DAY_FIRST_RE = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})(?!\d)")
COMPACT_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def _norm(c) -> str:
    s = str(c).strip().lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _matches(alias: str, col: str, exact: bool) -> bool:
    if exact:
        return alias.replace(" ", "") == col.replace(" ", "")
    if len(alias.replace(" ", "")) < MIN_SUBSTRING_LEN:
        return alias in col.split()
    return alias.replace(" ", "") in col.replace(" ", "")


def guess_columns(columns: list[str]) -> dict[str, str | None]:
    """Spalten der Klinik-Tabelle den vier Bedeutungen zuordnen. Exakte Treffer vor Teiltreffern."""
    normed = [(c, _norm(c)) for c in columns]
    out: dict[str, str | None] = {k: None for k in EXACT_ORDER}
    used: set = set()
    for exact, order in ((True, EXACT_ORDER), (False, PARTIAL_ORDER)):
        for key in order:
            if out[key] is not None:
                continue
            hit = next((orig for alias in ALIASES[key] for orig, n in normed
                        if orig not in used and n and _matches(alias, n, exact)), None)
            if hit is not None:
                out[key] = hit
                used.add(hit)
    return out


def _is_missing(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _is_bool(v) -> bool:
    return isinstance(v, bool) or type(v).__name__ in ("bool_", "bool")


def _as_number(v) -> float | None:
    """Zahl oder Zahl als Text ("1", "1.0", "2,0") -> float, sonst None. Wahr/Falsch zaehlt nicht."""
    if _is_missing(v) or _is_bool(v):
        return None
    if isinstance(v, numbers.Real):
        f = float(v)
    else:
        try:
            f = float(str(v).strip().replace(",", "."))
        except ValueError:
            return None
    return f if f == f and abs(f) != float("inf") else None


def _cell_text(v) -> str:
    """Zelleninhalt als Text; leer bei fehlendem Wert; 7.0 -> "7"."""
    if _is_missing(v):
        return ""
    if isinstance(v, numbers.Real) and not _is_bool(v) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


NEGATION_RE = re.compile(r"\b(nein|no|not|none|kein|keine|keiner|keinen|ohne|negativ|negative|neg)\b")


def parse_grade(v) -> int | None:
    """Befund -> 0 keine Abstossung, 1 Zwischengruppe, 2 Abstossung. Unbekannt/leer -> None."""
    if _is_missing(v):
        return None
    if _is_bool(v):
        return 2 if bool(v) else 0
    num = _as_number(v)
    if num is not None:
        return NUMBER_TO_GRADE.get(num)
    s = re.sub(r"\s+", " ", str(v).strip().lower().replace("ß", "ss"))
    if s in ("", "nan"):
        return None
    for grade in (0, 1, 2):
        if s in GRADE_WORDS[grade]:
            return grade
    m = ISHLT_RE.search(s)  # z.B. "ISHLT 2R", "Grad 1R"
    if m:
        return NUMBER_TO_GRADE[int(m.group(1))]
    # Verneinungen zuerst, sonst wuerde "negativ fuer Abstossung" als Abstossung gelesen
    if s.startswith(("kein", "no ", "neg", "ohne")):
        return 0
    # ... auch mitten im Text: "Abstossung: nein", "nein (keine Abstossung)", "not rejected", "rejection: no"
    if NEGATION_RE.search(s):
        return 0
    if "zwischen" in s or "border" in s:
        return 1
    if "abst" in s or "reject" in s or "pos" in s:
        return 2
    return None


def _timestamp(year: int, month: int, day: int):
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT


def _parse_date_value(v):
    """Erst Jahr-Monat-Tag (2025-01-02 ist der 2. Januar), sonst Tag zuerst (02.01.2025, 05/01/2025)."""
    if _is_missing(v):
        return pd.NaT
    if isinstance(v, (dt.datetime, dt.date)):
        return pd.Timestamp(v)
    s = _cell_text(v)
    m = ISO_RE.match(s) or COMPACT_RE.match(s)
    if m:
        return _timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = DAY_FIRST_RE.match(s)
    if m:
        year = int(m.group(3))
        return _timestamp(year + 2000 if year < 100 else year, int(m.group(2)), int(m.group(1)))
    if re.search(r"[a-zA-Z]", s):  # ausgeschriebene Monate wie "2 Jan 2025"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
    return pd.NaT


def _parse_dates(col: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(col):
        dates = col
    else:
        dates = pd.Series(pd.to_datetime([_parse_date_value(v) for v in col], errors="coerce"),
                          index=col.index, dtype="datetime64[ns]")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    if dates.isna().any():
        example = _cell_text(col[dates.isna()].iloc[0]) or "(leer)"
        raise IngestError(f"Biopsie-Tabelle: Datum nicht lesbar in Spalte '{name}', z.B. '{example}' "
                          "(erwartet z.B. 02.01.2025 oder 2025-01-02)")
    return dates.dt.normalize().astype("datetime64[ns]")


def _parse_grades(col: pd.Series) -> pd.Series:
    grades = col.map(parse_grade)
    if grades.isna().any():
        bad = list(dict.fromkeys(_cell_text(v) or "(leer)" for v in col[grades.isna()]))[:3]
        raise IngestError("Biopsie-Tabelle: Befund nicht verstanden: " + ", ".join(bad))
    grades = grades.astype(int)
    # Mehrdeutigkeit 0/1: Stehen in der Spalte ausschliesslich die Zahlen 0 und 1 (keine 2/3, keine
    # Worte), dann nutzt die Klinik 0/1 als nein/ja. Die 1 bedeutet dann Abstossung (Stufe 2).
    # Sobald eine 2/3 oder ein Wort vorkommt, gilt das Stufenschema und 1 ist die Zwischengruppe.
    nums = [_as_number(v) for v in col]
    if nums and all(n in (0, 1) for n in nums):
        grades = grades.replace({1: 2})
    return grades


def _resolve_mapping(columns: list[str], mapping: dict | None) -> dict[str, str | None]:
    if mapping is None:
        return guess_columns(columns)
    m = {k: (str(v).strip() if v is not None and str(v).strip() else None) for k, v in mapping.items()}
    unknown = [v for v in m.values() if v is not None and v not in columns]
    if unknown:
        raise IngestError("Biopsie-Tabelle: gewählte Spalte gibt es nicht: " + ", ".join(unknown))
    # Was die Zuordnung offen laesst, wird aus den uebrigen Spalten geraten
    rest = guess_columns([c for c in columns if c not in m.values()])
    return {k: (m[k] if k in m else rest[k]) for k in EXACT_ORDER}


def _empty() -> pd.DataFrame:
    return pd.DataFrame({COL_PATIENT: pd.Series(dtype=object), COL_BIOPSY_NO: pd.Series(dtype="int64"),
                         COL_DATE: pd.Series(dtype="datetime64[ns]"), COL_GRADE: pd.Series(dtype="int64"),
                         COL_REJECTION: pd.Series(dtype=bool)})


def load_biopsies(src, mapping: dict | None = None) -> pd.DataFrame:
    """Pfad, Datei-Puffer oder bereits geladene Rohtabelle -> patient, biopsy_no, date, grade, rejection."""
    raw = src.copy() if isinstance(src, pd.DataFrame) else read_table(src)
    raw.columns = [str(c).strip() for c in raw.columns]
    m = _resolve_mapping(list(raw.columns), mapping)
    missing = [k for k in REQUIRED_NAMES if not m.get(k)]
    if missing:
        raise IngestError("Biopsie-Tabelle: Spalte nicht gefunden für: "
                          + ", ".join(REQUIRED_NAMES[k] for k in missing))

    # Zeilen ohne Patient fallen weg; das schliesst komplett leere Zeilen ein
    patient_text = raw[m["patient"]].map(_cell_text)
    raw = raw[patient_text != ""]
    if raw.empty:
        return _empty()

    out = pd.DataFrame(index=raw.index)
    out[COL_PATIENT] = patient_text[raw.index].map(normalize_patient)
    out[COL_DATE] = _parse_dates(raw[m["date"]], m["date"])
    out[COL_GRADE] = _parse_grades(raw[m["rejection"]])
    out[COL_REJECTION] = out[COL_GRADE] == 2
    number = None
    if m.get("biopsy_no"):
        number = pd.to_numeric(raw[m["biopsy_no"]], errors="coerce")
        if number.isna().any() or (number % 1 != 0).any():
            number = None
    out = out.sort_values([COL_PATIENT, COL_DATE], kind="mergesort")
    # Ohne brauchbare Nummer: je Patient nach Datum durchzaehlen
    out[COL_BIOPSY_NO] = number[out.index] if number is not None else out.groupby(COL_PATIENT).cumcount() + 1
    out[COL_BIOPSY_NO] = out[COL_BIOPSY_NO].astype("int64")
    out[COL_GRADE] = out[COL_GRADE].astype("int64")
    return out[OUTPUT_COLUMNS].reset_index(drop=True)
