import io

import numpy as np
import pandas as pd
import pytest

from kardio.ingest.biopsies import guess_columns, load_biopsies, parse_grade
from kardio.ingest.ecg import IngestError

COLS = ["patient", "biopsy_no", "date", "grade", "rejection"]
NOTHING = {"patient": None, "biopsy_no": None, "date": None, "rejection": None}


def ts(*days):
    return [pd.Timestamp(d) for d in days]


# ---------- guess_columns ----------

def test_guess_german():
    m = guess_columns(["Patient", "Biopsie_Nr", "Datum", "Befund"])
    assert m == {"patient": "Patient", "biopsy_no": "Biopsie_Nr", "date": "Datum", "rejection": "Befund"}


def test_guess_english_unordered():
    m = guess_columns(["result", "biopsy number", "id", "date"])
    assert m == {"patient": "id", "biopsy_no": "biopsy number", "date": "date", "rejection": "result"}


def test_guess_missing_returns_none():
    assert guess_columns(["foo", "bar"]) == NOTHING


def test_guess_biopsiedatum_trap():
    m = guess_columns(["Patientenkürzel", "Biopsiedatum", "Abstoßung"])
    assert m == {"patient": "Patientenkürzel", "biopsy_no": None, "date": "Biopsiedatum",
                 "rejection": "Abstoßung"}


def test_guess_biopsie_nr_is_not_date():
    m = guess_columns(["Biopsie_Nr", "Biopsiedatum", "Kuerzel", "Ergebnis"])
    assert m == {"patient": "Kuerzel", "biopsy_no": "Biopsie_Nr", "date": "Biopsiedatum",
                 "rejection": "Ergebnis"}


def test_guess_substring_date_before_biopsy_no():
    m = guess_columns(["Pseudonym", "Datum der Biopsie", "ISHLT Grad"])
    assert m == {"patient": "Pseudonym", "biopsy_no": None, "date": "Datum der Biopsie",
                 "rejection": "ISHLT Grad"}


@pytest.mark.parametrize("col,key", [
    ("  PATIENT ", "patient"), ("Patient_ID", "patient"), ("Name", "patient"), ("ID", "patient"),
    ("Biopsienummer", "biopsy_no"), ("Biopsy-No", "biopsy_no"), ("Nr", "biopsy_no"), ("Nr.", "biopsy_no"),
    ("DATE", "date"), ("biopsy_date", "date"),
    ("Abstossung", "rejection"), ("Rejection", "rejection"), ("grade", "rejection"),
    ("ISHLT", "rejection"), ("ACR", "rejection"), ("Biopsiebefund", "rejection"),
])
def test_guess_single_alias(col, key):
    m = guess_columns([col])
    assert m[key] == col
    assert [k for k, v in m.items() if v] == [key]


def test_guess_short_alias_needs_whole_word():
    # "id" steckt in "Hybrid", "nr" in "Unruhe": darf nicht treffen
    assert guess_columns(["Hybrid", "Unruhe"]) == NOTHING


def test_guess_never_assigns_column_twice():
    m = guess_columns(["Biopsie", "Name"])
    vals = [v for v in m.values() if v]
    assert len(vals) == len(set(vals))


# ---------- parse_grade ----------

@pytest.mark.parametrize("v,exp", [
    (0, 0), (1, 1), (2, 2), (3, 2), (1.0, 1), (0.0, 0), ("1.0", 1), ("2,0", 2), ("3", 2), (" 0 ", 0),
    (np.int64(2), 2), (np.float64(0.0), 0),
    ("nein", 0), ("Nein", 0), ("NO", 0), ("n", 0), ("false", 0), ("negativ", 0), ("negative", 0),
    ("keine", 0), ("keine Abstoßung", 0), ("keine Abstossung", 0), ("none", 0), ("no rejection", 0),
    ("0R", 0),
    ("zwischen", 1), ("Zwischengruppe", 1), ("borderline", 1), ("grenzwertig", 1), ("mild", 1),
    ("leicht", 1), ("1R", 1),
    ("ja", 2), ("yes", 2), ("y", 2), ("J", 2), ("true", 2), ("positiv", 2), ("positive", 2),
    ("Abstoßung", 2), ("Abstossung", 2), ("rejection", 2), ("2R", 2), ("3R", 2),
    ("kein Hinweis auf Abstoßung", 0), ("no signs of rejection", 0),
    ("negativ für Abstoßung", 0), ("ohne Abstoßung", 0),
    ("Zwischenbefund", 1), ("Borderline-Befund", 1),
    ("akute Abstoßung", 2), ("acute rejection", 2), ("Befund positiv", 2), ("ISHLT 2R", 2),
    ("Grad 1R", 1),
    (True, 2), (False, 0), (np.True_, 2), (np.False_, 0),
    ("", None), ("   ", None), (None, None), (float("nan"), None), (pd.NA, None), (pd.NaT, None),
    ("unklar", None), ("?", None), (7, None), (-1, None), (1.5, None), ("nan", None),
])
def test_parse_grade(v, exp):
    got = parse_grade(v)
    assert got == exp and (got is None or type(got) is int)


# ---------- load_biopsies ----------

def test_load_small(biopsies_small_path):
    b = load_biopsies(biopsies_small_path)
    assert list(b.columns) == COLS
    assert b["patient"].tolist() == ["AA", "AA", "BB", "BB"]
    assert b["biopsy_no"].tolist() == [1, 2, 1, 2]
    assert b["date"].tolist() == ts("2025-01-02", "2025-01-05", "2025-01-05", "2025-01-08")
    assert b["grade"].tolist() == [0, 2, 0, 1]
    assert b["rejection"].tolist() == [False, True, False, False]
    assert b["date"].dtype.kind == "M"
    assert b["biopsy_no"].dtype.kind == "i" and b["grade"].dtype.kind == "i"
    assert b["rejection"].dtype == bool
    assert list(b.index) == [0, 1, 2, 3]


def test_load_variant(biopsies_variant_path):
    b = load_biopsies(biopsies_variant_path)
    assert list(b.columns) == COLS
    assert b["patient"].tolist() == ["AA", "AA"]
    assert b["biopsy_no"].tolist() == [1, 2]
    assert b["date"].tolist() == ts("2025-01-02", "2025-01-05")
    # Spalte enthaelt nur 0/1 -> 1 heisst "ja, Abstossung"
    assert b["grade"].tolist() == [0, 2]
    assert b["rejection"].tolist() == [False, True]


def test_zero_one_two_column_keeps_one_as_borderline():
    raw = pd.DataFrame({"Patient": ["AA"] * 3, "Datum": ["2025-01-01", "2025-01-02", "2025-01-03"],
                        "Befund": [0, 1, 2]})
    b = load_biopsies(raw)
    assert b["grade"].tolist() == [0, 1, 2]
    assert b["rejection"].tolist() == [False, False, True]


@pytest.mark.parametrize("values", [[0, 1], ["0", "1"], [0.0, 1.0], ["0", "1.0"]])
def test_zero_one_only_column_means_yes_no(values):
    raw = pd.DataFrame({"Patient": ["AA"] * 2, "Datum": ["2025-01-01", "2025-01-02"], "Befund": values})
    assert load_biopsies(raw)["grade"].tolist() == [0, 2]


def test_one_next_to_words_stays_borderline():
    raw = pd.DataFrame({"Patient": ["AA"] * 2, "Datum": ["2025-01-01", "2025-01-02"],
                        "Befund": ["1", "nein"]})
    assert load_biopsies(raw)["grade"].tolist() == [1, 0]


def test_iso_date_not_swapped():
    raw = pd.DataFrame({"Patient": ["AA", "AA", "AA"],
                        "Datum": ["2025-01-02", "2025-02-01", "2025-03-04 10:30:00"],
                        "Befund": ["nein", "nein", "ja"]})
    b = load_biopsies(raw)
    assert b["date"].tolist() == ts("2025-01-02", "2025-02-01", "2025-03-04")


def test_mixed_date_formats_day_first():
    raw = pd.DataFrame({"Patient": ["AA"] * 5,
                        "Datum": ["02.01.2025", "05/01/2025", "2025-01-07", "9.1.25", "13-01-2025"],
                        "Befund": ["nein"] * 5})
    b = load_biopsies(raw)
    assert b["date"].tolist() == ts("2025-01-02", "2025-01-05", "2025-01-07", "2025-01-09", "2025-01-13")


def test_datetime_typed_column_is_normalized():
    raw = pd.DataFrame({"Patient": ["AA", "AA"],
                        "Datum": pd.to_datetime(["2025-01-02 14:30", "2025-02-01 08:00"]),
                        "Befund": ["nein", "ja"]})
    b = load_biopsies(raw)
    assert b["date"].tolist() == ts("2025-01-02", "2025-02-01")


def test_unparseable_date_raises():
    raw = pd.DataFrame({"Patient": ["AA", "AA"], "Tag der Biopsie": ["2025-01-02", "irgendwann"],
                        "Befund": ["nein", "ja"]})
    with pytest.raises(IngestError) as e:
        load_biopsies(raw, mapping={"patient": "Patient", "date": "Tag der Biopsie",
                                    "rejection": "Befund", "biopsy_no": None})
    assert "Tag der Biopsie" in str(e.value) and "irgendwann" in str(e.value)


def test_impossible_date_raises():
    raw = pd.DataFrame({"Patient": ["AA"], "Datum": ["31.02.2025"], "Befund": ["nein"]})
    with pytest.raises(IngestError) as e:
        load_biopsies(raw)
    assert "31.02.2025" in str(e.value)


def test_missing_all_columns_raises(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("foo;bar\n1;2\n", encoding="utf-8")
    with pytest.raises(IngestError) as e:
        load_biopsies(p)
    assert str(e.value) == "Biopsie-Tabelle: Spalte nicht gefunden für: Patient, Datum, Befund"


def test_missing_one_column_raises():
    raw = pd.DataFrame({"Patient": ["AA"], "Befund": ["ja"]})
    with pytest.raises(IngestError) as e:
        load_biopsies(raw)
    assert str(e.value) == "Biopsie-Tabelle: Spalte nicht gefunden für: Datum"


def test_unknown_finding_raises_with_examples():
    raw = pd.DataFrame({"Patient": ["AA"] * 5, "Datum": ["2025-01-01"] * 5,
                        "Befund": ["nein", "unklar", "??", "tbd", "offen"]})
    with pytest.raises(IngestError) as e:
        load_biopsies(raw)
    msg = str(e.value)
    assert "unklar" in msg and "??" in msg and "tbd" in msg and "offen" not in msg


def test_manual_mapping_without_biopsy_no(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("x;y;z\nAA;2025-01-05;ja\nAA;2025-01-02;nein\nBB;2025-01-03;nein\n", encoding="utf-8")
    b = load_biopsies(p, mapping={"patient": "x", "date": "y", "rejection": "z", "biopsy_no": None})
    assert b["patient"].tolist() == ["AA", "AA", "BB"]
    assert b["date"].tolist() == ts("2025-01-02", "2025-01-05", "2025-01-03")
    assert b["biopsy_no"].tolist() == [1, 2, 1]
    assert b["grade"].tolist() == [0, 2, 0]


def test_mapping_overrides_guess():
    raw = pd.DataFrame({"Patient": ["falsch"], "Kind": ["AA"], "Datum": ["2025-01-02"], "Befund": ["ja"]})
    b = load_biopsies(raw, mapping={"patient": "Kind", "date": "Datum", "rejection": "Befund"})
    assert b["patient"].tolist() == ["AA"]


def test_mapping_to_unknown_column_raises():
    raw = pd.DataFrame({"Patient": ["AA"], "Datum": ["2025-01-02"], "Befund": ["ja"]})
    with pytest.raises(IngestError) as e:
        load_biopsies(raw, mapping={"patient": "gibtsnicht", "date": "Datum", "rejection": "Befund"})
    assert "gibtsnicht" in str(e.value)


def test_unparseable_biopsy_no_is_renumbered():
    raw = pd.DataFrame({"Patient": ["AA", "AA", "BB"], "Nr": ["zwei", "1", "5"],
                        "Datum": ["2025-01-05", "2025-01-02", "2025-01-03"], "Befund": ["ja", "nein", "nein"]})
    b = load_biopsies(raw)
    assert b["biopsy_no"].tolist() == [1, 2, 1]


def test_dataframe_input_is_not_modified():
    raw = pd.DataFrame({" Patient ": ["AA"], "Datum": ["02.01.2025"], "Befund": ["ja"]})
    before = raw.copy()
    b = load_biopsies(raw)
    assert b["date"].tolist() == ts("2025-01-02") and b["grade"].tolist() == [2]
    pd.testing.assert_frame_equal(raw, before)


def test_buffer_input():
    buf = io.StringIO("Patient;Datum;Befund\nAA;02.01.2025;nein\n")
    b = load_biopsies(buf)
    assert b["patient"].tolist() == ["AA"] and b["date"].tolist() == ts("2025-01-02")


def test_empty_table(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("Patient;Biopsie_Nr;Datum;Befund\n", encoding="utf-8")
    for src in (p, pd.DataFrame(columns=["Patient", "Datum", "Befund"])):
        b = load_biopsies(src)
        assert list(b.columns) == COLS and len(b) == 0
        assert b["date"].dtype.kind == "M" and b["biopsy_no"].dtype.kind == "i"
        assert b["grade"].dtype.kind == "i" and b["rejection"].dtype == bool


def test_patient_spelling_normalized():
    raw = pd.DataFrame({"Patient": [" aa ", "AA", "Aa"], "Datum": ["2025-01-03", "2025-01-01", "2025-01-02"],
                        "Befund": ["nein", "nein", "ja"]})
    b = load_biopsies(raw)
    assert b["patient"].tolist() == ["AA", "AA", "AA"]
    assert b["biopsy_no"].tolist() == [1, 2, 3]
    assert b["grade"].tolist() == [0, 2, 0]


def test_empty_rows_and_empty_patients_dropped(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("Patient;Biopsie_Nr;Datum;Befund\nAA;1;2025-01-02;nein\n;;;\n;2;2025-01-05;ja\n"
                 "  ;3;2025-01-06;ja\nBB;1;2025-01-05;ja\n", encoding="utf-8")
    b = load_biopsies(p)
    assert b["patient"].tolist() == ["AA", "BB"]
    assert b["biopsy_no"].tolist() == [1, 1]
    assert b["grade"].tolist() == [0, 2]


def test_numeric_patient_ids_survive_empty_rows(tmp_path):
    # Leere Zeile macht die Spalte zur Kommazahl; aus Patient 7 darf nicht "7.0" werden
    p = tmp_path / "b.csv"
    p.write_text("Patient;Datum;Befund\n7;2025-01-02;nein\n;;\n12;2025-01-05;ja\n", encoding="utf-8")
    assert load_biopsies(p)["patient"].tolist() == ["12", "7"]


@pytest.mark.parametrize("value,expected", [
    ("Abstoßung: nein", 0), ("nein (keine Abstoßung)", 0), ("not rejected", 0), ("Rejection: no", 0),
    ("Abstossung negativ", 0), ("Abstoßung: ja", 2), ("rejection", 2), ("Abstoßung", 2),
    ("Zwischengruppe", 1),
])
def test_negation_anywhere_in_finding(value, expected):
    from kardio.ingest.biopsies import parse_grade
    assert parse_grade(value) == expected
