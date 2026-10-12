import pandas as pd

from kardio.ingest.biopsies import load_biopsies
from kardio.ingest.ecg import load_ecg, select_daily
from kardio.ingest.report import QualityReport, build_report
from tests.conftest import empty_biopsies, make_biopsies

LABELS = ["Patienten mit EKG", "Patienten mit Biopsien", "Patienten in beiden Tabellen",
          "Nur in EKG-Tabelle", "Nur in Biopsie-Tabelle", "Messtage gesamt", "Davon Tage ohne Werte",
          "Verworfene Mehrfachmessungen", "Biopsien", "Davon Abstoßung", "Davon Zwischengruppe", "Zeitraum"]


def test_report_small(ecg_small_path, biopsies_small_path):
    daily, dropped = select_daily(load_ecg(ecg_small_path))
    r = build_report(daily, dropped, load_biopsies(biopsies_small_path))
    assert r.n_patients_ecg == 2 and r.n_patients_biopsy == 2 and r.n_patients_matched == 2
    assert r.patients_only_ecg == [] and r.patients_only_biopsy == []
    assert r.n_days == 5 and r.n_empty_days == 1 and r.n_dropped_duplicates == 2
    assert r.n_biopsies == 4 and r.n_rejections == 1 and r.n_borderline == 1
    assert r.date_min == pd.Timestamp("2025-01-01") and r.date_max == pd.Timestamp("2025-01-05")
    rows = dict(r.as_rows())
    assert list(rows) == LABELS
    assert rows["Patienten mit EKG"] == "2" and rows["Biopsien"] == "4"
    assert rows["Davon Abstoßung"] == "1" and rows["Davon Zwischengruppe"] == "1"
    assert rows["Nur in EKG-Tabelle"] == "-"
    assert rows["Zeitraum"] == "01.01.2025 bis 05.01.2025"


def test_report_empty_biopsies(ecg_small_path):
    daily, dropped = select_daily(load_ecg(ecg_small_path))
    r = build_report(daily, dropped, empty_biopsies())
    assert r.n_patients_ecg == 2 and r.n_patients_biopsy == 0 and r.n_patients_matched == 0
    assert r.patients_only_ecg == ["AA", "BB"] and r.patients_only_biopsy == []
    assert r.n_biopsies == 0 and r.n_rejections == 0 and r.n_borderline == 0
    assert dict(r.as_rows())["Nur in EKG-Tabelle"] == "AA, BB"


def test_report_empty_loaded_biopsies(ecg_small_path):
    daily, dropped = select_daily(load_ecg(ecg_small_path))
    b = load_biopsies(pd.DataFrame(columns=["Patient", "Datum", "Befund"]))
    r = build_report(daily, dropped, b)
    assert r.n_biopsies == 0 and r.n_patients_biopsy == 0


def test_report_empty_daily(ecg_small_path):
    daily, dropped = select_daily(load_ecg(ecg_small_path))
    b = make_biopsies([("ZZ", 1, "2025-01-02", 2)])
    for empty in (daily.iloc[0:0], pd.DataFrame()):
        r = build_report(empty, dropped.iloc[0:0], b)
        assert r.n_patients_ecg == 0 and r.n_days == 0 and r.n_empty_days == 0
        assert r.n_dropped_duplicates == 0 and r.patients_only_biopsy == ["ZZ"]
        assert r.n_rejections == 1
        assert r.date_min is None and r.date_max is None
        assert dict(r.as_rows())["Zeitraum"] == "-"


def test_report_everything_empty():
    r = build_report(pd.DataFrame(), pd.DataFrame(), empty_biopsies())
    assert r.n_patients_ecg == 0 and r.n_biopsies == 0
    assert len(r.as_rows()) == len(LABELS)


def test_report_unmatched_patients(daily_df):
    b = make_biopsies([("P1", 1, "2025-01-10", 0), ("XX", 1, "2025-01-10", 1)])
    r = build_report(daily_df, daily_df.iloc[0:0], b)
    assert r.n_patients_matched == 1
    assert r.patients_only_ecg == ["P2", "P3"] and r.patients_only_biopsy == ["XX"]
    assert r.n_borderline == 1 and r.n_rejections == 0 and r.n_empty_days == 0


def test_long_patient_lists_are_shortened():
    names = [f"P{i:02d}" for i in range(15)]
    r = QualityReport(n_patients_ecg=15, n_patients_biopsy=0, n_patients_matched=0,
                      patients_only_ecg=names, patients_only_biopsy=[], n_days=0, n_empty_days=0,
                      n_dropped_duplicates=0, n_biopsies=0, n_rejections=0, n_borderline=0,
                      date_min=None, date_max=None)
    rows = dict(r.as_rows())
    assert rows["Nur in EKG-Tabelle"] == "P00, P01, P02 … (+12)"
    assert rows["Nur in Biopsie-Tabelle"] == "-"
    r.patients_only_ecg = names[:4]
    assert dict(r.as_rows())["Nur in EKG-Tabelle"] == "P00, P01, P02, P03"
