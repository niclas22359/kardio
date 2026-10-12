import io
import pandas as pd
import pytest
from kardio.ingest.ecg import load_ecg, select_daily, normalize_patient, IngestError
from kardio.constants import VALUE_COLUMNS, FEATURE_GROUPS, STATS, DEV_FEATURES, WINDOW_FEATURE_COLUMNS
from kardio.config import load_analysis_config, load_generator_config


def test_constants_and_config():
    assert len(FEATURE_GROUPS) == 11 and STATS == ["mean", "sd", "min", "max"]
    assert len(VALUE_COLUMNS) == 44 and len(DEV_FEATURES) == 33 and len(WINDOW_FEATURE_COLUMNS) == 132
    assert load_analysis_config().window_days == 14
    g = load_generator_config()
    assert g.rejection_rate == 0.30 and g.pattern["R_amplitude_mean"] == -0.15


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
    assert aa1["included_no"].iloc[0] == 35  # vollstaendig + meiste Herzschlaege
    assert len(dropped) == 2
    aa3 = daily[(daily["patient"] == "AA") & (daily["date"] == "2025-01-03")]
    assert len(aa3) == 1 and aa3[VALUE_COLUMNS].isna().all(axis=None)  # leerer Tag bleibt als Luecke


def test_select_daily_real_mc(real_ecg_path):
    daily, dropped = select_daily(load_ecg(real_ecg_path))
    assert (daily["patient"] == "MC").sum() == 1
    assert len(dropped) == 41


def test_comma_separator_detected():
    txt = "Name,Date,included_no,cc," + ",".join(VALUE_COLUMNS) + "\n" \
          "AA,ecg_2025-01-01.csv,30,0.95," + ",".join(["1.5"] * 44) + "\n"
    df = load_ecg(io.StringIO(txt))
    assert len(df) == 1 and df["R_rate_mean"].iloc[0] == 1.5


def test_decimal_comma_with_semicolon():
    txt = "Name;Date;included_no;cc;" + ";".join(VALUE_COLUMNS) + "\n" \
          "AA;ecg_2025-01-01.csv;30;0,95;" + ";".join(["1,5"] * 44) + "\n"
    df = load_ecg(io.StringIO(txt))
    assert df["R_rate_mean"].iloc[0] == 1.5


def test_wrong_file_raises(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("Patient;Biopsie_Nr;Datum;Befund\nAA;1;2025-01-02;nein\n", encoding="utf-8")
    with pytest.raises(IngestError) as e:
        load_ecg(p)
    assert "Name" in str(e.value)


def test_patient_code_na_is_kept():
    txt = "Name;Date;included_no;cc;" + ";".join(VALUE_COLUMNS) + "\n" \
          "NA;ecg_2025-01-01.csv;30;0.95;" + ";".join(["1.5"] * 44) + "\n"
    df = load_ecg(io.StringIO(txt))
    assert df["patient"].tolist() == ["NA"]


def test_row_limit_at_the_entrance(monkeypatch):
    import kardio.ingest.ecg as ecg_mod
    monkeypatch.setattr(ecg_mod, "MAX_ROWS", 3)
    txt = "Name;Date;included_no;cc;" + ";".join(VALUE_COLUMNS) + "\n" + \
          "".join(f"AA;ecg_2025-01-0{i}.csv;30;0.95;" + ";".join(["1.5"] * 44) + "\n" for i in range(1, 6))
    with pytest.raises(IngestError) as e:
        load_ecg(io.StringIO(txt))
    assert "höchstens 3" in str(e.value)
