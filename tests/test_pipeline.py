"""End-to-End-Nachweise aus der Spezifikation (Abschnitt 7, Punkte 3 bis 5)."""
import pytest
from kardio.config import load_generator_config
from kardio.constants import WINDOW_FEATURE_COLUMNS
from kardio.generator.demo import write_demo
from kardio.ingest.ecg import IngestError
from kardio.pipeline import run_analysis, save_processed

PATTERN_GROUPS = {"R_amplitude", "qRT_interval", "R_rate", "T_amplitude"}


@pytest.fixture(scope="module")
def result_on(tmp_path_factory):
    cfg = load_generator_config()
    return run_analysis(*write_demo(cfg, tmp_path_factory.mktemp("on")),
                        data_source_label="DEMO-DATEN mit eingebautem Muster")


@pytest.fixture(scope="module")
def result_off(tmp_path_factory):
    cfg = load_generator_config()
    cfg.pattern_enabled = False
    return run_analysis(*write_demo(cfg, tmp_path_factory.mktemp("off")),
                        data_source_label="DEMO-DATEN ohne Muster")


def test_pattern_is_recovered(result_on):
    r = result_on
    sig = r.ranking[r.ranking["p_adj"] < 0.05]
    assert len(sig) > 0
    # alle vier eingebauten Merkmalsgruppen werden signifikant gefunden
    assert PATTERN_GROUPS <= set(sig["group"])
    # und die staerksten Treffer sind eingebaute Merkmale
    assert set(r.ranking.head(8)["group"]) <= PATTERN_GROUPS
    assert r.ranking_reliable
    assert r.model_report.status == "ok" and r.model_report.auc_cv > 0.8
    assert r.timeline["risk"].notna().sum() > 0
    assert r.is_demo


def test_null_finds_nothing(result_off):
    r = result_off
    assert (r.ranking["p_adj"] < 0.05).sum() == 0
    assert r.model_report.status == "ok"
    assert 0.35 < r.model_report.auc_cv < 0.65


def test_risk_rises_before_rejection(result_on):
    """Im Mittel liegt das Risiko in den 3 Tagen vor einer Abstossung ueber dem sonstigen Niveau."""
    r = result_on
    t = r.timeline.dropna(subset=["risk"])
    rej = r.biopsies[r.biopsies["grade"] == 2]
    near = []
    for _, b in rej.iterrows():
        m = (t["patient"] == b["patient"]) & (t["date"] <= b["date"]) & \
            (t["date"] >= b["date"] - __import__("pandas").Timedelta(days=3))
        near.extend(t.loc[m, "risk"].tolist())
    assert len(near) > 20
    assert sum(near) / len(near) > t["risk"].median() + 0.2


def test_real_file_with_sparse_biopsies(real_ecg_path, tmp_path):
    """Nur 3 echte Patienten: laeuft durch, meldet ehrlich 'zu wenig Daten'."""
    b = tmp_path / "b.csv"
    b.write_text("Patient;Biopsie_Nr;Datum;Befund\nMW;1;2024-04-20;nein\nJM239;1;2026-02-06;ja\n"
                 "XX;1;2025-01-01;nein\n", encoding="utf-8")
    r = run_analysis(real_ecg_path, b)
    assert r.model_report.status == "insufficient"
    assert not r.ranking_reliable
    assert len(r.ranking) == len(WINDOW_FEATURE_COLUMNS)
    assert len(r.timeline) > 0 and r.timeline["risk"].isna().all()
    assert r.report.patients_only_biopsy == ["XX"]
    assert "MC" in r.report.patients_only_ecg
    assert len(r.windows) == 3


def test_real_file_without_any_biopsy(real_ecg_path, tmp_path):
    b = tmp_path / "b.csv"
    b.write_text("Patient;Biopsie_Nr;Datum;Befund\n", encoding="utf-8")
    r = run_analysis(real_ecg_path, b)
    assert r.model_report.status == "insufficient" and len(r.windows) == 0


def test_wrong_file_in_ecg_slot(tmp_path):
    b = tmp_path / "b.csv"
    b.write_text("Patient;Datum;Befund\nMW;2024-04-20;nein\n", encoding="utf-8")
    with pytest.raises(IngestError):
        run_analysis(b, b)


def test_save_processed(result_on, tmp_path):
    out = save_processed(result_on, tmp_path / "p")
    assert (out / "feature_ranking.csv").exists() and (out / "risk_timeline.csv").exists()


def test_cli_demo(tmp_path, capsys):
    from kardio.run import main
    assert main(["--demo", "--out", str(tmp_path)]) == 0
    assert "Risikomodell" in capsys.readouterr().out
