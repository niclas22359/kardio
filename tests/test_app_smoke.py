"""Rauchtest des Dashboards: startet, Kennwortabfrage, Demo-Lauf, alle Reiter mit echten Daten."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ui_common  # noqa: E402
import ui_features  # noqa: E402
import ui_upload  # noqa: E402

MAIN = str(ROOT / "app" / "main.py")
REAL_ECG = ROOT / "data" / "raw" / "ecg_real.csv"
needs_real = pytest.mark.skipif(not REAL_ECG.exists(), reason="echte Patientendaten liegen nur lokal")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Kein Kennwort, kein KI-Schluessel; was die App aus .env nachlaedt, wird danach entfernt."""
    before = dict(os.environ)
    # Leer statt geloescht: load_dotenv ueberschreibt vorhandene Variablen nicht
    monkeypatch.setenv("APP_PASSWORD", "")
    monkeypatch.setenv("MGA_API_KEY", "")
    monkeypatch.chdir(ROOT)
    yield
    for key in set(os.environ) - set(before):
        os.environ.pop(key, None)


def new_app(timeout: int = 300) -> AppTest:
    return AppTest.from_file(MAIN, default_timeout=timeout)


def all_text(at: AppTest) -> str:
    parts = []
    for kind in ("title", "warning", "info", "error", "success", "markdown", "caption", "subheader"):
        parts += [str(e.value) for e in getattr(at, kind)]
    return "\n".join(parts)


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def tiny_biopsies() -> pd.DataFrame:
    return pd.DataFrame({"Patient": ["MW", "MW", "MC", "MC"],
                         "Datum": ["10.04.2024", "20.04.2024", "01.02.2025", "01.03.2025"],
                         "Befund": ["nein", "ja", "nein", "Zwischengruppe"]})


# ---------- Start und Kennwort ----------

def test_app_starts_without_password():
    at = new_app().run()
    assert_clean(at)
    assert "Kardio" in at.title[0].value
    assert len(at.tabs) == 6
    assert at.session_state["result"] is None
    assert ui_common.EMPTY_TEXT in all_text(at)
    assert at.button(key="btn_analyse").disabled


def test_password_gate(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "geheim")
    at = new_app().run()
    assert not at.exception
    assert len(at.tabs) == 0
    assert at.text_input[0].label == "Kennwort"

    at.text_input[0].set_value("falsch").run()
    assert [e.value for e in at.error] == ["Kennwort falsch."]
    assert len(at.tabs) == 0

    at.text_input[0].set_value("geheim").run()
    assert_clean(at)
    assert len(at.tabs) == 6


# ---------- Demo-Lauf ----------

def test_demo_with_pattern_runs_analysis():
    at = new_app().run()
    at.button(key="btn_demo").click().run()
    assert_clean(at)
    result = at.session_state["result"]
    assert result is not None and result.is_demo
    banner = "\n".join(w.value for w in at.warning)
    assert "DEMO-DATEN mit eingebautem Muster" in banner
    assert "Keine medizinische Aussage" in banner
    assert result.model_report.status == "ok"
    # Widgets aendern darf die Analyse nicht neu starten
    at.selectbox(key="feat_metric").set_value("cv").run()
    assert_clean(at)
    assert at.session_state["result"] is result


def test_demo_without_pattern():
    at = new_app().run()
    at.toggle(key="demo_pattern").set_value(False).run()
    at.button(key="btn_demo").click().run()
    assert_clean(at)
    assert at.session_state["result"] is not None
    banner = "\n".join(w.value for w in at.warning)
    assert "ohne Muster" in banner and "DEMO" in banner


# ---------- echte Datei, Modell nicht moeglich ----------

@needs_real
def test_real_data_renders_all_tabs():
    ecg, err = ui_upload.check_ecg(REAL_ECG.read_bytes())
    assert err is None
    biopsies, err = ui_upload.check_biopsies(tiny_biopsies(), None)
    assert err is None
    result = ui_upload.run_real(ecg, biopsies)
    assert result.model_report.status == "insufficient"
    assert result.data_source_label == "Hochgeladene Daten (3 Patienten)"

    at = new_app()
    at.session_state["result"] = result
    at.run()
    assert_clean(at)
    assert any("Hochgeladene Daten (3 Patienten)" in i.value for i in at.info)
    assert not any("DEMO" in w.value for w in at.warning)
    text = all_text(at)
    assert "Kein Risikomodell möglich" in text
    assert "Zu wenige Abstoßungen für eine belastbare Statistik" in text
    assert "Grenzen dieser Auswertung" in text
    for patient in ("MC", "JM239"):
        at.selectbox(key="pat_patient").set_value(patient).run()
        assert_clean(at)


@needs_real
def test_explain_failure_is_friendly(monkeypatch):
    import kardio.explain.text as text_module

    def boom(*args, **kwargs):
        raise RuntimeError("Netz weg")

    monkeypatch.setattr(text_module, "explain", boom)
    ecg, _ = ui_upload.check_ecg(REAL_ECG.read_bytes())
    biopsies, _ = ui_upload.check_biopsies(tiny_biopsies(), None)
    at = new_app()
    at.session_state["result"] = ui_upload.run_real(ecg, biopsies)
    at.run()
    at.button(key="btn_explain").click().run()
    assert_clean(at)
    assert any("Befundtext konnte gerade nicht erzeugt werden" in i.value for i in at.info)


# ---------- Pruef-Funktionen des Hochladens ----------

@needs_real
def test_check_ecg_accepts_real_file():
    ecg, err = ui_upload.check_ecg(REAL_ECG.read_bytes())
    assert err is None
    assert ecg["patient"].nunique() == 3 and len(ecg) == 70


def test_check_ecg_rejects_wrong_file():
    wrong = tiny_biopsies().to_csv(sep=";", index=False).encode("utf-8")
    ecg, err = ui_upload.check_ecg(wrong)
    assert ecg is None
    assert err.startswith("EKG-Tabelle: fehlende Spalten")
    assert "Traceback" not in err


def test_check_ecg_rejects_unreadable_bytes():
    ecg, err = ui_upload.check_ecg(b"\xff\xfe\x00\x01kein csv")
    assert ecg is None and err


def test_biopsies_recognised_columns():
    data = tiny_biopsies().to_csv(sep=";", index=False).encode("utf-8")
    raw, guess, err = ui_upload.inspect_biopsies(data)
    assert err is None
    assert ui_upload.missing_roles(guess) == []
    biopsies, err = ui_upload.check_biopsies(raw, None)
    assert err is None and len(biopsies) == 4
    assert list(biopsies["grade"]) == [0, 1, 0, 2]  # sortiert nach Patient, Datum: MC, MC, MW, MW


def test_biopsies_unknown_columns_need_mapping():
    frame = tiny_biopsies().rename(columns={"Patient": "Spalte A", "Datum": "Spalte B", "Befund": "Spalte C"})
    raw, guess, err = ui_upload.inspect_biopsies(frame.to_csv(index=False).encode("utf-8"))
    assert err is None
    assert ui_upload.missing_roles(guess) == ["patient", "date", "rejection"]

    _, err = ui_upload.check_biopsies(raw, {"patient": "Spalte A", "date": None, "rejection": "Spalte C",
                                            "biopsy_no": None})
    assert "Datum der Biopsie" in err

    biopsies, err = ui_upload.check_biopsies(raw, {"patient": "Spalte A", "date": "Spalte B",
                                                   "rejection": "Spalte C", "biopsy_no": None})
    assert err is None and int(biopsies["rejection"].sum()) == 1

    _, err = ui_upload.check_biopsies(raw, {"patient": "Spalte A", "date": "Spalte C",
                                            "rejection": "Spalte B", "biopsy_no": None})
    assert err.startswith("Biopsie-Tabelle: Datum nicht lesbar")


def test_demo_assumptions_in_plain_german():
    from kardio.config import load_generator_config
    cfg = load_generator_config()
    rows = dict(ui_upload.demo_assumptions(cfg))
    assert rows["Erfundene Patienten"] == str(cfg.n_patients)
    assert rows["R-Amplitude"] == "sinkt um 15 %"
    assert rows["Herzrate"] == "steigt um 10 %"
    cfg.pattern_enabled = False
    assert "Eingebautes Muster" in dict(ui_upload.demo_assumptions(cfg))


# ---------- Formate und Diagramm ----------

@pytest.mark.parametrize("value, expected", [
    (0.00001, "<0.0001"), (0.0001, "0.0001"), (0.01564, "0.0156"), (0.5, "0.5000"),
    (0.9999, "0.9999"), (0.99995, ">0.9999"), (1.0, ">0.9999"), (float("nan"), "–"), (None, "–"),
])
def test_fmt_p(value, expected):
    assert ui_common.fmt_p(value) == expected


def test_labels():
    assert ui_common.feature_label("R_duration", "cv") == "R-Dauer (QRS) CV"
    assert ui_common.column_label("R_rate_mean__zmean") == "Herzrate Absolutwert · Mittlere Abweichung"
    assert ui_common.column_label("unbekannt") == "unbekannt"


def test_box_grid_survives_empty_groups():
    from kardio.constants import FEATURE_GROUPS
    from kardio.stats.compare import RANKING_COLUMNS
    windows = pd.DataFrame({"patient": ["A"], "grade": [0], "usable": [True],
                            "R_amplitude_mean__zmean": [0.4]})
    fig = ui_features.box_grid(windows, pd.DataFrame(columns=RANKING_COLUMNS), "mean", "zmean")
    assert len(fig.data) == 3 * len(FEATURE_GROUPS)
    titles = [a.text for a in fig.layout.annotations[:len(FEATURE_GROUPS)]]
    assert titles[0] == "<b>R-Amplitude Absolutwert</b>"
    assert fig.layout.annotations[0].font.color == "black"


def test_csv_export_defuses_formula_cells():
    import pandas as pd
    from ui_upload import to_csv_bytes
    df = pd.DataFrame({"patient": ["=cmd|' /C calc'!A0", "+SUM(1)", "@x", "MW", "-2"], "wert": [-1.5, 2.0, 3.0, 4.0, 5.0]})
    lines = to_csv_bytes(df).decode("utf-8-sig").splitlines()
    assert lines[1].startswith("'=cmd") and lines[2].startswith("'+SUM") and lines[3].startswith("'@x")
    assert lines[4].startswith("MW;") and lines[5].startswith("'-2;")
    assert lines[1].endswith(";-1.5")  # echte Zahlen bleiben Zahlen


def test_password_throttle_delay_and_short_cooldown(monkeypatch):
    import auth
    waited = []
    monkeypatch.setattr(auth.time, "sleep", waited.append)
    auth._failures.clear()
    auth._cooldown_until = 0.0
    for _ in range(auth.MAX_FAILURES_PER_WINDOW - 1):
        auth._register_failure()
    assert not auth._in_cooldown() and all(w >= 1 for w in waited)   # jeder Fehlversuch kostet Zeit
    auth._register_failure()
    assert auth._in_cooldown()                                        # Flut -> kurze Pause fuer alle
    assert auth.COOLDOWN_SECONDS <= 60                                # kurz: kein dauerhaftes Aussperren
    monkeypatch.setattr(auth.time, "time", lambda: auth._cooldown_until + 1)
    assert not auth._in_cooldown()                                    # Pause laeuft von selbst ab
    auth._failures.clear()
    auth._cooldown_until = 0.0


def test_validation_tab_demo_and_null(monkeypatch):
    """Zweiter Strang: gelerntes Modell auf unabhaengige Demo-Daten anwenden, mit und ohne Muster."""
    monkeypatch.setenv("APP_PASSWORD", "")
    monkeypatch.setenv("MGA_API_KEY", "")
    at = AppTest.from_file("app/main.py", default_timeout=600).run()
    [b for b in at.button if b.key == "btn_demo"][0].click().run()
    assert at.session_state["result"].model is not None
    [b for b in at.button if b.key == "btn_val_demo"][0].click().run()
    assert not at.exception and not [e for e in at.error]
    v = at.session_state["validation"]
    assert v.metrics.status == "ok" and v.metrics.auc > 0.85
    assert set(v.daily["patient"]).isdisjoint(set(at.session_state["result"].daily["patient"]))
    [b for b in at.button if b.key == "btn_val_null"][0].click().run()
    assert not at.exception and not [e for e in at.error]
    assert 0.35 < at.session_state["validation"].metrics.auc < 0.65
    # neues Training verwirft die alte Pruefung
    [b for b in at.button if b.key == "btn_demo"][0].click().run()
    assert "validation" not in at.session_state
