"""Zeitraum vor der Biopsie als freie Zahl; Abschnitt mit den Prüfungen auf Patientenebene."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ui_features_patient  # noqa: E402
import ui_upload  # noqa: E402


def test_window_days_setting_changes_analysis():
    assert ui_upload.analysis_cfg(7).window_days == 7
    assert ui_upload.analysis_cfg(None).window_days == 14
    assert ui_upload.analysis_cfg(1).window_days == ui_upload.MIN_WINDOW_DAYS
    assert ui_upload.analysis_cfg(999).window_days == ui_upload.MAX_WINDOW_DAYS
    r = ui_upload.run_demo(True, window_days=7)
    assert r.cfg.window_days == 7 and r.model_report.status == "ok"
    assert r.windows["n_data_days"].max() <= 8          # 7 Tage davor plus Biopsietag
    t = ui_features_patient.patient_level_table(r.ranking)
    assert "p Patienten berücksichtigt" in t.columns and len(t) == 15
    assert ui_features_patient.paired_chart(r.windows, r.ranking["column"].iloc[0]) is not None


def test_window_days_input_in_app(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "")
    monkeypatch.setenv("MGA_API_KEY", "")
    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=600).run()
    field = [n for n in at.number_input if n.key == "window_days"][0]
    field.set_value(7).run()
    [b for b in at.button if b.key == "btn_demo"][0].click().run()
    assert not at.exception and not [e for e in at.error]
    assert at.session_state["result"].cfg.window_days == 7
    assert any("7 Tage vor jeder Biopsie" in w.value for w in at.warning)
    assert any("Mehrere Biopsien je Patient" in s.value for s in at.subheader)
