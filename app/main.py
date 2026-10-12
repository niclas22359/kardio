"""Einstieg des Dashboards: streamlit run app/main.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Kardio – EKG-Früherkennung", layout="wide")

try:  # lokale Schluesseldatei; online kommen die Werte aus der Umgebung
    from dotenv import load_dotenv
    if (ROOT / ".env").exists():
        load_dotenv(ROOT / ".env")
except Exception:  # noqa: BLE001  (fehlendes Paket darf die App nicht stoppen)
    pass

import auth  # noqa: E402
import ui_common  # noqa: E402
import ui_features  # noqa: E402
import ui_model  # noqa: E402
import ui_overview  # noqa: E402
import ui_patient  # noqa: E402
import ui_upload  # noqa: E402
import ui_validate  # noqa: E402

TABS = [
    ("📂 Daten laden (Training)", ui_upload.render),
    ("🚦 Übersicht", ui_overview.render),
    ("📊 Merkmale", ui_features.render),
    ("🫀 Patient", ui_patient.render),
    ("🧮 Modell", ui_model.render),
    ("🔎 Prüfen (2. Datensatz)", ui_validate.render),
]


def _safe(render) -> None:
    """Ein Fehler in einem Reiter darf nie als Programmfehler-Seite beim Nutzer landen."""
    try:
        render()
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ in ("RerunException", "StopException"):
            raise
        # Details nur ins Server-Protokoll, nie in den Browser (koennten Pfade oder Interna zeigen)
        import logging
        logging.getLogger("kardio").exception("Fehler beim Anzeigen eines Reiters")
        st.error(f"Dieser Bereich konnte nicht angezeigt werden ({type(e).__name__}). "
                 "Bitte die Analyse neu starten.")


def main() -> None:
    st.session_state.setdefault("result", None)
    st.title("Kardio – EKG-Früherkennung von Abstoßungen")
    st.caption("Kündigt sich eine Abstoßung in den Tagen vor der Biopsie im EKG der Apple Watch an?")
    if not auth.check_password():
        return
    ui_common.show_banner(st.session_state["result"])
    for tab, (_, render) in zip(st.tabs([name for name, _ in TABS]), TABS):
        with tab:
            _safe(render)


main()
