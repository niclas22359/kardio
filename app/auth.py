"""Kennwortabfrage: nur aktiv, wenn die Umgebungsvariable APP_PASSWORD gesetzt ist."""
from __future__ import annotations

import hmac
import os
import time

import streamlit as st


def password_ok(entered: str, expected: str) -> bool:
    return hmac.compare_digest(str(entered).encode("utf-8"), str(expected).encode("utf-8"))


# Bremse gegen Durchprobieren, bewusst als Mittelweg:
# - jeder Fehlversuch kostet Wartezeit (trifft nur die Sitzung, die falsch geraten hat)
# - bei einer Flut von Fehlversuchen pausiert die Anmeldung kurz fuer alle. Die Pause ist kurz,
#   damit niemand die Seite dauerhaft fuer andere sperren kann, begrenzt aber das Durchprobieren
#   auf hoechstens MAX_FAILURES_PER_WINDOW Versuche je Zeitfenster.
FAILURE_DELAY_SECONDS = 1.5
MAX_FAILURES_PER_WINDOW = 20
WINDOW_SECONDS = 60
COOLDOWN_SECONDS = 30
_failures: list[float] = []
_cooldown_until = 0.0


def _in_cooldown() -> bool:
    return time.time() < _cooldown_until


def _register_failure() -> None:
    global _cooldown_until
    now = time.time()
    _failures[:] = [t for t in _failures if now - t < WINDOW_SECONDS]
    _failures.append(now)
    if len(_failures) >= MAX_FAILURES_PER_WINDOW:
        _cooldown_until = now + COOLDOWN_SECONDS
        _failures.clear()
    time.sleep(FAILURE_DELAY_SECONDS)


def check_password() -> bool:
    """True, wenn die App gezeigt werden darf. Zeigt sonst nur das Kennwortfeld."""
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        if os.environ.get("RENDER"):
            # Online ohne Kennwort waeren hochgeladene Patiententabellen fuer jeden erreichbar.
            st.error("Diese Seite ist gesperrt: Es wurde kein Kennwort hinterlegt. "
                     "Bitte in den Einstellungen des Dienstes den Wert APP_PASSWORD eintragen.")
            return False
        return True
    if st.session_state.get("auth_ok"):
        return True
    st.subheader("Anmeldung")
    entered = st.text_input("Kennwort", type="password", key="password_input")
    clicked = st.button("Anmelden", type="primary", key="password_submit")
    if entered or clicked:
        if _in_cooldown():
            st.error("Zu viele Fehlversuche. Bitte in einer halben Minute erneut versuchen.")
            return False
        if password_ok(entered, expected):
            st.session_state["auth_ok"] = True
            st.rerun()
        _register_failure()
        st.error("Kennwort falsch.")
    return False
