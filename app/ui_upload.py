"""Reiter 1: Tabellen hochladen oder Demo starten, pruefen, analysieren."""
from __future__ import annotations

import io
import traceback

import pandas as pd
import streamlit as st

from kardio.config import load_analysis_config, load_generator_config
from kardio.generator.demo import generate_demo
from kardio.ingest.biopsies import guess_columns, load_biopsies
from kardio.ingest.ecg import IngestError, load_ecg, read_table
from kardio.pipeline import AnalysisResult, run_analysis_frames

from ui_common import DEMO_WITH_PATTERN, DEMO_WITHOUT_PATTERN, group_name, show_table

ROLE_LABELS = {"patient": "Patient", "date": "Datum der Biopsie", "rejection": "Befund",
               "biopsy_no": "Biopsie-Nummer"}
REQUIRED_ROLES = ["patient", "date", "rejection"]
NONE_OPTION = "– keine –"
ECG_HINT = ("Erwartet wird die Auswertungstabelle der EKG-Messungen: eine Zeile pro Messung mit den "
            "Spalten „Name“ (Patientenkürzel), „Date“ (Dateiname mit Datum, z. B. ecg_2025-08-28_1.csv) "
            "und den Messwerten wie „R_amplitude_mean“, „T_duration_sd“ usw.")
BIOPSY_HINT = ("Erwartet wird eine Zeile pro Biopsie mit Patientenkürzel, Datum (z. B. 02.01.2025) und "
               "Befund (z. B. ja / nein / Zwischengruppe oder 0R / 1R / 2R).")


# ---------- reine Pruef-Funktionen (ohne Oberflaeche, direkt testbar) ----------

def check_ecg(data: bytes) -> tuple[pd.DataFrame | None, str | None]:
    """Inhalt einer EKG-Datei pruefen -> (eingelesene Tabelle, None) oder (None, Fehlermeldung)."""
    try:
        df = load_ecg(io.BytesIO(data))
    except IngestError as e:
        return None, str(e)
    except Exception as e:  # noqa: BLE001
        return None, f"Datei konnte nicht gelesen werden: {e}"
    if len(df) == 0:
        return None, "EKG-Tabelle: Die Datei enthält keine Messungen."
    return df, None


def inspect_biopsies(data: bytes) -> tuple[pd.DataFrame | None, dict, str | None]:
    """Biopsie-Datei roh lesen und Spalten raten -> (Rohtabelle, Zuordnung, Fehlermeldung)."""
    empty = {k: None for k in ROLE_LABELS}
    try:
        raw = read_table(io.BytesIO(data))
    except IngestError as e:
        return None, empty, str(e)
    except Exception as e:  # noqa: BLE001
        return None, empty, f"Datei konnte nicht gelesen werden: {e}"
    if len(raw.columns) < 3:
        return None, empty, ("Biopsie-Tabelle: Es wurden weniger als drei Spalten erkannt. "
                             "Nötig sind Patient, Datum und Befund.")
    return raw, guess_columns(list(raw.columns)), None


def missing_roles(mapping: dict) -> list[str]:
    """Welche der drei Pflichtangaben wurden nicht erkannt?"""
    return [k for k in REQUIRED_ROLES if not mapping.get(k)]


def check_biopsies(raw: pd.DataFrame, mapping: dict | None) -> tuple[pd.DataFrame | None, str | None]:
    """Biopsie-Rohtabelle mit Zuordnung pruefen -> (Tabelle, None) oder (None, Fehlermeldung)."""
    if mapping is not None:
        missing = missing_roles(mapping)
        if missing:
            return None, ("Biopsie-Tabelle: Bitte eine Spalte zuordnen für: "
                          + ", ".join(ROLE_LABELS[k] for k in missing))
        chosen = [v for v in mapping.values() if v]
        if len(chosen) != len(set(chosen)):
            return None, "Biopsie-Tabelle: Dieselbe Spalte wurde mehrfach zugeordnet."
    try:
        df = load_biopsies(raw, mapping)
    except IngestError as e:
        return None, str(e)
    except Exception as e:  # noqa: BLE001
        return None, f"Biopsie-Tabelle konnte nicht gelesen werden: {e}"
    if len(df) == 0:
        return None, "Biopsie-Tabelle: Die Datei enthält keine Biopsien."
    return df, None


def real_label(ecg: pd.DataFrame) -> str:
    return f"Hochgeladene Daten ({ecg['patient'].nunique()} Patienten)"


MIN_WINDOW_DAYS, MAX_WINDOW_DAYS = 3, 90


def analysis_cfg(window_days: int | None = None):
    """Einstellungen der Auswertung; window_days = Zeitraum vor jeder Biopsie, der verdichtet wird."""
    cfg = load_analysis_config()
    if window_days is not None:
        cfg.window_days = int(min(max(int(window_days), MIN_WINDOW_DAYS), MAX_WINDOW_DAYS))
    return cfg


def run_real(ecg: pd.DataFrame, biopsies: pd.DataFrame, window_days: int | None = None) -> AnalysisResult:
    return run_analysis_frames(ecg, biopsies, analysis_cfg(window_days), data_source_label=real_label(ecg))


def demo_frames(pattern: bool):
    """Demo-Rohtabellen im Format der echten Dateien plus die verwendeten Einstellungen."""
    cfg = load_generator_config()
    cfg.pattern_enabled = bool(pattern)
    ecg_raw, bx_raw, _ = generate_demo(cfg)
    return ecg_raw, bx_raw, cfg


_FORMULA_START = ("=", "+", "-", "@", chr(9), chr(13))


def _defuse(value):
    """Text, den Excel als Formel ausfuehren wuerde (z.B. ein Kuerzel '=cmd|...'), mit ' entschaerfen."""
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value
    return value


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == object or pd.api.types.is_string_dtype(safe[col]):
            safe[col] = safe[col].map(_defuse)
    safe.columns = [_defuse(str(c)) for c in safe.columns]
    return safe.to_csv(sep=";", index=False).encode("utf-8-sig")


def run_demo(pattern: bool, window_days: int | None = None) -> AnalysisResult:
    ecg_raw, bx_raw, _ = demo_frames(pattern)
    ecg = load_ecg(io.StringIO(ecg_raw.to_csv(sep=";", index=False)))
    biopsies = load_biopsies(bx_raw)
    label = DEMO_WITH_PATTERN if pattern else DEMO_WITHOUT_PATTERN
    return run_analysis_frames(ecg, biopsies, analysis_cfg(window_days), data_source_label=label)


def demo_assumptions(cfg) -> list[tuple[str, str]]:
    """Generator-Einstellungen in Alltagssprache."""
    rows = [
        ("Erfundene Patienten", str(cfg.n_patients)),
        ("Beobachtungszeit je Patient", f"{cfg.days} Tage"),
        ("Tage ohne Messung", f"etwa {cfg.missing_day_rate * 100:.0f} %"),
        ("Anteil Biopsien mit Abstoßung", f"etwa {cfg.rejection_rate * 100:.0f} %"),
        ("Anteil Biopsien in der Zwischengruppe", f"etwa {cfg.borderline_rate * 100:.0f} %"),
    ]
    if cfg.pattern_enabled:
        rows.append(("Beginn der EKG-Veränderung", f"{cfg.pattern_days_before} Tage vor der Abstoßung"))
        rows.append(("Stärke in der Zwischengruppe",
                     f"{cfg.borderline_pattern_factor * 100:.0f} % der vollen Veränderung"))
        rows += pattern_rows(cfg)
    else:
        rows.append(("Eingebautes Muster", "keines – Abstoßungen hinterlassen keine Spur im EKG"))
    return rows


def pattern_rows(cfg) -> list[tuple[str, str]]:
    out = []
    for col, rel in (cfg.pattern or {}).items():
        group = col[:-5] if col.endswith("_mean") else col
        direction = "steigt" if rel > 0 else "sinkt"
        out.append((group_name(group), f"{direction} um {abs(rel) * 100:.0f} %"))
    return out


# ---------- Oberflaeche ----------

_cached_check_ecg = st.cache_data(show_spinner=False, max_entries=4)(check_ecg)
_cached_inspect = st.cache_data(show_spinner=False, max_entries=4)(inspect_biopsies)


def demo_csv(pattern: bool) -> tuple[bytes, bytes]:
    ecg_raw, bx_raw, _ = demo_frames(pattern)
    return to_csv_bytes(ecg_raw), to_csv_bytes(bx_raw)


_cached_demo_csv = st.cache_data(show_spinner=False, max_entries=2)(demo_csv)


def _store(result: AnalysisResult) -> None:
    st.session_state["result"] = result
    st.session_state["explain_cache"] = {}
    st.session_state.pop("validation", None)      # neue Gewichte: alte Pruefung gilt nicht mehr
    st.session_state.pop("validation_error", None)
    st.session_state.pop("analysis_error", None)


def _analyse(run, *args) -> None:
    """Analyse starten; Fehler werden als verstaendliche Meldung gemerkt, nie als Absturzseite."""
    try:
        with st.spinner("Analyse läuft … das dauert meist 10 bis 30 Sekunden."):
            result = run(*args)
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger("kardio").exception("Analyse fehlgeschlagen")  # Details nur ins Server-Protokoll
        message = str(e) if isinstance(e, IngestError) else f"unerwarteter Fehler ({type(e).__name__})"
        st.session_state["analysis_error"] = (message, "")
        return
    _store(result)
    st.rerun()


def _ecg_slot(key: str = "up_ecg"):
    up = st.file_uploader("EKG-Tabelle (CSV)", type=["csv", "txt"], key=key,
                          help="Eine Zeile pro EKG-Messung: Patientenkürzel, Datum und die Messwerte "
                               "(Amplituden, Dauern, Intervalle, Herzrate).")
    if up is None:
        return None
    ecg, err = _cached_check_ecg(up.getvalue())
    if err:
        st.error(err)
        st.caption(ECG_HINT)
        return None
    st.success(f"EKG-Tabelle gelesen: {ecg['patient'].nunique()} Patienten, {len(ecg)} Messungen.")
    return ecg


def _mapping_form(raw: pd.DataFrame, guess: dict, prefix: str = "map") -> dict:
    st.warning("Nicht alle nötigen Spalten wurden erkannt. Bitte zuordnen, was in welcher Spalte steht.")
    columns = list(raw.columns)
    mapping = {}
    for role in ["patient", "date", "rejection", "biopsy_no"]:
        optional = role == "biopsy_no"
        options = [NONE_OPTION] + columns
        index = options.index(guess[role]) if guess.get(role) in options else 0
        label = ROLE_LABELS[role] + (" (freiwillig)" if optional else "")
        choice = st.selectbox(label, options, index=index, key=f"{prefix}_{role}")
        mapping[role] = None if choice == NONE_OPTION else choice
    return mapping


def _biopsy_slot(key: str = "up_bx", label: str = "Biopsie-Tabelle (CSV)"):
    up = st.file_uploader(label, type=["csv", "txt"], key=key,
                          help="Eine Zeile pro Biopsie: Patientenkürzel, Datum und Befund "
                               "(Abstoßung ja / nein / Zwischengruppe). Spaltennamen dürfen abweichen.")
    if up is None:
        return None
    raw, guess, err = _cached_inspect(up.getvalue())
    if err:
        st.error(err)
        st.caption(BIOPSY_HINT)
        return None
    if missing_roles(guess):
        mapping = _mapping_form(raw, guess, prefix=f"map_{key}")
    else:
        mapping = None
        found = [f"{ROLE_LABELS[k]} = „{v}“" for k, v in guess.items() if v]
        st.caption("Erkannte Spalten: " + " · ".join(found))
    biopsies, err = check_biopsies(raw, mapping)
    if err:
        st.error(err)
        st.caption(BIOPSY_HINT)
        return None
    n_rej = int((biopsies["grade"] == 2).sum())
    st.success(f"Biopsie-Tabelle gelesen: {len(biopsies)} Biopsien von "
               f"{biopsies['patient'].nunique()} Patienten, davon {n_rej} mit Abstoßung.")
    return biopsies


def _demo_section() -> None:
    with st.container(border=True):
        st.subheader("Keine Daten zur Hand? Demo starten")
        pattern = st.toggle(
            "Muster eingebaut (Abstoßung kündigt sich im EKG an)", value=True, key="demo_pattern",
            help="An: In die erfundenen Daten ist eine EKG-Veränderung vor jeder Abstoßung eingebaut. "
                 "Die Analyse muss sie wiederfinden. Aus: Gegenprobe ohne Muster, die Analyse darf "
                 "dann nichts finden.")
        if st.button("Demo-Daten laden und analysieren", key="btn_demo", use_container_width=True):
            _analyse(run_demo, pattern, st.session_state.get("window_days"))
        with st.expander("Demo-Annahmen"):
            cfg = load_generator_config()
            cfg.pattern_enabled = pattern
            st.caption("Die Demo-Patienten sind erfunden. Ihre Wertebereiche stammen aus echten "
                       "Messungen, die drei echten Patienten laufen unverändert mit.")
            show_table(pd.DataFrame(demo_assumptions(cfg), columns=["Annahme", "Wert"]))
        with st.expander("Demo-Tabellen herunterladen (zum Ausprobieren des Hochladens)"):
            try:
                ecg_csv, bx_csv = _cached_demo_csv(pattern)
            except Exception as e:  # noqa: BLE001
                st.info(f"Demo-Tabellen konnten nicht erzeugt werden: {e}")
                return
            c1, c2 = st.columns(2)
            c1.download_button("EKG-Tabelle (Demo)", ecg_csv, "ekg_demo.csv", "text/csv",
                               key="dl_demo_ecg", use_container_width=True)
            c2.download_button("Biopsie-Tabelle (Demo)", bx_csv, "biopsien_demo.csv", "text/csv",
                               key="dl_demo_bx", use_container_width=True)


def _result_section(result: AnalysisResult) -> None:
    st.divider()
    st.subheader("Qualitätsbericht")
    report = result.report
    if report.patients_only_ecg:
        st.warning("Nur in der EKG-Tabelle (ohne Biopsien): " + ", ".join(report.patients_only_ecg)
                   + ". Diese Patienten gehen nicht in den Gruppenvergleich ein.")
    if report.patients_only_biopsy:
        st.warning("Nur in der Biopsie-Tabelle (ohne EKG): " + ", ".join(report.patients_only_biopsy)
                   + ". Für diese Biopsien gibt es keine EKG-Werte.")
    left, right = st.columns([3, 2])
    with left:
        show_table(pd.DataFrame(report.as_rows(), columns=["Angabe", "Wert"]), height=460)
    with right:
        st.markdown("**Ergebnisse herunterladen**")
        st.download_button("Merkmals-Rangliste (CSV)", to_csv_bytes(result.ranking),
                           "merkmals_rangliste.csv", "text/csv", key="dl_ranking",
                           use_container_width=True)
        st.download_button("Risikoverlauf je Patient und Tag (CSV)", to_csv_bytes(result.timeline),
                           "risikoverlauf.csv", "text/csv", key="dl_timeline",
                           use_container_width=True)
        st.caption("Die Auswertung steht jetzt in den Reitern »Übersicht«, »Merkmale«, »Patient« "
                   "und »Modell«.")


def _window_input() -> int:
    default = load_analysis_config().window_days
    with st.container(border=True):
        left, right = st.columns([1, 2])
        days = left.number_input(
            "Zeitraum vor der Biopsie (Tage)", min_value=MIN_WINDOW_DAYS, max_value=MAX_WINDOW_DAYS,
            value=default, step=1, key="window_days",
            help="So viele Tage vor jeder Biopsie werden zu den Kennzahlen verdichtet. "
                 "Gilt für eigene Tabellen und für die Demo.")
        right.markdown(
            f"Für jede Biopsie werden die **{int(days)} Tage davor** ausgewertet: mittlere und größte "
            "Abweichung vom persönlichen Normal, Trend und Anteil auffälliger Tage. "
            "Ein kürzerer Zeitraum reagiert schärfer auf späte Veränderungen, braucht aber lückenlose "
            "Messungen. Nach einer Änderung bitte neu analysieren.")
    return int(days)


def render() -> None:
    window_days = _window_input()
    st.subheader("Eigene Tabellen hochladen")
    c1, c2 = st.columns(2)
    with c1:
        ecg = _ecg_slot()
    with c2:
        biopsies = _biopsy_slot()
    ready = ecg is not None and biopsies is not None
    if st.button("🔴 Analysieren", type="primary", disabled=not ready, key="btn_analyse",
                 use_container_width=True):
        _analyse(run_real, ecg, biopsies, window_days)
    if not ready:
        st.caption("Der Knopf wird aktiv, sobald beide Tabellen hochgeladen und lesbar sind.")

    _demo_section()

    error = st.session_state.get("analysis_error")
    if error:
        st.error(f"Die Analyse ist fehlgeschlagen: {error[0]}")

    result = st.session_state.get("result")
    if result is not None:
        _result_section(result)
