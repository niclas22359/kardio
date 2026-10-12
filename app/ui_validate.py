"""Reiter »Prüfen«: das im ersten Reiter gelernte Modell auf einen zweiten, unabhängigen Datensatz anwenden."""
from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kardio.config import load_generator_config
from kardio.constants import GRADE_COLORS, GRADE_LABELS
from kardio.generator.demo import generate_demo
from kardio.ingest.biopsies import load_biopsies
from kardio.ingest.ecg import IngestError, load_ecg
from kardio.validate import ValidationResult, lead_time_table, run_validation_frames

import ui_patient
import ui_upload
from ui_common import LINE_BLUE, base_layout, fmt_date, fmt_num, get_result, show_table

DEMO_B = "DEMO-DATEN, unabhängiger Datensatz B (mit eingebautem Muster)"
DEMO_NULL = "DEMO-DATEN, unabhängiger Datensatz ohne Muster (Gegenprobe)"
SEED_B, SEED_NULL = 202, 303


def demo_second_frames(pattern: bool):
    """Zweiter Demo-Datensatz: andere erfundene Patienten (anderer Zufallsstart), gleiche Regeln."""
    cfg = load_generator_config()
    cfg.seed = SEED_B if pattern else SEED_NULL
    cfg.n_patients = 30
    cfg.pattern_enabled = bool(pattern)
    ecg_raw, bx_raw, _ = generate_demo(cfg, include_real=False)
    prefix = "B" if pattern else "N"
    ecg_raw, bx_raw = ecg_raw.copy(), bx_raw.copy()
    ecg_raw["Name"] = ecg_raw["Name"].str.replace("D", prefix, n=1)
    bx_raw["Patient"] = bx_raw["Patient"].str.replace("D", prefix, n=1)
    return ecg_raw, bx_raw


def run_demo_validation(trained, pattern: bool) -> ValidationResult:
    ecg_raw, bx_raw = demo_second_frames(pattern)
    ecg = load_ecg(io.StringIO(ecg_raw.to_csv(sep=";", index=False)))
    return run_validation_frames(trained, ecg, load_biopsies(bx_raw), DEMO_B if pattern else DEMO_NULL)


def run_real_validation(trained, ecg: pd.DataFrame, biopsies: pd.DataFrame | None) -> ValidationResult:
    label = f"Zweiter Datensatz ({ecg['patient'].nunique()} Patienten)"
    return run_validation_frames(trained, ecg, biopsies, label)


def overlap(trained, ecg: pd.DataFrame) -> list[str]:
    """Patientenkürzel, die schon im Trainings-Datensatz vorkommen (dann wäre die Prüfung nicht unabhängig)."""
    return sorted(set(ecg["patient"]) & set(trained.daily["patient"]))


def roc_chart(m) -> go.Figure | None:
    if m.roc is None or len(m.roc) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Zufall (Münzwurf)",
                             line=dict(color="#999999", dash="dash")))
    fig.add_trace(go.Scatter(x=m.roc["fpr"], y=m.roc["tpr"], mode="lines", name="Modell am zweiten Datensatz",
                             line=dict(color=LINE_BLUE, width=3)))
    base_layout(fig, height=380)
    fig.update_xaxes(title_text="Fehlalarme (Anteil der Biopsien ohne Abstoßung)", range=[0, 1])
    fig.update_yaxes(title_text="Erkannte Abstoßungen (Anteil)", range=[0, 1.02])
    return fig


def risk_by_grade_chart(result: ValidationResult) -> go.Figure | None:
    w = result.windows.dropna(subset=["risk"])
    if len(w) == 0:
        return None
    fig = go.Figure()
    for grade in (0, 1, 2):
        values = w.loc[w["grade"] == grade, "risk"] * 100
        fig.add_trace(go.Box(y=values, name=GRADE_LABELS[grade], boxpoints="outliers",
                             fillcolor=GRADE_COLORS[grade], line=dict(color="black", width=1.5),
                             marker=dict(color=GRADE_COLORS[grade], size=5)))
    red = result.cfg.red_threshold * 100
    fig.add_hline(y=red, line=dict(color="#d32f2f", dash="dot", width=2),
                  annotation_text=f"Schwelle Rot ({red:.0f} %)", annotation_position="top left",
                  annotation_font=dict(color="#d32f2f", size=12))
    base_layout(fig, height=380, showlegend=False)
    fig.update_yaxes(title_text="Risiko am Tag der Biopsie (%)", range=[0, 100])
    return fig


def confusion_table(m) -> pd.DataFrame:
    return pd.DataFrame({
        "Befund der Biopsie": ["Abstoßung", "keine Abstoßung"],
        "Modell: rot (Alarm)": [m.tp, m.fp],
        "Modell: nicht rot": [m.fn, m.tn],
    })


def _run(fn, *args) -> None:
    try:
        with st.spinner("Der zweite Datensatz wird mit dem gelernten Modell bewertet …"):
            st.session_state["validation"] = fn(*args)
        st.session_state.pop("validation_error", None)
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger("kardio").exception("Pruefung fehlgeschlagen")
        expected = isinstance(e, (IngestError, ValueError))
        st.session_state["validation_error"] = str(e) if expected else f"unerwarteter Fehler ({type(e).__name__})"
        return
    st.rerun()


def _explain_box() -> None:
    with st.expander("Wozu dieser Reiter? Training und Prüfung in zwei Sätzen", expanded=False):
        st.markdown(
            "- **Training (Reiter »Daten laden«):** Aus einem Datensatz mit bekannten Biopsiebefunden lernt "
            "das Modell, welche EKG-Veränderungen vor einer Abstoßung auftreten. Das Ergebnis sind die "
            "Gewichte im Reiter »Modell«.\n"
            "- **Prüfung (dieser Reiter):** Die Gewichte bleiben eingefroren. Ein zweiter Datensatz mit "
            "anderen Patienten wird nur noch bewertet. Stimmen Alarm und Biopsiebefund auch hier überein, "
            "funktioniert das Modell bei Patienten, die es nie gesehen hat.\n"
            "- **Ohne Biopsietabelle** liefert dieser Reiter nur die Risikowerte je Patient. Das ist der "
            "spätere Alltagseinsatz: neue EKG-Daten rein, Ampel raus.\n"
            "- Für jeden Patienten wird sein **persönliches Normal aus seinen eigenen Messtagen** gebildet. "
            "Neu gelernt wird dabei nichts.")


def _input_section(trained) -> None:
    st.subheader("Zweiten Datensatz hochladen")
    left, right = st.columns(2)
    with left:
        ecg = ui_upload._ecg_slot(key="up_ecg2")
    with right:
        biopsies = ui_upload._biopsy_slot(key="up_bx2", label="Biopsie-Tabelle (CSV, freiwillig)")
    if ecg is not None:
        same = overlap(trained, ecg)
        if same:
            shown = ", ".join(same[:5]) + (" …" if len(same) > 5 else "")
            st.warning(f"**{len(same)} Patienten kommen schon im Trainings-Datensatz vor** ({shown}). "
                       "Die Prüfung ist dann nicht unabhängig und fällt zu gut aus.", icon="⚠️")
    if st.button("🔎 Mit dem gelernten Modell prüfen", type="primary", disabled=ecg is None,
                 key="btn_validate", use_container_width=True):
        _run(run_real_validation, trained, ecg, biopsies)
    if ecg is None:
        st.caption("Der Knopf wird aktiv, sobald die EKG-Tabelle hochgeladen und lesbar ist. "
                   "Die Biopsie-Tabelle ist freiwillig: ohne sie gibt es Risikowerte, aber keine Prüfung.")

    with st.container(border=True):
        st.markdown("**Kein zweiter Datensatz zur Hand?** Zwei erfundene Datensätze mit anderen Patienten:")
        c1, c2 = st.columns(2)
        if c1.button("Demo-Datensatz B prüfen (mit Muster)", key="btn_val_demo", use_container_width=True,
                     help="30 andere erfundene Patienten, gleiche Regeln wie im Training."):
            _run(run_demo_validation, trained, True)
        if c2.button("Gegenprobe: Datensatz ohne Muster prüfen", key="btn_val_null", use_container_width=True,
                     help="Hier hinterlässt die Abstoßung keine Spur im EKG. Das Modell darf nichts erkennen."):
            _run(run_demo_validation, trained, False)

    error = st.session_state.get("validation_error")
    if error:
        st.error(f"Die Prüfung ist fehlgeschlagen: {error}")


def _metrics_section(v: ValidationResult) -> None:
    m, cfg = v.metrics, v.cfg
    if m.status == "unlabeled":
        st.info(m.note, icon="ℹ️")
        return
    if m.status == "few":
        st.warning(m.note, icon="⚠️")
    red = f"{m.threshold * 100:.0f} %"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trennschärfe im Training", fmt_num(m.train_auc) if m.train_auc is not None else "–",
              help="AUC aus dem Training, geprüft an jeweils zurückgehaltenen Patienten. "
                   "0,5 = Münzwurf, 1,0 = perfekt.")
    ci = f"95-%-Bereich {fmt_num(m.auc_ci[0])} bis {fmt_num(m.auc_ci[1])}" if m.auc_ci else None
    k2.metric("Trennschärfe am zweiten Datensatz", fmt_num(m.auc) if m.auc is not None else "–", ci,
              delta_color="off", help="AUC des eingefrorenen Modells an den neuen Patienten.")
    k3.metric("Erkannte Abstoßungen", f"{m.sensitivity * 100:.0f} %" if m.sensitivity is not None else "–",
              help=f"Anteil der Biopsien mit Abstoßung, bei denen das Risiko am Biopsietag bei {red} oder "
                   "höher lag (Sensitivität).")
    k4.metric("Richtig entwarnt", f"{m.specificity * 100:.0f} %" if m.specificity is not None else "–",
              help=f"Anteil der Biopsien ohne Abstoßung, bei denen das Risiko unter {red} lag (Spezifität).")
    st.caption(f"Grundlage: {m.n_windows} verwertbare Biopsien von {m.n_patients} Patienten, davon "
               f"{m.n_rejections} mit Abstoßung. Die Zwischengruppe zählt hier nicht mit. "
               f"Verwertbar heißt: in den {cfg.window_days} Tagen davor mindestens "
               f"{cfg.min_window_data_days} Messtage und nicht in der Behandlungsphase nach einer Abstoßung.")

    if m.auc is not None and m.status == "ok":
        if m.auc >= 0.8:
            st.success("**Das Modell erkennt Abstoßungen auch bei Patienten, die es nie gesehen hat.** "
                       "Bei Demo-Daten heißt das nur: Die Maschine funktioniert.", icon="✅")
        elif m.auc <= 0.6:
            st.info("**Das Modell erkennt an diesem Datensatz nichts** (nahe am Münzwurf). Bei der Gegenprobe "
                    "ohne Muster ist genau das richtig: Es schlägt nicht einfach immer Alarm.", icon="🪙")
        else:
            st.warning("**Das Modell trennt an diesem Datensatz nur mäßig.**", icon="⚠️")

    left, right = st.columns(2)
    with left:
        st.markdown("**Alarm gegen Befund** (Anzahl Biopsien)")
        show_table(confusion_table(m))
        fig = risk_by_grade_chart(v)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key="val_risk_by_grade")
    with right:
        fig = roc_chart(m)
        if fig is not None:
            st.markdown("**Erkannte Abstoßungen gegen Fehlalarme**")
            st.plotly_chart(fig, use_container_width=True, key="val_roc")
            st.caption("Je weiter die blaue Kurve über der gestrichelten Linie liegt, desto besser. "
                       "Jeder Punkt der Kurve entspricht einer anderen Alarmschwelle.")

    lead = lead_time_table(v)
    if len(lead):
        hit = lead["first_red_days_before"].notna()
        if hit.any():
            st.markdown(
                f"**Vorwarnzeit:** Bei {int(hit.sum())} von {len(lead)} Abstoßungen stand die Ampel am "
                f"Biopsietag auf Rot, im Mittel seit {lead.loc[hit, 'first_red_days_before'].median():.0f} "
                f"Tagen (Median).")
        else:
            st.markdown(f"**Vorwarnzeit:** Bei keiner der {len(lead)} Abstoßungen stand die Ampel am "
                        "Biopsietag auf Rot.")


def _patient_section(v: ValidationResult) -> None:
    st.subheader("Einzelner Patient")
    last = v.timeline.dropna(subset=["risk"]).sort_values("date").groupby("patient").tail(1)
    last = last.sort_values("risk", ascending=False)
    patients = list(last["patient"]) + sorted(set(v.daily["patient"]) - set(last["patient"]))
    if not patients:
        st.info("Keine Patienten im zweiten Datensatz.")
        return
    patient = st.selectbox("Patient (sortiert nach letztem Risiko)", patients, key="val_patient")
    fig = ui_patient.risk_chart(v, patient)
    if fig is None:
        st.info("Für diesen Patienten gibt es zu wenige Messtage für einen Risikowert.")
    else:
        st.plotly_chart(fig, use_container_width=True, key="val_patient_risk")
    bx = v.windows[v.windows["patient"] == patient].sort_values("date")
    if len(bx):
        table = pd.DataFrame({
            "Nr.": bx["biopsy_no"].astype(int),
            "Datum": bx["date"].map(fmt_date),
            "Befund": bx["grade"].map(GRADE_LABELS),
            "Risiko am Biopsietag": bx["risk"].map(lambda r: "–" if pd.isna(r) else f"{r * 100:.0f} %"),
            "Modell": bx["risk"].map(lambda r: "–" if pd.isna(r)
                                     else ("🔴 Alarm" if r >= v.cfg.red_threshold else "kein Alarm")),
        })
        show_table(table)


def render() -> None:
    trained = get_result()
    if trained is None:
        return
    _explain_box()
    if getattr(trained, "model", None) is None:
        st.info("**Noch kein Modell gelernt.** Der Trainings-Datensatz im Reiter »Daten laden« reicht nicht "
                f"für ein Risikomodell. {trained.model_report.reason}", icon="ℹ️")
        return
    mr = trained.model_report
    st.success(f"**Gelerntes Modell:** aus „{trained.data_source_label}“, {mr.n_windows} Biopsien von "
               f"{mr.n_patients} Patienten, Trennschärfe im Training {fmt_num(mr.auc_cv)}. "
               "Die Gewichte bleiben für die Prüfung eingefroren.", icon="🧊")
    _input_section(trained)

    v = st.session_state.get("validation")
    if v is None:
        return
    if v.trained_on != trained.data_source_label:
        st.session_state.pop("validation", None)   # Training wurde inzwischen ersetzt
        return
    st.divider()
    st.subheader(f"Ergebnis der Prüfung: {v.data_source_label}")
    if v.is_demo:
        st.caption("Erfundene Daten. Keine medizinische Aussage.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Patienten", v.report.n_patients_ecg)
    c2.metric("Messtage", v.report.n_days)
    c3.metric("Biopsien", v.report.n_biopsies)
    _metrics_section(v)
    _patient_section(v)
    st.download_button("Risiko je Biopsie im zweiten Datensatz (CSV)",
                       ui_upload.to_csv_bytes(v.windows[["patient", "biopsy_no", "date", "grade", "risk",
                                                         "n_data_days", "usable"]]),
                       "pruefung_risiko_je_biopsie.csv", "text/csv", key="dl_validation")
