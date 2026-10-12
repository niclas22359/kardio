"""Reiter 2: Kennzahlen und Ampel je Patient."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui_common import LIGHT_ICONS, fmt_num, get_result, patient_summary, show_table


def render() -> None:
    result = get_result()
    if result is None:
        return
    report, model, cfg = result.report, result.model_report, result.cfg

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Patienten", report.n_patients_ecg)
    k2.metric("Messtage", report.n_days)
    k3.metric("Biopsien", report.n_biopsies)
    k4.metric("Abstoßungen", report.n_rejections)
    k5.metric("Modellgüte (AUC)", fmt_num(model.auc_cv) if model.status == "ok" else "–",
              help="Trennschärfe des Risikomodells: 0.5 = Zufall, 1.0 = trennt Abstoßung und "
                   "keine Abstoßung perfekt. Geprüft an Patienten, die das Modell nicht kannte.")

    if model.status != "ok":
        st.info(f"**Kein Risikomodell möglich.** {model.reason} Die Merkmals-Statistik im Reiter "
                "»Merkmale« läuft trotzdem; die Ampel bleibt ohne Modell grau.", icon="ℹ️")

    st.subheader("Patienten")
    summary = patient_summary(result)
    table = pd.DataFrame({
        "Ampel": summary["light"].map(LIGHT_ICONS),
        "Patient": summary["patient"],
        "Letztes Risiko (%)": (summary["risk"] * 100).round(0),
        "Messtage": summary["n_days"],
        "Biopsien": summary["n_biopsies"],
        "Zwischengruppe": summary["n_mid"],
        "Abstoßungen": summary["n_rej"],
        "Letzte Messung": summary["last"],
    })
    risk_help = ("Vom Modell geschätzte Wahrscheinlichkeit, dass die letzten Tage so aussehen wie vor "
                 "einer Abstoßung. Leer, wenn kein Modell oder zu wenige Messtage vorliegen.")
    show_table(
        table, height=520,
        help={"Letztes Risiko (%)": risk_help},
        formats={"Letztes Risiko (%)": lambda v: f"{v:.0f} %"},
        column_config={
            "Ampel": st.column_config.TextColumn("Ampel", width="small"),
            "Letztes Risiko (%)": st.column_config.NumberColumn(
                "Letztes Risiko (%)", format="%d %%", help=risk_help),
            "Letzte Messung": st.column_config.DateColumn("Letzte Messung", format="DD.MM.YYYY"),
        })
    amber, red = cfg.amber_threshold * 100, cfg.red_threshold * 100
    st.caption(f"🟢 unter {amber:.0f} %  ·  🟡 {amber:.0f} bis unter {red:.0f} %  ·  "
               f"🔴 ab {red:.0f} %  ·  ⚪ keine Aussage (kein Modell oder zu wenige Messtage)")

    with st.expander("Was sehe ich hier?"):
        st.markdown(
            f"- Für jeden Patienten wird aus allen seinen Messtagen ein **persönliches Normal** "
            f"gebildet. Jede spätere Messung wird daran gemessen.\n"
            f"- Die **{cfg.window_days} Tage vor jeder Biopsie** werden zu wenigen Zahlen verdichtet "
            f"und zwischen den Gruppen „keine Abstoßung“, „Zwischengruppe“ und „Abstoßung“ verglichen.\n"
            f"- Das **Risiko** sagt, wie sehr die letzten {cfg.window_days} Tage eines Patienten den "
            f"Tagen vor einer Abstoßung ähneln. Die Ampel fasst das zusammen.\n"
            f"- Die Tabelle ist nach Risiko sortiert: Wer oben steht, sollte zuerst angesehen werden.\n"
            f"- Das ist ein Hinweis für den Arzt, **keine Diagnose** und kein Ersatz für die Biopsie.")
