"""Reiter 4: ein Patient im Verlauf, mit persoenlichem Normalband, Biopsien, Risiko und Befundtext."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kardio.constants import FEATURE_GROUPS, GRADE_LABELS, METRIC_LABELS, METRICS

from ui_common import (FEATURE_UNITS, LIGHT_ICONS, LINE_BLUE, add_biopsy_markers, base_layout, date_axis,
                       feature_label, fmt_pct, gapless, get_result, group_name, patient_summary,
                       show_table)

DISCLAIMER = "Ersetzt keine Biopsie und keine ärztliche Beurteilung."


def _date_range(result, patient: str) -> list | None:
    """Gemeinsame Zeitachse beider Diagramme: alle Messtage und Biopsien des Patienten."""
    dates = pd.concat([result.daily.loc[result.daily["patient"] == patient, "date"],
                       result.biopsies.loc[result.biopsies["patient"] == patient, "date"]]).dropna()
    if len(dates) == 0:
        return None
    pad = pd.Timedelta(days=3)
    return [dates.min() - pad, dates.max() + pad]


def value_chart(result, patient: str, group: str, metric: str) -> go.Figure | None:
    feature = f"{group}_{metric}"
    dev = result.deviations
    series = dev[(dev["patient"] == patient) & (dev["feature"] == feature)][["date", "value"]]
    if series["value"].notna().sum() == 0:
        return None
    line = gapless(series, "value")
    fig = go.Figure()
    base = result.baselines
    base = base[(base["patient"] == patient) & (base["feature"] == feature)]
    if len(base) and pd.notna(base["center"].iloc[0]) and pd.notna(base["scale"].iloc[0]):
        center, scale = float(base["center"].iloc[0]), float(base["scale"].iloc[0])
        low, high = center - 2 * scale, center + 2 * scale
        if metric in ("cv", "dev"):
            low = max(low, 0.0)
        x0, x1 = line["date"].min(), line["date"].max()
        fig.add_trace(go.Scatter(x=[x0, x1, x1, x0], y=[low, low, high, high], fill="toself",
                                 fillcolor="rgba(46,204,64,0.18)", line=dict(width=0), mode="lines",
                                 name="Persönlicher Normalbereich", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=line["date"], y=line["value"], mode="lines+markers", connectgaps=False,
                             name=feature_label(group, metric), line=dict(color=LINE_BLUE, width=2),
                             marker=dict(size=4),
                             hovertemplate="%{x|%d.%m.%Y}<br>%{y:.4g}<extra></extra>"))
    bx = result.biopsies[result.biopsies["patient"] == patient]
    add_biopsy_markers(fig, bx)
    unit = {"mean": FEATURE_UNITS.get(group, ""), "cv": "Anteil", "dev": "Anteil"}[metric]
    base_layout(fig, height=400)
    date_axis(fig)
    fig.update_xaxes(range=_date_range(result, patient))
    fig.update_yaxes(title_text=f"{feature_label(group, metric)}" + (f" ({unit})" if unit else ""))
    return fig


def risk_chart(result, patient: str) -> go.Figure | None:
    tl = result.timeline
    series = tl[tl["patient"] == patient][["date", "risk"]]
    if series["risk"].notna().sum() == 0:
        return None
    line = gapless(series, "risk")
    cfg = result.cfg
    fig = go.Figure(go.Scatter(x=line["date"], y=line["risk"] * 100, mode="lines", connectgaps=False,
                               name="Risiko", line=dict(color=LINE_BLUE, width=2.5),
                               hovertemplate="%{x|%d.%m.%Y}<br>Risiko %{y:.0f} %<extra></extra>"))
    for value, color, text in ((cfg.amber_threshold, "#ff9800", "Schwelle Gelb"),
                               (cfg.red_threshold, "#d32f2f", "Schwelle Rot")):
        fig.add_hline(y=value * 100, line=dict(color=color, dash="dot", width=2),
                      annotation_text=f"{text} ({value * 100:.0f} %)", annotation_position="top right",
                      annotation_font=dict(color=color, size=12))
    add_biopsy_markers(fig, result.biopsies[result.biopsies["patient"] == patient])
    base_layout(fig, height=360)
    date_axis(fig)
    fig.update_xaxes(range=_date_range(result, patient))
    fig.update_yaxes(title_text="Risiko (%)", range=[0, 100])
    return fig


def biopsy_table(result, patient: str) -> pd.DataFrame:
    w = result.windows[result.windows["patient"] == patient].sort_values("date")
    need = result.cfg.min_window_data_days

    def verdict(row) -> str:
        if row["usable"]:
            return "ja"
        if row["under_treatment"]:
            return "nein (kurz nach einer Abstoßung)"
        return f"nein (weniger als {need} Messtage)"

    return pd.DataFrame({
        "Nr": w["biopsy_no"].to_numpy(),
        "Datum": w["date"].to_numpy(),
        "Befund": [GRADE_LABELS.get(int(g), "–") for g in w["grade"]],
        "Datentage im Fenster": w["n_data_days"].to_numpy(),
        "Verwertbar": [verdict(r) for _, r in w.iterrows()],
    })


def _explain_section(result, patient: str) -> None:
    st.subheader("Befundtext")
    cache = st.session_state.setdefault("explain_cache", {})
    if st.button("Befundtext erzeugen", key="btn_explain"):
        try:
            from kardio.explain.text import explain
            with st.spinner("Befundtext wird erzeugt … das kann bis zu einer halben Minute dauern."):
                text, source = explain(result, patient)
            cache[patient] = (str(text), str(source))
        except Exception as e:  # noqa: BLE001
            cache.pop(patient, None)
            st.info(f"Der Befundtext konnte gerade nicht erzeugt werden ({type(e).__name__}). "
                    "Die Auswertung oben ist davon nicht betroffen.")
    if patient in cache:
        text, source = cache[patient]
        with st.container(border=True):
            st.markdown(text)
        st.caption("Quelle: KI-Text (Zahlen geprüft)" if source == "llm" else "Quelle: Vorlagentext")
    st.caption(DISCLAIMER)


def render() -> None:
    result = get_result()
    if result is None:
        return
    summary = patient_summary(result)
    if len(summary) == 0:
        st.info("In den Tabellen wurden keine Patienten gefunden.")
        return
    lights = dict(zip(summary["patient"], summary["light"]))
    risks = dict(zip(summary["patient"], summary["risk"]))

    c1, c2, c3 = st.columns(3)
    patient = c1.selectbox(
        "Patient", list(summary["patient"]), key="pat_patient",
        format_func=lambda p: f"{LIGHT_ICONS[lights[p]]} {p} · Risiko {fmt_pct(risks[p])}",
        help="Sortiert nach aktuellem Risiko, höchstes zuerst.")
    group = c2.selectbox("Merkmal", FEATURE_GROUPS, format_func=group_name, key="pat_group")
    metric = c3.selectbox(
        "Kennzahl", METRICS, format_func=lambda m: METRIC_LABELS[m], key="pat_metric",
        help="Absolutwert: der gemessene Wert. CV: Unruhe innerhalb einer Messung. "
             "Abweichung: Abstand vom persönlichen Normal.")

    st.subheader(f"{feature_label(group, metric)} im Verlauf")
    fig = value_chart(result, patient, group, metric)
    if fig is None:
        st.info("Für dieses Merkmal liegen bei diesem Patienten keine Messwerte vor.")
    else:
        st.plotly_chart(fig, use_container_width=True, key="pat_value")
        st.caption("Grün hinterlegt: der persönliche Normalbereich dieses Patienten. Senkrechte Linien: "
                   "Biopsien, gefärbt nach Befund. Lücken in der Linie: Tage ohne Messung.")

    st.subheader("Risikoverlauf")
    fig = risk_chart(result, patient)
    if fig is None:
        reason = (result.model_report.reason if result.model_report.status != "ok"
                  else "Für diesen Patienten liegen zu wenige Messtage vor.")
        st.info(f"Kein Risikoverlauf verfügbar. {reason}")
    else:
        st.plotly_chart(fig, use_container_width=True, key="pat_risk")

    st.subheader("Biopsien dieses Patienten")
    table = biopsy_table(result, patient)
    if len(table) == 0:
        st.info("Für diesen Patienten sind keine Biopsien hinterlegt.")
    else:
        helps = {
            "Datentage im Fenster": f"Tage mit Messung in den {result.cfg.window_days} Tagen vor der Biopsie.",
            "Verwertbar": "Nicht verwertbar sind Biopsien mit zu wenigen Messtagen davor und "
                          "Kontrollbiopsien kurz nach einer Abstoßung (Behandlung verändert das EKG)."}
        show_table(table, help=helps, column_config={
            "Datum": st.column_config.DateColumn("Datum", format="DD.MM.YYYY"),
            "Datentage im Fenster": st.column_config.NumberColumn(help=helps["Datentage im Fenster"]),
            "Verwertbar": st.column_config.TextColumn(help=helps["Verwertbar"])})

    _explain_section(result, patient)
