"""Abschnitt der Merkmalsseite: Prüfungen, die mehrere Biopsien je Patient berücksichtigen."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kardio.constants import GRADE_LABELS, METRIC_LABELS
from kardio.stats.patient_level import paired_differences

from ui_common import AGG_LABELS, base_layout, column_label, fmt_p, group_name, show_table


def patient_level_table(ranking: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    top = ranking.sort_values(["p_perm_adj", "p_perm", "p_adj"], na_position="last").head(n)
    return pd.DataFrame({
        "Merkmal": [group_name(g) for g in top["group"]],
        "Kennzahl": [METRIC_LABELS.get(m, m) for m in top["metric"]],
        "Verdichtung": [AGG_LABELS.get(a, a) for a in top["agg"]],
        "p einfach": [fmt_p(p) for p in top["p_adj"]],
        "p Patienten berücksichtigt": [fmt_p(p) for p in top["p_perm_adj"]],
        "p innerhalb Patient": [fmt_p(p) for p in top["p_within_adj"]],
        "Patienten mit beidem": top["n_paired"].fillna(0).astype(int).to_numpy(),
        "Gleiche Richtung": ["–" if pd.isna(v) else f"{v * 100:.0f} %" for v in top["share_same_direction"]],
    })


def paired_chart(windows: pd.DataFrame, column: str) -> go.Figure | None:
    """Je Patient eine Linie: sein Mittel ohne Abstoßung (links) zu seinem Mittel vor Abstoßung (rechts)."""
    d = paired_differences(windows, column)
    if len(d) == 0:
        return None
    fig = go.Figure()
    labels = [GRADE_LABELS[0], GRADE_LABELS[2]]
    for patient, mean_no, mean_rej, diff in d.itertuples(index=False):
        fig.add_trace(go.Scatter(
            x=labels, y=[mean_no, mean_rej], mode="lines+markers", showlegend=False,
            line=dict(color="#1f77b4" if diff < 0 else "#d62728", width=1.5),
            marker=dict(size=6), opacity=0.65, name=str(patient),
            hovertemplate="%{fullData.name}<br>%{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=[d["mean_no"].median(), d["mean_rej"].median()],
                             mode="lines+markers", name="Median aller Patienten",
                             line=dict(color="black", width=4), marker=dict(size=10)))
    base_layout(fig, height=420)
    fig.update_yaxes(title_text=column_label(column) + " (Abstand zum persönlichen Normal)")
    return fig


def render_section(result) -> None:
    rk = result.ranking
    st.subheader("Mehrere Biopsien je Patient berücksichtigt")
    st.markdown(
        "Die Diagramme oben behandeln jede Biopsie als eigenen Fall. Ein Patient mit zehn Biopsien zählt "
        "dort zehnmal, die p-Werte fallen dadurch zu günstig aus. Zwei Prüfungen korrigieren das:\n"
        "- **Patienten berücksichtigt:** Die Befunde werden nur innerhalb eines Patienten zufällig "
        "vertauscht. Alle Biopsien bleiben drin, der p-Wert ist ehrlich.\n"
        "- **Innerhalb Patient:** Je Patient wird sein Mittel vor Abstoßungen mit seinem Mittel vor "
        "unauffälligen Biopsien verglichen. Jeder Patient zählt genau einmal und ist seine eigene "
        "Kontrolle. Patienten ohne Abstoßung oder ohne unauffällige Biopsie fallen hier heraus.")
    n_simple = int((rk["p_adj"] < 0.05).sum())
    n_perm = int((rk["p_perm_adj"] < 0.05).sum())
    n_within = int((rk["p_within_adj"] < 0.05).sum())
    n_pairs = int(rk["n_paired"].max()) if rk["n_paired"].notna().any() else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Auffällig, einfacher Test", n_simple,
              help="Kennzahlen mit korrigiertem p < 0.05, jede Biopsie als eigener Fall.")
    c2.metric("Auffällig, Patienten berücksichtigt", n_perm,
              help="Kennzahlen mit korrigiertem p < 0.05 im Vertauschungstest.")
    c3.metric("Auffällig, innerhalb Patient", n_within,
              help="Kennzahlen mit korrigiertem p < 0.05 im Vergleich je Patient.")
    c4.metric("Patienten mit beiden Befunden", n_pairs,
              help="Nur diese Patienten können in den beiden Prüfungen etwas beitragen.")
    if rk["p_perm"].notna().sum() == 0:
        st.info("Für diese Prüfungen braucht es Patienten, die sowohl eine Biopsie mit als auch eine ohne "
                "Abstoßung haben. Im aktuellen Datensatz gibt es dafür zu wenige.", icon="ℹ️")
        return
    if n_simple > 0 and n_perm == 0:
        st.warning("**Die Auffälligkeiten des einfachen Tests halten der Prüfung auf Patientenebene nicht "
                   "stand.** Sie könnten von einzelnen Patienten mit vielen Biopsien stammen.", icon="⚠️")
    helps = {
        "p einfach": "Korrigierter p-Wert, jede Biopsie als eigener Fall (wie in den Diagrammen oben).",
        "p Patienten berücksichtigt": "Korrigierter p-Wert, wenn die Befunde nur innerhalb eines Patienten "
                                      "vertauscht werden.",
        "p innerhalb Patient": "Korrigierter p-Wert für die Differenz je Patient (Wilcoxon-Test). "
                               "„–“ bei weniger als 6 Patienten mit beiden Befunden.",
        "Patienten mit beidem": "Patienten mit mindestens einer Biopsie mit und einer ohne Abstoßung.",
        "Gleiche Richtung": "Anteil dieser Patienten, bei denen sich der Wert in dieselbe Richtung verschiebt.",
    }
    config = {name: st.column_config.Column(name, help=text) for name, text in helps.items()}
    show_table(patient_level_table(rk), column_config=config, help=helps)

    options = rk.sort_values(["p_within", "p_perm"], na_position="last")["column"].tolist()
    column = st.selectbox("Verlauf je Patient für", options, format_func=column_label, key="feat_paired_col")
    fig = paired_chart(result.windows, column)
    if fig is None:
        st.info("Für dieses Merkmal gibt es keinen Patienten mit beiden Befunden.")
    else:
        st.plotly_chart(fig, use_container_width=True, key="feat_paired")
        st.caption("Jede dünne Linie ist ein Patient: links sein Durchschnitt vor unauffälligen Biopsien, "
                   "rechts sein Durchschnitt vor Abstoßungen. Blau = bei ihm niedriger vor Abstoßung, "
                   "rot = höher. Laufen fast alle Linien in dieselbe Richtung, ist das Muster nicht von "
                   "Einzelnen getrieben.")
