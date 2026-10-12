"""Reiter 5: Guete und Gewichte des Risikomodells, Grenzen der Auswertung."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui_common import base_layout, column_label, fmt_num, fmt_pct, get_result


def weights_chart(weights, n: int = 15) -> go.Figure | None:
    if weights is None or len(weights) == 0:
        return None
    top = weights.head(n).iloc[::-1]
    positive = top["weight"].to_numpy() > 0
    fig = go.Figure(go.Bar(
        x=top["weight"], y=[column_label(c) for c in top["column"]], orientation="h",
        marker=dict(color=np.where(positive, "#d32f2f", "#2e7d32"), line=dict(color="black", width=1)),
        hovertemplate="%{y}<br>Gewicht %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="black", width=1.5))
    base_layout(fig, height=34 * len(top) + 130, showlegend=False)
    fig.update_layout(margin=dict(l=20, r=30, t=30, b=50))
    fig.update_xaxes(title_text="Gewicht im Modell (rechts = erhöht das Risiko, links = senkt es)")
    fig.update_yaxes(automargin=True)
    return fig


def _limits(result) -> None:
    with st.container(border=True):
        st.markdown("#### Grenzen dieser Auswertung")
        points = []
        if result.is_demo:
            points.append("Die **Demo-Daten sind erfunden**. Die Zahlen zeigen, dass das Verfahren "
                          "funktioniert, nicht, dass der Zusammenhang beim Menschen besteht.")
        else:
            points.append("Solange nur wenige Patienten und Abstoßungen vorliegen, sind alle Zahlen "
                          "**vorläufig**. Die Demo-Daten des Programms sind erfunden und belegen nichts.")
        points += [
            "Mehrere Biopsien **desselben Patienten sind nicht unabhängig** voneinander. Die p-Werte "
            "wirken dadurch sicherer, als sie sind.",
            "Das Verfahren ist **rückblickend**: Es wurde an bereits bekannten Befunden entwickelt und "
            "muss an neuen Patienten bestätigt werden.",
            "Das Ergebnis ist **keine Therapieempfehlung** und ersetzt weder die Biopsie noch die "
            "ärztliche Beurteilung.",
        ]
        st.markdown("\n".join(f"- {p}" for p in points))


def render() -> None:
    result = get_result()
    if result is None:
        return
    model, cfg = result.model_report, result.cfg

    if model.status != "ok":
        st.info(f"**Kein Risikomodell möglich.** {model.reason}", icon="ℹ️")
        st.markdown(
            f"**Was nötig wäre:** mindestens {cfg.min_windows_total} verwertbare Biopsien, davon "
            f"mindestens {cfg.min_rejection_windows} mit und {cfg.min_rejection_windows} ohne Abstoßung, "
            f"verteilt auf mehrere Patienten. Verwertbar ist eine Biopsie, wenn in den "
            f"{cfg.window_days} Tagen davor an mindestens {cfg.min_window_data_days} Tagen ein EKG "
            f"vorliegt.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Verwertbare Biopsien (ohne Zwischengruppe)", model.n_windows)
        c2.metric("davon Abstoßungen", model.n_rejections)
        c3.metric("Patienten", model.n_patients)
        st.caption("Der Gruppenvergleich im Reiter »Merkmale« läuft unabhängig vom Modell.")
        _limits(result)
        return

    ci = (f"95-%-Bereich {fmt_num(model.auc_ci[0])} bis {fmt_num(model.auc_ci[1])}"
          if model.auc_ci else "95-%-Bereich nicht bestimmbar")
    red = f"{model.threshold * 100:.0f} %"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Modellgüte (AUC)", fmt_num(model.auc_cv),
              help="0.5 = Zufall, 1.0 = perfekte Trennung von Abstoßung und keiner Abstoßung.")
    k1.caption(ci)
    k2.metric("Sensitivität", fmt_pct(model.sensitivity),
              help=f"Anteil der Abstoßungen, bei denen das Risiko vorher über der roten Schwelle ({red}) lag.")
    k3.metric("Spezifität", fmt_pct(model.specificity),
              help=f"Anteil der Biopsien ohne Abstoßung, bei denen das Risiko unter der roten Schwelle ({red}) blieb.")
    k4.metric("Vergleichsmodell (AUC)", fmt_num(model.auc_gb),
              help="Ein zweites, komplizierteres Rechenverfahren zur Kontrolle. Liegt es ähnlich, "
                   "reicht das einfache, erklärbare Modell aus.")
    st.caption(f"Grundlage: {model.n_windows} verwertbare Biopsien (ohne Zwischengruppe), davon "
               f"{model.n_rejections} mit Abstoßung, von {model.n_patients} Patienten.")

    with st.expander("Was bedeuten diese Zahlen?", expanded=True):
        st.markdown(
            f"- **Modellgüte (AUC):** Greift man zufällig eine Biopsie mit und eine ohne Abstoßung "
            f"heraus, gibt das Modell in so vielen Fällen der Abstoßung das höhere Risiko.\n"
            f"- **Sensitivität:** Wie viele Abstoßungen hätte die rote Ampel (ab {red}) angezeigt?\n"
            f"- **Spezifität:** Wie oft blieb die Ampel bei Biopsien ohne Abstoßung zu Recht unter Rot?\n"
            f"- **Ehrliche Prüfung:** Das Modell wird immer an Patienten geprüft, die es beim Lernen "
            f"nicht gesehen hat. Dazu werden reihum **ganze Patienten** beiseitegelegt. So lernt es "
            f"keine einzelnen Personen auswendig.\n"
            f"- Die **Zwischengruppe** bleibt beim Lernen außen vor und wird nur mitbewertet.")

    st.subheader("Worauf achtet das Modell?")
    fig = weights_chart(model.weights)
    if fig is None:
        st.info("Für dieses Modell liegen keine Gewichte vor.")
    else:
        st.plotly_chart(fig, use_container_width=True, key="model_weights")
        st.caption("Die 15 stärksten Gewichte. Rot: Steigt dieser Wert in den Tagen vor der Biopsie, "
                   "steigt das Risiko. Grün: Steigt dieser Wert, sinkt das Risiko.")
    _limits(result)
