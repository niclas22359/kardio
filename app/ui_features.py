"""Reiter 3: Gruppenvergleich je EKG-Merkmal im Stil der GraphPad-Prism-Auswertung der Klinik."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from kardio.config import load_generator_config
from kardio.constants import AGGS, FEATURE_GROUPS, GRADE_COLORS, GRADE_LABELS, METRIC_LABELS, METRICS

import ui_features_patient
from ui_common import (AGG_AXIS, AGG_LABELS, FONT, SIGNIFICANT_GREEN, base_layout, column_label,
                       feature_label, fmt_p, get_result, group_name, has_pattern, show_table)

N_COLS = 4
# Klammern wie in Prism: (linke Gruppe, rechte Gruppe, Spalte in der Rangliste, Hoehe ueber den Boxen)
BRACKETS = [(0, 1, "p_01", 0.10), (1, 2, "p_12", 0.25), (0, 2, "p_02", 0.40)]
HEADROOM = 0.56


def usable(windows: pd.DataFrame) -> pd.DataFrame:
    if "usable" not in windows.columns:
        return windows
    return windows[windows["usable"].eq(True).to_numpy()]


def _group_values(windows: pd.DataFrame, column: str) -> dict[int, np.ndarray]:
    if column not in windows.columns or len(windows) == 0:
        return {g: np.array([]) for g in (0, 1, 2)}
    v = pd.to_numeric(windows[column], errors="coerce").to_numpy(dtype=float)
    grade = pd.to_numeric(windows["grade"], errors="coerce").to_numpy(dtype=float)
    return {g: v[(grade == g) & np.isfinite(v)] for g in (0, 1, 2)}


def _axis_ref(index: int) -> str:
    return "" if index == 1 else str(index)


def box_grid(windows: pd.DataFrame, ranking: pd.DataFrame, metric: str, agg: str) -> go.Figure:
    """Raster aus Boxplots: je Merkmal drei Gruppen in Gruen, Gelb, Rot mit p-Werten als Klammern."""
    w = usable(windows)
    p_lookup = ranking.set_index("column") if len(ranking) else pd.DataFrame()
    n_rows = math.ceil(len(FEATURE_GROUPS) / N_COLS)
    titles = [feature_label(g, metric) for g in FEATURE_GROUPS]
    fig = make_subplots(rows=n_rows, cols=N_COLS, subplot_titles=titles,
                        vertical_spacing=0.13, horizontal_spacing=0.07)
    n_titles = len(fig.layout.annotations)

    for i, group in enumerate(FEATURE_GROUPS):
        row, col, ref = i // N_COLS + 1, i % N_COLS + 1, _axis_ref(i + 1)
        column = f"{group}_{metric}__{agg}"
        values = _group_values(w, column)
        for g in (0, 1, 2):
            fig.add_trace(go.Box(
                y=values[g], x=[g] * len(values[g]), name=GRADE_LABELS[g], width=0.6,
                fillcolor=GRADE_COLORS[g], line=dict(color="black", width=1.6),
                marker=dict(color=GRADE_COLORS[g], size=5, line=dict(color="black", width=0.5)),
                boxpoints="outliers", showlegend=False,
                hovertemplate=GRADE_LABELS[g] + "<br>%{y:.3f}<extra></extra>"), row=row, col=col)

        everything = np.concatenate([values[g] for g in (0, 1, 2)])
        lo, hi = (float(everything.min()), float(everything.max())) if len(everything) else (0.0, 1.0)
        span = (hi - lo) or max(abs(hi), 1.0)
        p_row = p_lookup.loc[column] if column in p_lookup.index else {}
        for a, b, key, level in BRACKETS:
            y = hi + level * span
            tick = 0.03 * span
            fig.add_shape(type="path", xref=f"x{ref}", yref=f"y{ref}",
                          path=f"M {a + 0.04},{y - tick} L {a + 0.04},{y} L {b - 0.04},{y} L {b - 0.04},{y - tick}",
                          line=dict(color="black", width=1.2))
            fig.add_annotation(x=(a + b) / 2, y=y, xref=f"x{ref}", yref=f"y{ref}",
                               text=fmt_p(p_row.get(key, np.nan)), showarrow=False, yshift=9,
                               font=dict(size=12, color="black"))
        fig.update_yaxes(range=[lo - 0.06 * span, hi + HEADROOM * span], row=row, col=col,
                         title_text=AGG_AXIS[agg] if col == 1 else None, title_font=dict(size=12))
        fig.update_xaxes(range=[-0.6, 2.6], tickmode="array", tickvals=[0, 1, 2],
                         ticktext=[GRADE_LABELS[g] for g in (0, 1, 2)], tickangle=-35,
                         row=row, col=col)
        p02 = p_row.get("p_02", np.nan)
        significant = pd.notna(p02) and float(p02) < 0.05
        fig.layout.annotations[i].update(
            text=f"<b>{titles[i]}</b>",
            font=dict(size=15, color=SIGNIFICANT_GREEN if significant else "black"))

    assert n_titles == len(FEATURE_GROUPS)
    fig.update_xaxes(showline=True, linecolor="black", linewidth=1.5, showgrid=False, ticks="outside",
                     tickfont=dict(color="black", size=12))
    fig.update_yaxes(showline=True, linecolor="black", linewidth=1.5, showgrid=False, ticks="outside",
                     zeroline=False, tickfont=dict(color="black", size=12))
    fig.update_layout(template="plotly_white", font=FONT, height=330 * n_rows + 60, boxgap=0.1,
                      margin=dict(l=70, r=20, t=50, b=90), plot_bgcolor="white")
    return fig


def ranking_table(ranking: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    top = ranking.head(n)
    return pd.DataFrame({
        "Merkmal": [group_name(g) for g in top["group"]],
        "Kennzahl": [METRIC_LABELS.get(m, m) for m in top["metric"]],
        "Verdichtung": [AGG_LABELS.get(a, a) for a in top["agg"]],
        "p korrigiert": [fmt_p(p) for p in top["p_adj"]],
        "Trennschärfe (AUC)": top["auc"].to_numpy(dtype=float),
        "Effektstärke": top["cliffs_delta"].to_numpy(dtype=float),
        "n keine Abstoßung": top["n_no"].to_numpy(),
        "n Zwischengruppe": top["n_mid"].to_numpy(),
        "n Abstoßung": top["n_rej"].to_numpy(),
    })


def auc_chart(ranking: pd.DataFrame, n: int = 15) -> go.Figure | None:
    top = ranking.head(n).dropna(subset=["auc"])
    if len(top) == 0:
        return None
    top = top.iloc[::-1]
    significant = top["p_adj"].lt(0.05).fillna(False)
    fig = go.Figure(go.Bar(
        x=top["auc"], y=[column_label(c) for c in top["column"]], orientation="h",
        marker=dict(color=np.where(significant, "#c62828", "#9e9e9e"), line=dict(color="black", width=1)),
        text=[f"{a:.2f}" for a in top["auc"]], textposition="outside",
        hovertemplate="%{y}<br>Trennschärfe (AUC): %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color="black", dash="dash", width=1.5))
    fig.add_annotation(x=0.5, y=1.0, yref="paper", text="Zufall", showarrow=False, yshift=12,
                       font=dict(size=13))
    base_layout(fig, height=max(320, 34 * len(top) + 120), showlegend=False)
    fig.update_layout(margin=dict(l=20, r=40, t=50, b=50))
    fig.update_xaxes(title_text="Trennschärfe (AUC)", range=[0.4, 1.05])
    fig.update_yaxes(automargin=True)
    return fig


def _pattern_box(result) -> None:
    cfg = st.session_state.get("demo_cfg") or load_generator_config()
    parts = []
    for col, rel in (cfg.pattern or {}).items():
        group = col[:-5] if col.endswith("_mean") else col
        parts.append(f"**{group_name(group)}** {'+' if rel > 0 else '−'}{abs(rel) * 100:.0f} %")
    if parts:
        st.success("In die Demo-Daten eingebaut (ab etwa "
                   f"{cfg.pattern_days_before} Tagen vor einer Abstoßung): " + " · ".join(parts)
                   + ". Zum Vergleich: Genau diese Merkmale sollte die Analyse oben in der Rangliste "
                     "finden.", icon="🎯")


def render() -> None:
    result = get_result()
    if result is None:
        return
    if not result.ranking_reliable:
        need = result.cfg.min_rejection_windows_stats
        st.warning("**Zu wenige Abstoßungen für eine belastbare Statistik.** Nötig sind mindestens "
                   f"{need} verwertbare Biopsien mit Abstoßung und {need} ohne. Die Diagramme zeigen, "
                   "was vorhanden ist; p-Werte sind nur als Orientierung zu lesen.", icon="⚠️")
    if has_pattern(result):
        _pattern_box(result)

    with st.expander("Wie lese ich diese Diagramme?"):
        st.markdown(
            f"**Ein Punkt = eine Biopsie.** Für jede Biopsie werden die {result.cfg.window_days} Tage "
            "davor zu einer Zahl verdichtet. Die Biopsien sind nach Befund in drei Kästen sortiert: "
            "keine Abstoßung (grün), Zwischengruppe (gelb), Abstoßung (rot).\n\n"
            "**Die drei Kennzahlen**\n"
            "- **Absolutwert:** der gemessene Wert selbst, z. B. die Höhe der R-Zacke, gemittelt über "
            "alle Herzschläge einer Messung.\n"
            "- **CV (Variationskoeffizient):** Streuung geteilt durch Mittelwert, berechnet über die "
            "Herzschläge **einer** Messung. Er sagt, wie unruhig der Wert von Schlag zu Schlag ist. "
            "0,10 bedeutet: die Schläge schwanken um etwa 10 % um ihren Mittelwert.\n"
            "- **Abweichung:** wie weit der Tageswert vom persönlichen Normal des Patienten entfernt "
            "liegt, in Prozent und ohne Vorzeichen. 0,05 bedeutet 5 % daneben, egal ob höher oder "
            "niedriger.\n\n"
            "**Die senkrechte Achse** zeigt nicht Millivolt oder Sekunden, sondern den Abstand zum "
            "persönlichen Normal des Patienten, gemessen in seiner eigenen üblichen Schwankung. "
            "0 = wie immer. −1 = um eine übliche Tagesschwankung niedriger als sonst. "
            "So werden Patienten mit von Natur aus hoher und niedriger R-Zacke vergleichbar.\n\n"
            "**Die Zahlen über den Klammern** sind p-Werte: die Wahrscheinlichkeit, einen so großen "
            "Unterschied zwischen zwei Gruppen rein zufällig zu sehen. Unter 0,05 gilt als auffällig, "
            "dann wird der Titel grün. Liegt der rote Kasten tiefer als der grüne, ist der Wert vor "
            "Abstoßungen niedriger als sonst.")

    c1, c2 = st.columns(2)
    metric = c1.selectbox(
        "Kennzahl", METRICS, format_func=lambda m: METRIC_LABELS[m], key="feat_metric",
        help="Absolutwert: der gemessene Wert. CV: wie unruhig der Wert innerhalb einer Messung ist. "
             "Abweichung: wie weit der Wert vom persönlichen Normal des Patienten entfernt liegt.")
    agg = c2.selectbox(
        f"Verdichtung des {result.cfg.window_days}-Tage-Fensters", AGGS,
        format_func=lambda a: AGG_LABELS[a], key="feat_agg",
        help="Wie die Tage vor der Biopsie zu einer Zahl zusammengefasst werden: Mittelwert der "
             "Abweichung, größte Abweichung, Trend über die Tage oder Anteil auffälliger Tage.")

    n_usable = len(usable(result.windows))
    st.plotly_chart(box_grid(result.windows, result.ranking, metric, agg), use_container_width=True,
                    key="feat_grid")
    st.caption(f"Grundlage: {n_usable} verwertbare von {len(result.windows)} Biopsien. Über den Boxen "
               "stehen die p-Werte der Gruppenvergleiche (Mann-Whitney-U). Ein grüner Titel bedeutet: "
               "„keine Abstoßung“ und „Abstoßung“ unterscheiden sich deutlich (p < 0.05). "
               "Abweichungen sind in Einheiten der persönlichen Schwankung angegeben "
               "(0 = wie gewohnt, 2 = deutlich auffällig).")

    st.subheader("Rangliste: Welche Merkmale trennen am besten?")
    if result.ranking["p_raw"].notna().sum() == 0:
        st.info("Für eine Rangliste fehlen Vergleichsgruppen: Es braucht mindestens zwei verwertbare "
                "Biopsien mit und zwei ohne Abstoßung.")
        return
    helps = {
        "p korrigiert": "Irrtumswahrscheinlichkeit für den Vergleich „keine Abstoßung“ gegen "
                        "„Abstoßung“, korrigiert dafür, dass sehr viele Merkmale gleichzeitig geprüft "
                        "werden. Unter 0.05 gilt als auffällig.",
        "Trennschärfe (AUC)": "Wie gut dieses eine Merkmal Abstoßung und keine Abstoßung "
                              "auseinanderhält: 0.5 = Zufall, 1.0 = perfekt.",
        "Effektstärke": "Cliffs Delta von −1 bis +1. Positiv: Wert ist bei Abstoßung höher. "
                        "Negativ: niedriger. Nahe 0: kein Unterschied.",
        "n keine Abstoßung": "Anzahl verwertbarer Biopsien ohne Abstoßung.",
        "n Zwischengruppe": "Anzahl verwertbarer Biopsien der Zwischengruppe.",
        "n Abstoßung": "Anzahl verwertbarer Biopsien mit Abstoßung.",
    }
    config = {name: st.column_config.NumberColumn(name, help=text) for name, text in helps.items()}
    config["p korrigiert"] = st.column_config.TextColumn("p korrigiert", help=helps["p korrigiert"])
    for name in ("Trennschärfe (AUC)", "Effektstärke"):
        config[name] = st.column_config.NumberColumn(name, format="%.2f", help=helps[name])
    show_table(ranking_table(result.ranking), column_config=config, help=helps,
               formats={"Trennschärfe (AUC)": lambda v: f"{v:.2f}", "Effektstärke": lambda v: f"{v:+.2f}"})
    fig = auc_chart(result.ranking)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, key="feat_auc")
        st.caption("Die 15 Merkmale mit dem kleinsten korrigierten p-Wert. Rot: auch nach Korrektur "
                   "auffällig (p < 0.05). Grau: könnte Zufall sein.")
    ui_features_patient.render_section(result)
