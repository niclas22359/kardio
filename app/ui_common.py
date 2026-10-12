"""Gemeinsame Bausteine fuer alle Reiter: Namen, Farben, Zahlenformate, Biopsie-Markierungen."""
from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kardio.constants import FEATURE_GROUPS, GRADE_COLORS, GRADE_LABELS, METRIC_LABELS
from kardio.model.risk import traffic_light

FEATURE_NAMES = {
    "R_amplitude": "R-Amplitude", "R_upslope": "R-Anstieg", "P_amplitude": "P-Amplitude",
    "T_amplitude": "T-Amplitude", "R_duration": "R-Dauer (QRS)", "P_duration": "P-Dauer",
    "T_duration": "T-Dauer", "RT_interval": "RT-Intervall", "qRT_interval": "qRT-Intervall",
    "PR_interval": "PR-Intervall", "R_rate": "Herzrate",
}
FEATURE_UNITS = {
    "R_amplitude": "µV", "R_upslope": "µV/s", "P_amplitude": "µV", "T_amplitude": "µV",
    "R_duration": "s", "P_duration": "s", "T_duration": "s", "RT_interval": "s",
    "qRT_interval": "s", "PR_interval": "s", "R_rate": "Schläge/min",
}
AGG_LABELS = {
    "zmean": "Mittlere Abweichung", "zmaxabs": "Größte Abweichung",
    "slope": "Trend", "alertfrac": "Anteil auffälliger Tage",
}
AGG_AXIS = {
    "zmean": "Mittlere Abweichung", "zmaxabs": "Größte Abweichung",
    "slope": "Trend pro Tag", "alertfrac": "Anteil auffälliger Tage",
}
LIGHT_ICONS = {"green": "🟢", "amber": "🟡", "red": "🔴", "grey": "⚪"}
LIGHT_ORDER = {"red": 0, "amber": 1, "green": 2, "grey": 3}
SIGNIFICANT_GREEN = "#00a000"
LINE_BLUE = "#1f4e79"
DATE_FORMAT = "%d.%m.%Y"
FONT = dict(family="Arial, sans-serif", size=14, color="#222222")
DEMO_WITH_PATTERN = "DEMO-DATEN mit eingebautem Muster"
DEMO_WITHOUT_PATTERN = "DEMO-DATEN ohne Muster"
EMPTY_TEXT = ("Noch keine Analyse – bitte zuerst im Reiter »Daten laden« Tabellen hochladen "
              "oder die Demo starten.")


def group_name(group: str) -> str:
    return FEATURE_NAMES.get(group, str(group).replace("_", " "))


def feature_label(group: str, metric: str) -> str:
    """'R_amplitude', 'cv' -> 'R-Amplitude CV'."""
    return f"{group_name(group)} {METRIC_LABELS.get(metric, metric)}"


def column_label(column: str) -> str:
    """'R_amplitude_mean__zmean' -> 'R-Amplitude Absolutwert · Mittlere Abweichung'."""
    feature, _, agg = str(column).partition("__")
    group, _, metric = feature.rpartition("_")
    if group not in FEATURE_GROUPS:
        return str(column)
    return f"{feature_label(group, metric)} · {AGG_LABELS.get(agg, agg)}"


def fmt_p(p) -> str:
    """p-Wert wie in GraphPad Prism: <0.0001, >0.9999, sonst vier Nachkommastellen; '–' wenn leer."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "–"
    if math.isnan(p):
        return "–"
    if p < 0.0001:
        return "<0.0001"
    if p > 0.9999:
        return ">0.9999"
    return f"{p:.4f}"


def fmt_num(x, digits: int = 2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if math.isnan(x) else f"{x:.{digits}f}"


def fmt_pct(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if math.isnan(x) else f"{x * 100:.0f} %"


def fmt_date(d) -> str:
    return "–" if d is None or pd.isna(d) else pd.Timestamp(d).strftime(DATE_FORMAT)


def has_pattern(result) -> bool:
    return result.is_demo and "mit eingebautem muster" in result.data_source_label.lower()


def show_banner(result) -> None:
    """Datenbasis immer sichtbar: Demo als Warnung, echte Daten als Hinweis."""
    if result is None:
        return
    label = f"{result.data_source_label} · ausgewertet: {result.cfg.window_days} Tage vor jeder Biopsie"
    if not result.is_demo:
        st.info(f"**{label}**", icon="ℹ️")
    elif has_pattern(result):
        st.warning(f"**{label}** – Keine medizinische Aussage. Zeigt nur, dass die Analyse ein "
                   "vorhandenes Muster findet.", icon="⚠️")
    else:
        st.warning(f"**{label}** – Keine medizinische Aussage. Gegenprobe: Ohne eingebautes Muster "
                   "darf die Analyse auch nichts finden.", icon="⚠️")


def get_result():
    """Analyseergebnis oder None; zeigt im zweiten Fall den freundlichen Leerhinweis."""
    result = st.session_state.get("result")
    if result is None:
        st.info(EMPTY_TEXT, icon="📂")
    return result


def patient_summary(result) -> pd.DataFrame:
    """Eine Zeile pro Patient, sortiert nach aktuellem Risiko (hoch zuerst, ohne Risiko zuletzt)."""
    daily, bx, tl = result.daily, result.biopsies, result.timeline
    days = daily.groupby("patient")["date"].agg(n_days="count", last="max")
    risk = (tl.dropna(subset=["risk"]).sort_values("date").groupby("patient")["risk"].last()
            if len(tl) else pd.Series(dtype=float))
    names = sorted(set(days.index) | set(bx["patient"].unique() if len(bx) else []))
    out = pd.DataFrame(index=pd.Index(names, name="patient"))
    out["risk"] = risk.reindex(names)
    out["n_days"] = days["n_days"].reindex(names).fillna(0).astype(int)
    out["last"] = days["last"].reindex(names)
    for col, grade in (("n_mid", 1), ("n_rej", 2)):
        counts = bx[bx["grade"] == grade].groupby("patient").size() if len(bx) else pd.Series(dtype=int)
        out[col] = counts.reindex(names).fillna(0).astype(int)
    total = bx.groupby("patient").size() if len(bx) else pd.Series(dtype=int)
    out["n_biopsies"] = total.reindex(names).fillna(0).astype(int)
    out["light"] = [traffic_light(r, result.cfg) for r in out["risk"]]
    out["_order"] = out["light"].map(LIGHT_ORDER)
    out = out.reset_index().sort_values(["_order", "risk", "patient"], ascending=[True, False, True],
                                        kind="stable")
    return out.drop(columns="_order").reset_index(drop=True)


def base_layout(fig: go.Figure, height: int = 380, **kwargs) -> go.Figure:
    fig.update_layout(template="plotly_white", font=FONT, height=height,
                      margin=dict(l=60, r=30, t=50, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      **kwargs)
    return fig


def date_axis(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(tickformat=DATE_FORMAT, hoverformat=DATE_FORMAT, title_text="Datum")
    return fig


def add_biopsy_markers(fig: go.Figure, biopsies: pd.DataFrame) -> go.Figure:
    """Senkrechte Linien an den Biopsie-Tagen, gefaerbt nach Befund, mit Legende."""
    if biopsies is None or len(biopsies) == 0:
        return fig
    for grade in (0, 1, 2):
        dates = biopsies.loc[biopsies["grade"] == grade, "date"]
        if len(dates) == 0:
            continue
        color = GRADE_COLORS[grade]
        for d in dates:
            fig.add_shape(type="line", x0=d, x1=d, y0=0, y1=1, xref="x", yref="y domain",
                          line=dict(color=color, width=2.5, dash="solid" if grade == 2 else "dash"),
                          layer="below")
        # Unsichtbare Spur nur fuer den Legendeneintrag
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=f"Biopsie: {GRADE_LABELS[grade]}",
                                 line=dict(color=color, width=2.5,
                                           dash="solid" if grade == 2 else "dash")))
    return fig


def gapless(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Auf lueckenlose Tagesachse bringen, damit fehlende Tage als Luecke in der Linie erscheinen."""
    s = frame.dropna(subset=["date"]).drop_duplicates("date").set_index("date")[value_col].sort_index()
    if len(s) == 0:
        return pd.DataFrame({"date": pd.to_datetime([]), value_col: pd.Series(dtype=float)})
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    return s.rename_axis("date").reset_index()


# ---------- Tabellen ----------
# st.dataframe braucht die Zusatzbibliothek "pyarrow". Auf manchen Windows-Rechnern blockiert eine
# Sicherheitsrichtlinie deren Programmdatei. Dann wird dieselbe Tabelle als schlichte HTML-Tabelle
# gezeigt, damit die App ueberall laeuft.
_ARROW_OK: bool | None = None

_TABLE_CSS = """
<style>
.kardio-table-wrap {overflow: auto; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 0.6rem;}
.kardio-table {border-collapse: collapse; width: 100%; font-size: 0.92rem;}
.kardio-table th {position: sticky; top: 0; background: #f5f6f8; text-align: left; font-weight: 600;
                  padding: 7px 12px; border-bottom: 1px solid #d0d0d0; white-space: nowrap;}
.kardio-table th[title] {cursor: help; text-decoration: underline dotted #999;}
.kardio-table td {padding: 6px 12px; border-bottom: 1px solid #eeeeee;}
.kardio-table td.num {text-align: right; font-variant-numeric: tabular-nums;}
.kardio-table tr:hover td {background: #fafafa;}
</style>
"""


def arrow_available() -> bool:
    global _ARROW_OK
    if _ARROW_OK is None:
        try:
            import pyarrow  # noqa: F401
            _ARROW_OK = True
        except Exception:  # noqa: BLE001  (ImportError oder blockierte Programmdatei)
            _ARROW_OK = False
    return _ARROW_OK


def _cell(value, fmt=None) -> tuple[str, bool]:
    """Zelleninhalt als Text und ob er rechtsbuendig (Zahl) gesetzt wird."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "–", False
    if fmt is not None:
        return str(fmt(value)), not isinstance(value, str)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return fmt_date(value), False
    if isinstance(value, (bool, np.bool_)):
        return ("ja" if value else "nein"), False
    if isinstance(value, (int, np.integer)):
        return str(int(value)), True
    if isinstance(value, (float, np.floating)):
        return (str(int(value)) if float(value).is_integer() else f"{value:.2f}"), True
    return str(value), False


def table_html(df: pd.DataFrame, help: dict | None = None, formats: dict | None = None,
               height: int | None = None) -> str:
    help, formats = help or {}, formats or {}
    head = "".join(
        f'<th title="{html.escape(help[c], quote=True)}">{html.escape(str(c))}</th>' if c in help
        else f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for row in df.itertuples(index=False, name=None):
        cells = []
        for col, value in zip(df.columns, row):
            text, numeric = _cell(value, formats.get(col))
            cells.append(f'<td class="num">{html.escape(text)}</td>' if numeric
                         else f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    style = f' style="max-height: {int(height)}px"' if height else ""
    return (f'{_TABLE_CSS}<div class="kardio-table-wrap"{style}><table class="kardio-table">'
            f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def show_table(df: pd.DataFrame, column_config: dict | None = None, help: dict | None = None,
               formats: dict | None = None, height: int | None = None) -> None:
    """Tabelle ueber die volle Breite; sortierbar, wenn moeglich, sonst als schlichte Tabelle.

    column_config gilt fuer die sortierbare Variante, help/formats fuer die schlichte."""
    if arrow_available():
        try:
            kwargs = {"height": min(height, 38 + 35 * len(df))} if height else {}
            st.dataframe(df, hide_index=True, use_container_width=True, column_config=column_config,
                         **kwargs)
            return
        except ImportError:
            pass
    st.markdown(table_html(df, help, formats, height), unsafe_allow_html=True)
