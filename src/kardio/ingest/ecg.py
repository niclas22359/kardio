"""Station 1a: EKG-Tabelle einlesen und auf eine Messung pro Patient und Tag reduzieren."""
from __future__ import annotations
import re
import pandas as pd
from kardio.constants import VALUE_COLUMNS, COL_PATIENT, COL_DATE, COL_INCLUDED, COL_CC


class IngestError(ValueError):
    """Verstaendliche Fehlermeldung fuer den Nutzer (Datei passt nicht)."""


MAX_ROWS = 500_000   # weit ueber jeder realistischen Klinik-Tabelle; schuetzt vor Ueberlast durch Riesendateien

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def normalize_patient(s) -> str:
    return str(s).strip().upper()


def read_table(path_or_buffer) -> pd.DataFrame:
    """CSV mit unbekanntem Trennzeichen (Semikolon, Komma, Tab) lesen."""
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    try:
        # keep_default_na=False: ein Patientenkuerzel wie "NA" darf nicht als leere Zelle gelesen werden
        df = pd.read_csv(path_or_buffer, sep=None, engine="python", encoding="utf-8-sig",
                         keep_default_na=False, na_values=[""])
    except Exception as e:  # noqa: BLE001
        raise IngestError(f"Datei konnte nicht gelesen werden: {e}") from e
    if len(df) > MAX_ROWS:   # Groessengrenze gleich am Eingang, vor jeder weiteren Verarbeitung
        raise IngestError(f"Die Tabelle hat {len(df)} Zeilen. Erlaubt sind höchstens {MAX_ROWS}.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_ecg(path_or_buffer) -> pd.DataFrame:
    """Rohtabelle -> eine Zeile pro Messung mit patient, date, included_no, cc und 44 Werten."""
    raw = read_table(path_or_buffer)
    missing = [c for c in ["Name", "Date"] + VALUE_COLUMNS if c not in raw.columns]
    if missing:
        more = f" (und {len(missing) - 6} weitere)" if len(missing) > 6 else ""
        raise IngestError("EKG-Tabelle: fehlende Spalten: " + ", ".join(missing[:6]) + more)
    out = pd.DataFrame()
    out[COL_PATIENT] = raw["Name"].map(normalize_patient)
    dates = raw["Date"].astype(str).str.extract(DATE_RE)[0]
    if dates.isna().any():
        raise IngestError("EKG-Tabelle: Datum konnte nicht aus Spalte 'Date' gelesen werden "
                          "(erwartet z.B. ecg_2025-08-28_1.csv)")
    out[COL_DATE] = pd.to_datetime(dates)
    if "included_no" in raw.columns:
        out[COL_INCLUDED] = pd.to_numeric(raw["included_no"], errors="coerce").fillna(0).astype(int)
    else:
        out[COL_INCLUDED] = 0
    out[COL_CC] = pd.to_numeric(raw["cc"], errors="coerce") if "cc" in raw.columns else float("nan")
    for c in VALUE_COLUMNS:
        col = raw[c]
        if col.dtype == object:  # Dezimalkomma abfangen
            col = col.astype(str).str.replace(",", ".", regex=False)
        out[c] = pd.to_numeric(col, errors="coerce")
    return out


def select_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pro Patient und Tag bleibt die vollstaendigste Messung.

    Reihenfolge: wenigste leere Felder, dann meiste verwertete Herzschlaege, dann hoechste Qualitaet.
    Rueckgabe: (taegliche Tabelle, verworfene Mehrfachmessungen).
    """
    d = df.copy()
    d["_nan"] = d[VALUE_COLUMNS].isna().sum(axis=1)
    d = d.sort_values([COL_PATIENT, COL_DATE, "_nan", COL_INCLUDED, COL_CC],
                      ascending=[True, True, True, False, False], kind="mergesort")
    keep = ~d.duplicated([COL_PATIENT, COL_DATE], keep="first")
    daily = d[keep].drop(columns="_nan").reset_index(drop=True)
    dropped = d[~keep].drop(columns="_nan").reset_index(drop=True)
    return daily, dropped
