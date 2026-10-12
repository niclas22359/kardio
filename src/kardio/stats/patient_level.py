"""Tests, die beruecksichtigen, dass mehrere Biopsien desselben Patienten zusammenhaengen.

Der einfache Gruppenvergleich (compare.py) behandelt jede Biopsie als unabhaengigen Fall. Ein Patient
mit zehn Biopsien zaehlt dort zehnmal, die p-Werte fallen dadurch zu guenstig aus. Hier zwei Wege,
die das beheben:

1. Vergleich INNERHALB des Patienten: je Patient ein Mittelwert seiner Fenster vor Abstossung und einer
   seiner Fenster ohne Abstossung; getestet wird die Differenz (Wilcoxon-Vorzeichen-Rang-Test).
   Jeder Patient zaehlt genau einmal und ist seine eigene Kontrolle.
2. Vertauschungstest: die Befunde werden nur innerhalb eines Patienten zufaellig vertauscht. So bleibt
   erhalten, welche Fenster zu wem gehoeren, und der p-Wert ist ehrlich. Alle Daten bleiben drin.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests
from kardio.config import AnalysisConfig
from kardio.constants import WINDOW_FEATURE_COLUMNS

PATIENT_COLUMNS = ["column", "n_paired", "diff_within", "share_same_direction",
                   "p_within", "p_within_adj", "p_perm", "p_perm_adj"]
N_PERMUTATIONS = 10000   # kleinster moeglicher p-Wert 0,0001; noetig, damit die Korrektur ueber 132 Tests greifen kann
MIN_PERMUTATIONS = 200
MAX_WINDOWS = 50_000                 # darueber kein Vertauschungstest: harte Grenze, damit die Mindestzahl an
                                     # Vertauschungen die Obergrenze fuer die Gesamtarbeit nicht aushebeln kann
MAX_PERMUTATION_CELLS = 20_000_000   # Vertauschungen x Fenster insgesamt (begrenzt die Rechenzeit)
CHUNK_CELLS = 1_000_000              # Vertauschungen x Fenster je Portion (begrenzt den Speicher auf wenige MB)
MIN_PAIRS = 6          # darunter kann der Vorzeichen-Rang-Test nicht unter 0,05 fallen


def _prepare(windows: pd.DataFrame):
    w = windows
    if "usable" in w.columns:
        w = w[w["usable"].eq(True).to_numpy()]
    grade = pd.to_numeric(w["grade"], errors="coerce").to_numpy(dtype=float) if "grade" in w.columns \
        else np.full(len(w), np.nan)
    keep = (grade == 0) | (grade == 2)
    w = w[keep]
    y = (grade[keep] == 2).astype(float)
    patients = w["patient"].astype(str).to_numpy() if "patient" in w.columns else np.array([], dtype=str)
    x = w.reindex(columns=WINDOW_FEATURE_COLUMNS).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    order = np.argsort(patients, kind="stable")          # Fenster je Patient zusammenhaengend
    return x[order], y[order], patients[order]


def _bh(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def paired_differences(windows: pd.DataFrame, column: str) -> pd.DataFrame:
    """Je Patient mit beiden Befunden: Mittel ohne Abstossung, Mittel vor Abstossung, Differenz."""
    x, y, patients = _prepare(windows)
    if column not in WINDOW_FEATURE_COLUMNS or len(y) == 0:
        return pd.DataFrame(columns=["patient", "mean_no", "mean_rej", "diff"])
    v = x[:, WINDOW_FEATURE_COLUMNS.index(column)]
    df = pd.DataFrame({"patient": patients, "y": y, "v": v}).dropna(subset=["v"])
    g = df.groupby(["patient", "y"])["v"].mean().unstack("y")
    if 0.0 not in g.columns or 1.0 not in g.columns:
        return pd.DataFrame(columns=["patient", "mean_no", "mean_rej", "diff"])
    g = g.dropna(subset=[0.0, 1.0]).rename(columns={0.0: "mean_no", 1.0: "mean_rej"}).reset_index()
    g["diff"] = g["mean_rej"] - g["mean_no"]
    return g[["patient", "mean_no", "mean_rej", "diff"]]


def _within_patient(x: np.ndarray, y: np.ndarray, patients: np.ndarray):
    """Wilcoxon auf die Differenz je Patient (Mittel vor Abstossung minus Mittel ohne)."""
    n_cols = x.shape[1]
    n_paired = np.zeros(n_cols, dtype=int)
    diff_med = np.full(n_cols, np.nan)
    same_dir = np.full(n_cols, np.nan)
    p = np.full(n_cols, np.nan)
    if len(y) == 0:
        return n_paired, diff_med, same_dir, p
    codes, _ = pd.factorize(patients)
    n_pat = codes.max() + 1
    valid = ~np.isnan(x)
    xz = np.where(valid, x, 0.0)
    sums = {c: np.zeros((n_pat, n_cols)) for c in (0, 1)}
    cnts = {c: np.zeros((n_pat, n_cols)) for c in (0, 1)}
    for c in (0, 1):
        m = y == c
        np.add.at(sums[c], codes[m], xz[m])
        np.add.at(cnts[c], codes[m], valid[m].astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_no = sums[0] / cnts[0]
        mean_rej = sums[1] / cnts[1]
    diff = mean_rej - mean_no                         # NaN, wenn dem Patienten eine Seite fehlt
    for j in range(n_cols):
        d = diff[:, j]
        d = d[np.isfinite(d)]
        n_paired[j] = len(d)
        if len(d) == 0:
            continue
        diff_med[j] = float(np.median(d))
        nz = d[d != 0]
        if len(nz):
            same_dir[j] = float(max((nz > 0).mean(), (nz < 0).mean()))
        if len(nz) >= MIN_PAIRS:
            p[j] = float(wilcoxon(nz, alternative="two-sided").pvalue)
    return n_paired, diff_med, same_dir, p


def _permutation(x: np.ndarray, y: np.ndarray, patients: np.ndarray, seed: int, n_perm: int) -> np.ndarray:
    """p-Wert je Spalte; Befunde werden nur innerhalb eines Patienten vertauscht.

    Pruefgroesse: mittlerer Rang vor Abstossung minus mittlerer Rang ohne (rangbasiert wie Mann-Whitney).
    """
    n, n_cols = x.shape
    p = np.full(n_cols, np.nan)
    if n == 0 or n > MAX_WINDOWS or y.sum() < 2 or (1 - y).sum() < 2:
        return p
    codes, _ = pd.factorize(patients)
    # nur Patienten mit beiden Befunden koennen durch Vertauschen etwas aendern
    both = np.isin(codes, [c for c in np.unique(codes) if 0 < y[codes == c].sum() < (codes == c).sum()])
    if both.sum() < 4:
        return p
    valid = ~np.isnan(x)
    ranks = np.zeros_like(x)
    for j in range(n_cols):
        v = valid[:, j]
        if v.sum() >= 2:
            ranks[v, j] = rankdata(x[v, j])
    vf = valid.astype(float)

    def stat(Y: np.ndarray) -> np.ndarray:            # Y: (k, n) mit 0/1
        pos_cnt, neg_cnt = Y @ vf, (1 - Y) @ vf
        with np.errstate(invalid="ignore", divide="ignore"):
            return (Y @ ranks) / pos_cnt - ((1 - Y) @ ranks) / neg_cnt

    observed = stat(y[None, :])[0]
    rng = np.random.default_rng(seed)
    # Speicher und Rechenzeit begrenzen: eine riesige hochgeladene Tabelle darf den Dienst nicht lahmlegen.
    # Die Vertauschungen laufen in Portionen fester Groesse; bei sehr vielen Fenstern werden es weniger.
    n_perm = int(max(MIN_PERMUTATIONS, min(n_perm, MAX_PERMUTATION_CELLS // max(n, 1))))
    chunk = int(max(1, CHUNK_CELLS // max(n, 1)))
    extreme = np.zeros(n_cols)
    done = 0
    while done < n_perm:
        k = min(chunk, n_perm - done)
        keys = rng.random((k, n))
        Y = np.empty((k, n))
        for i in range(k):                            # Zeilen sind nach Patient sortiert: Mischen im Block
            Y[i] = y[np.lexsort((keys[i], codes))]
        with np.errstate(invalid="ignore"):
            extreme += (np.abs(stat(Y)) >= np.abs(observed)[None, :] - 1e-12).sum(axis=0)
        done += k
    p = (1.0 + extreme) / (n_perm + 1.0)
    constant = np.array([valid[:, j].sum() < 4 or np.ptp(x[valid[:, j], j]) == 0 for j in range(n_cols)])
    p[constant | ~np.isfinite(observed)] = np.nan
    return p


def patient_level_tests(windows: pd.DataFrame, cfg: AnalysisConfig,
                        n_perm: int = N_PERMUTATIONS) -> pd.DataFrame:
    """Eine Zeile je Fenster-Merkmal mit den Spalten aus PATIENT_COLUMNS."""
    if len(windows) > MAX_WINDOWS:                    # Grenze VOR jeder Rechenarbeit pruefen
        windows = windows.iloc[0:0]
    x, y, patients = _prepare(windows)
    n_paired, diff_med, same_dir, p_within = _within_patient(x, y, patients)
    p_perm = _permutation(x, y, patients, cfg.seed, n_perm)
    return pd.DataFrame({
        "column": WINDOW_FEATURE_COLUMNS, "n_paired": n_paired, "diff_within": diff_med,
        "share_same_direction": same_dir, "p_within": p_within, "p_within_adj": _bh(p_within),
        "p_perm": p_perm, "p_perm_adj": _bh(p_perm),
    })[PATIENT_COLUMNS]
