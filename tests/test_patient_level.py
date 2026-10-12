import numpy as np
import pandas as pd
from kardio.config import AnalysisConfig
from kardio.constants import WINDOW_FEATURE_COLUMNS
from kardio.stats.compare import compare_features
from kardio.stats.patient_level import paired_differences, patient_level_tests

COL = "R_rate_mean__zmean"


def clustered(n_pat=20, per_pat=10, effect=0.0, patient_sd=0.0, one_class_patients=False, seed=0):
    """Je Patient mehrere Fenster. patient_sd = individuelle Eigenheit, die alle Fenster eines Patienten teilen."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pat):
        offset = rng.normal(0, patient_sd, len(WINDOW_FEATURE_COLUMNS))
        if one_class_patients:      # ganze Patienten entweder nur mit oder nur ohne Abstossung
            grades = [2 if i % 2 else 0] * per_pat
        else:
            grades = [2 if k % 3 == 0 else 0 for k in range(per_pat)]
        for g in grades:
            v = offset + rng.normal(0, 1, len(WINDOW_FEATURE_COLUMNS))
            row = dict(zip(WINDOW_FEATURE_COLUMNS, v))
            if g == 2:
                row[COL] += effect
            rows.append({"patient": f"P{i}", "grade": g, "rejection": g == 2, "usable": True, **row})
    return pd.DataFrame(rows)


def test_real_effect_is_found_by_both_patient_level_tests():
    t = patient_level_tests(clustered(effect=1.5), AnalysisConfig()).set_index("column")
    assert t.loc[COL, "p_within_adj"] < 0.05 and t.loc[COL, "p_perm_adj"] < 0.05
    assert t.loc[COL, "n_paired"] == 20 and t.loc[COL, "diff_within"] > 1.0
    assert t.loc[COL, "share_same_direction"] > 0.9


def test_patient_peculiarity_fools_naive_test_but_not_patient_level_tests():
    """Kernfall der Frage aus dem Raum: Patienten mit Abstossungen haben von sich aus andere Werte.
    Der einfache Test meldet viele Scheintreffer, die Tests auf Patientenebene nicht."""
    w = clustered(n_pat=20, per_pat=10, patient_sd=2.0, one_class_patients=True, seed=3)
    rank, _ = compare_features(w, AnalysisConfig())
    assert (rank["p_adj"] < 0.05).sum() >= 20          # naiv: jede Menge "Signifikanz" aus dem Nichts
    assert rank["p_perm"].isna().all()                 # kein Patient hat beide Befunde -> keine Aussage moeglich
    assert (rank["n_paired"] == 0).all() and rank["p_within"].isna().all()


def test_null_with_patient_peculiarity_has_no_patient_level_hits():
    w = clustered(n_pat=25, per_pat=9, patient_sd=2.0, seed=5)
    t = patient_level_tests(w, AnalysisConfig())
    assert (t["p_perm_adj"] < 0.05).sum() == 0 and (t["p_within_adj"] < 0.05).sum() == 0
    assert 0.02 < (t["p_perm"] < 0.05).mean() < 0.10   # unkorrigiert rund 5 % wie es sein soll


def test_edge_cases_do_not_crash():
    cfg = AnalysisConfig()
    empty = clustered().iloc[0:0]
    t = patient_level_tests(empty, cfg)
    assert len(t) == len(WINDOW_FEATURE_COLUMNS) and t["p_perm"].isna().all()
    w = clustered(n_pat=3, per_pat=3, effect=1.0)
    w["P_duration_cv__slope"] = np.nan
    w["T_duration_cv__slope"] = 1.0
    t = patient_level_tests(w, cfg).set_index("column")
    assert np.isnan(t.loc["P_duration_cv__slope", "p_perm"]) and np.isnan(t.loc["T_duration_cv__slope", "p_perm"])
    assert t["p_within"].isna().all()                  # nur 3 Paare: Test nicht moeglich


def test_paired_differences_table():
    d = paired_differences(clustered(effect=1.5), COL)
    assert list(d.columns) == ["patient", "mean_no", "mean_rej", "diff"] and len(d) == 20
    assert np.allclose(d["diff"], d["mean_rej"] - d["mean_no"])
    assert len(paired_differences(clustered().iloc[0:0], COL)) == 0


def test_ranking_carries_patient_level_columns_and_is_deterministic():
    w = clustered(effect=1.5)
    a, _ = compare_features(w, AnalysisConfig())
    b, _ = compare_features(w, AnalysisConfig())
    assert {"p_within", "p_perm", "p_perm_adj", "n_paired"} <= set(a.columns) and len(a) == 132
    pd.testing.assert_frame_equal(a, b)


def test_large_table_stays_bounded():
    """Sehr viele Fenster: Vertauschungen werden gedeckelt und in Portionen gerechnet (kein Speicherberg)."""
    import time
    import tracemalloc
    from kardio.stats import patient_level as pl
    w = clustered(n_pat=400, per_pat=30, effect=0.5)        # 12.000 Fenster
    tracemalloc.start()
    t0 = time.time()
    t = patient_level_tests(w, AnalysisConfig())
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert time.time() - t0 < 60
    assert peak < 600 * 1024 * 1024
    assert pl.MAX_PERMUTATION_CELLS // 12000 < pl.N_PERMUTATIONS      # hier greift die Deckelung
    assert t.set_index("column").loc[COL, "p_perm"] < 0.01


def test_hard_limit_on_table_size(monkeypatch):
    """Ueber der harten Grenze laeuft der Vertauschungstest gar nicht erst an (Ergebnis leer statt Ueberlast)."""
    from kardio.stats import patient_level as pl
    assert pl.MIN_PERMUTATIONS * pl.MAX_WINDOWS <= pl.MAX_PERMUTATION_CELLS   # Mindestzahl sprengt die Obergrenze nie
    monkeypatch.setattr(pl, "MAX_WINDOWS", 50)
    t = pl.patient_level_tests(clustered(n_pat=20, per_pat=10, effect=1.5), AnalysisConfig())
    assert t["p_perm"].isna().all()
