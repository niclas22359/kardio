"""Demo generator: real file format, switchable built-in pattern, reproducible."""
import io
import time

import numpy as np
import pandas as pd
import pytest

from kardio.config import load_generator_config
from kardio.constants import FEATURE_GROUPS, VALUE_COLUMNS
from kardio.generator.demo import generate_demo, write_demo
from kardio.ingest.ecg import load_ecg, select_daily

POSITIVE_GROUPS = ["R_amplitude", "R_upslope", "R_duration", "P_duration", "T_duration",
                   "RT_interval", "qRT_interval", "PR_interval", "R_rate"]
T_BLOCK = [c for c in VALUE_COLUMNS
           if c.startswith(("T_amplitude", "T_duration", "RT_interval", "qRT_interval"))]
REAL_NAMES = {"MC", "MW", "JM239"}


def make_cfg(n_patients=None, days=None, **kw):
    cfg = load_generator_config()
    if n_patients is not None:
        cfg.n_patients = n_patients
    if days is not None:
        cfg.days = days
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def read_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", float_precision="round_trip")


def load_ecg_frame(ecg: pd.DataFrame) -> pd.DataFrame:
    """Raw frame -> cleaned measurement table, through the real ingest code."""
    buf = io.StringIO()
    ecg.to_csv(buf, sep=";", index=False)
    buf.seek(0)
    return load_ecg(buf)


def rows_before(ecg: pd.DataFrame, dates_by_patient: dict, lo: int, hi: int) -> pd.Series:
    """Mask of rows lying lo..hi days before one of the given dates (negative = after)."""
    date = pd.to_datetime(ecg["Date"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0])
    mask = pd.Series(False, index=ecg.index)
    for pid, dates in dates_by_patient.items():
        for d in dates:
            delta = (pd.Timestamp(d) - date).dt.days
            mask |= (ecg["Name"] == pid) & delta.between(lo, hi)
    return mask


@pytest.fixture(scope="module")
def default_run():
    """Default config (40 patients x 365 days), demo patients only, with runtime."""
    cfg = make_cfg()
    t0 = time.perf_counter()
    ecg, bx, truth = generate_demo(cfg, include_real=False)
    return cfg, ecg, bx, truth, time.perf_counter() - t0


@pytest.fixture(scope="module")
def default_run_off():
    return generate_demo(make_cfg(pattern_enabled=False), include_real=False)


def test_header_identical_to_real_file(real_ecg_path, tmp_path):
    real_header = real_ecg_path.read_text(encoding="utf-8").splitlines()[0].split(";")
    ecg, bx, _ = generate_demo(make_cfg(3, 40), real_ecg_path)
    assert list(ecg.columns) == real_header
    assert list(bx.columns) == ["Patient", "Biopsie_Nr", "Datum", "Befund"]
    e, _ = write_demo(make_cfg(3, 40), tmp_path, real_ecg_path)
    assert e.read_text(encoding="utf-8").splitlines()[0].split(";") == real_header


def test_number_of_demo_patients(real_ecg_path):
    ecg, bx, truth = generate_demo(make_cfg(5, 120), real_ecg_path, include_real=False)
    expected = [f"D{i:03d}" for i in range(1, 6)]
    assert sorted(ecg["Name"].unique()) == expected
    assert sorted(bx["Patient"].unique()) == expected
    assert sorted(truth["rejection_dates"]) == expected
    assert sorted(truth["borderline_dates"]) == expected
    assert truth["pattern_enabled"] is True
    assert truth["pattern_columns"] == ["R_amplitude_mean", "qRT_interval_mean",
                                        "R_rate_mean", "T_amplitude_mean"]


def test_fixed_columns(default_run):
    _, ecg, _, _, _ = default_run
    full = ecg.dropna(subset=["cc"])
    assert (full[["hr", "rr", "rr_int"]] == 1).all().all()
    assert full["included_no"].between(14, 43).all()
    assert full["cc"].between(0.87, 0.97).all()
    assert ecg["Date"].str.fullmatch(r"ecg_\d{4}-\d{2}-\d{2}(_\d)?\.csv").all()
    assert not ecg.duplicated(["Name", "Date"]).any()


def test_befund_values_and_truth(default_run):
    _, _, bx, truth, _ = default_run
    assert bx["Befund"].isin(["nein", "Zwischengruppe", "ja"]).all()
    assert (bx["Befund"] == "Zwischengruppe").mean() > 0.05
    assert sum(map(len, truth["rejection_dates"].values())) == (bx["Befund"] == "ja").sum()
    assert sum(map(len, truth["borderline_dates"].values())) == (bx["Befund"] == "Zwischengruppe").sum()
    for pid, dates in truth["rejection_dates"].items():
        ja = bx[(bx["Patient"] == pid) & (bx["Befund"] == "ja")]["Datum"].tolist()
        assert [d.isoformat() for d in dates] == ja


def test_rejection_share_default_config(default_run):
    _, _, bx, _, _ = default_run
    rej = (bx["Befund"] == "ja").mean()
    print(f"default config: rejection share={rej:.3f} "
          f"borderline share={(bx['Befund'] == 'Zwischengruppe').mean():.3f} biopsies={len(bx)}")
    assert 0.15 <= rej <= 0.30


def test_rejection_share_structure(default_run, real_ecg_path):
    _, _, bx, _, _ = default_run
    assert 0.10 <= (bx["Befund"] == "ja").mean() <= 0.30  # default config, diluted by follow-ups
    # share among the regularly scheduled biopsies (follow-ups excluded) is the configured rate
    no_followups = make_cfg(followup_biopsies_after_rejection=0)
    plain = generate_demo(no_followups, real_ecg_path, include_real=False)[1]
    rate = no_followups.rejection_rate
    assert rate - 0.06 <= (plain["Befund"] == "ja").mean() <= rate + 0.06
    assert 0.10 <= (plain["Befund"] == "Zwischengruppe").mean() <= 0.20
    one_followup = make_cfg(followup_biopsies_after_rejection=1)
    diluted = generate_demo(one_followup, real_ecg_path, include_real=False)[1]
    assert 0.15 <= (diluted["Befund"] == "ja").mean() <= 0.30


def test_biopsy_numbers_dates_and_range(default_run):
    cfg, _, bx, _, _ = default_run
    for pid, g in bx.groupby("Patient"):
        dates = pd.to_datetime(g["Datum"], format="%Y-%m-%d")
        assert list(g["Biopsie_Nr"]) == list(range(1, len(g) + 1))
        assert dates.is_monotonic_increasing and dates.is_unique
        # transplant dates lie in the first 60 days of 2025; first biopsy on day 7
        assert pd.Timestamp("2025-01-08") <= dates.min() <= pd.Timestamp("2025-03-08")
        assert (dates.max() - dates.min()).days <= cfg.days - 1 - 7


def test_no_biopsy_after_last_day(real_ecg_path):
    cfg = make_cfg(12, 60, missing_day_rate=0.0)
    ecg, bx, _ = generate_demo(cfg, real_ecg_path, include_real=False)
    meas = load_ecg_frame(ecg)
    first, last = meas.groupby("patient")["date"].min(), meas.groupby("patient")["date"].max()
    assert ((last - first).dt.days == 59).all()
    for pid, g in bx.groupby("Patient"):
        dates = pd.to_datetime(g["Datum"])
        assert (dates.min() - first[pid]).days == 7
        assert dates.max() <= last[pid]


def test_no_new_rejection_during_followup_phase(default_run):
    cfg, _, bx, _, _ = default_run
    allowed = {k * cfg.followup_interval_days for k in range(1, cfg.followup_biopsies_after_rejection + 1)}
    n_followups_seen = 0
    for _, g in bx.groupby("Patient"):
        dates = pd.to_datetime(g["Datum"]).reset_index(drop=True)
        befund = g["Befund"].reset_index(drop=True)
        for i in befund.index[befund == "ja"]:
            delta = (dates - dates[i]).dt.days
            inside = (delta > 0) & (delta <= max(allowed))
            assert (befund[inside] == "nein").all()
            assert set(delta[inside]) <= allowed
            n_followups_seen += int(inside.sum())
    assert n_followups_seen > 0


def test_biopsy_schedule_dense_early_sparse_late(real_ecg_path):
    cfg = make_cfg(5, 365, rejection_rate=0.0)
    _, bx, _ = generate_demo(cfg, real_ecg_path, include_real=False)
    assert (bx["Befund"] != "ja").all()
    for _, g in bx.groupby("Patient"):
        gaps = pd.to_datetime(g["Datum"]).diff().dt.days.dropna().astype(int).tolist()
        assert gaps[:3] == [7, 7, 7]
        assert gaps[3:7] == [14, 14, 14, 14]
        assert 30 in gaps and gaps[-1] == 90
        assert gaps == sorted(gaps)


def test_roundtrip_through_ingest(tmp_path, real_ecg_path):
    cfg = make_cfg(6, 200)
    e, b = write_demo(cfg, tmp_path, real_ecg_path, include_real=False)
    daily, dropped = select_daily(load_ecg(e))
    assert len(dropped) > 0  # several measurements per day exist
    assert daily[VALUE_COLUMNS].isna().all(axis=1).sum() > 0  # completely empty days
    partial = daily[T_BLOCK].isna().all(axis=1) & daily["R_amplitude_mean"].notna()
    assert partial.sum() > 0  # rows without the T-wave block
    per_patient = daily.groupby("patient").size()
    assert per_patient.max() <= 200
    assert 0.84 <= per_patient.mean() / 200 <= 0.95  # ~10 % missing days
    bx = read_raw(b)
    assert len(bx) > 0 and set(bx["Patient"]) == set(daily["patient"])


def test_pattern_toggle_keeps_random_stream(real_ecg_path):
    cfg = make_cfg(10, 200)
    ecg_on, bx_on, truth_on = generate_demo(cfg, real_ecg_path)
    cfg.pattern_enabled = False
    ecg_off, bx_off, truth_off = generate_demo(cfg, real_ecg_path)
    assert bx_on.equals(bx_off)
    assert truth_on["rejection_dates"] == truth_off["rejection_dates"]
    assert truth_on["borderline_dates"] == truth_off["borderline_dates"]
    assert ecg_on["P_duration_mean"].equals(ecg_off["P_duration_mean"])
    assert not ecg_on["R_amplitude_mean"].equals(ecg_off["R_amplitude_mean"])
    touched = {f"{c.rsplit('_', 1)[0]}_{s}" for c in cfg.pattern for s in ("mean", "min", "max")}
    untouched = [c for c in ecg_on.columns if c not in touched]
    pd.testing.assert_frame_equal(ecg_on[untouched], ecg_off[untouched])
    assert truth_off["pattern_columns"] == [] and truth_off["pattern_enabled"] is False
    # real rows are never modified by the pattern
    pd.testing.assert_frame_equal(ecg_on[ecg_on["Name"].isin(REAL_NAMES)],
                                  ecg_off[ecg_off["Name"].isin(REAL_NAMES)])


def test_pattern_direction_and_size(default_run, default_run_off):
    cfg, ecg_on, _, truth, _ = default_run
    ecg_off = default_run_off[0]
    mask = rows_before(ecg_on, truth["rejection_dates"], 1, 3)
    assert mask.sum() > 50
    ratio = ecg_on.loc[mask, "R_amplitude_mean"].mean() / ecg_off.loc[mask, "R_amplitude_mean"].mean()
    print(f"\nR_amplitude_mean ratio 3 days before rejection: {ratio:.3f}")
    assert 0.85 <= ratio <= 0.92
    ratio_q = ecg_on.loc[mask, "qRT_interval_mean"].mean() / ecg_off.loc[mask, "qRT_interval_mean"].mean()
    assert 1.04 <= ratio_q <= 1.08
    # full strength exactly on the biopsy day
    on_day = rows_before(ecg_on, truth["rejection_dates"], 0, 0) & ecg_off["R_amplitude_mean"].notna()
    r0 = ecg_on.loc[on_day, "R_amplitude_mean"] / ecg_off.loc[on_day, "R_amplitude_mean"]
    assert np.allclose(r0, 1 + cfg.pattern["R_amplitude_mean"])
    # far away from any graded biopsy nothing changes
    near = rows_before(ecg_on, truth["rejection_dates"], -cfg.recovery_days, cfg.pattern_days_before)
    near |= rows_before(ecg_on, truth["borderline_dates"], -cfg.recovery_days, cfg.pattern_days_before)
    assert ecg_on.loc[~near, "R_amplitude_mean"].equals(ecg_off.loc[~near, "R_amplitude_mean"])


def test_borderline_pattern_is_weaker(default_run, default_run_off):
    cfg, ecg_on, _, truth, _ = default_run
    ecg_off = default_run_off[0]
    near_rej = rows_before(ecg_on, truth["rejection_dates"], -cfg.recovery_days, cfg.pattern_days_before)
    mask = rows_before(ecg_on, truth["borderline_dates"], 0, 0) & ~near_rej
    mask &= ecg_off["R_amplitude_mean"].notna()
    assert mask.sum() > 10
    ratio = ecg_on.loc[mask, "R_amplitude_mean"] / ecg_off.loc[mask, "R_amplitude_mean"]
    assert np.allclose(ratio, 1 + cfg.pattern["R_amplitude_mean"] * cfg.borderline_pattern_factor)


def test_reproducible(real_ecg_path):
    a = generate_demo(make_cfg(4, 90), real_ecg_path)
    b = generate_demo(make_cfg(4, 90), real_ecg_path)
    assert a[0].equals(b[0]) and a[1].equals(b[1]) and a[2] == b[2]
    c = generate_demo(make_cfg(4, 90, seed=7), real_ecg_path)
    assert not a[0]["R_amplitude_mean"].equals(c[0]["R_amplitude_mean"])


def test_real_patients_included_unchanged(real_ecg_path, tmp_path):
    cfg = make_cfg(3, 60)
    source = read_raw(real_ecg_path)
    ecg, bx, truth = generate_demo(cfg, real_ecg_path, include_real=True)
    assert REAL_NAMES <= set(ecg["Name"])
    real_rows = ecg[ecg["Name"].isin(REAL_NAMES)].reset_index(drop=True)
    assert len(real_rows) == len(source)
    assert real_rows[["Name", "Date"]].equals(source[["Name", "Date"]])
    num = [c for c in source.columns if c not in ("Name", "Date")]
    assert np.array_equal(real_rows[num].to_numpy(dtype=float, na_value=np.nan),
                          source[num].to_numpy(dtype=float), equal_nan=True)
    # also after writing to disk and reading back
    e, _ = write_demo(cfg, tmp_path, real_ecg_path, include_real=True)
    back = read_raw(e)
    back = back[back["Name"].isin(REAL_NAMES)].reset_index(drop=True)
    assert np.array_equal(back[num].to_numpy(dtype=float), source[num].to_numpy(dtype=float), equal_nan=True)
    # demo part is the same with and without real patients
    demo_only = generate_demo(cfg, real_ecg_path, include_real=False)
    assert REAL_NAMES.isdisjoint(demo_only[0]["Name"]) and REAL_NAMES.isdisjoint(demo_only[1]["Patient"])
    pd.testing.assert_frame_equal(ecg[~ecg["Name"].isin(REAL_NAMES)].reset_index(drop=True), demo_only[0])
    assert truth == demo_only[2]


def test_real_patient_biopsies_within_measured_range(real_ecg_path):
    ecg, bx, _ = generate_demo(make_cfg(2, 40), real_ecg_path)
    measured = load_ecg_frame(ecg)
    counts = bx[bx["Patient"].isin(REAL_NAMES)].groupby("Patient").size().to_dict()
    assert counts == {"MW": 4, "MC": 1, "JM239": 2}
    for pid in REAL_NAMES:
        g = bx[bx["Patient"] == pid]
        days = set(measured.loc[measured["patient"] == pid, "date"])
        dates = pd.to_datetime(g["Datum"])
        assert all(d in days for d in dates)  # only on measured dates
        assert list(g["Biopsie_Nr"]) == list(range(1, len(g) + 1))
        assert dates.is_monotonic_increasing
    mw = bx[bx["Patient"] == "MW"]
    april = mw[mw["Datum"].str.startswith("2024-04")]
    assert len(april) == 3 and (april["Befund"] == "ja").sum() == 1
    assert april.iloc[-1]["Befund"] == "ja" and april.iloc[-1]["Datum"] >= "2024-04-20"
    assert mw.iloc[-1]["Datum"] > "2024-04-30"
    assert (bx[bx["Patient"].isin({"MC", "JM239"})]["Befund"] == "nein").all()


def test_value_plausibility(default_run):
    _, ecg, _, _, _ = default_run
    for g in FEATURE_GROUPS:
        d = ecg.dropna(subset=[f"{g}_mean"])
        assert (d[f"{g}_min"] < d[f"{g}_mean"]).all(), g
        assert (d[f"{g}_mean"] < d[f"{g}_max"]).all(), g
        assert (d[f"{g}_sd"] > 0).all(), g
    for g in POSITIVE_GROUPS:
        d = ecg.dropna(subset=[f"{g}_mean"])
        assert (d[[f"{g}_mean", f"{g}_min", f"{g}_max"]] > 0).all().all(), g
    assert (ecg["P_amplitude_mean"].dropna() < 0).all()  # sign convention of the real data


def test_values_follow_real_levels_and_are_autocorrelated(default_run_off, real_ecg_path):
    ecg = default_run_off[0]
    real = load_ecg(real_ecg_path)
    mw = real[real["patient"] == "MW"].dropna(subset=VALUE_COLUMNS)
    for c in ["R_amplitude_mean", "P_duration_mean", "R_rate_mean", "R_amplitude_sd",
              "PR_interval_mean", "P_amplitude_mean", "T_amplitude_max", "R_rate_min"]:
        assert 0.7 < ecg[c].mean() / mw[c].median() < 1.3, c
    # every patient has an own level
    by_patient = ecg.groupby("Name")["P_duration_mean"]
    assert by_patient.mean().std() > 0.3 * by_patient.std().mean()
    # mild positive day-to-day autocorrelation
    daily, _ = select_daily(load_ecg_frame(ecg))
    acs = [g.set_index("date")["P_duration_mean"].asfreq("D").autocorr(lag=1)
           for _, g in daily.groupby("patient")]
    assert 0.15 < float(np.mean(acs)) < 0.6


def test_write_demo_writes_two_readable_files(tmp_path, real_ecg_path):
    out = tmp_path / "new" / "dir"
    e, b = write_demo(make_cfg(3, 60), out, real_ecg_path)
    assert e == out / "ecg_demo.csv" and b == out / "biopsies_demo.csv"
    assert e.exists() and b.exists()
    ecg, bx = read_raw(e), read_raw(b)
    assert len(ecg.columns) == 52 and len(ecg) > 100
    assert list(bx.columns) == ["Patient", "Biopsie_Nr", "Datum", "Befund"]
    assert pd.to_datetime(bx["Datum"], format="%Y-%m-%d").notna().all()
    # integer flags are written like in the real file ("1", not "1.0")
    lines = e.read_text(encoding="utf-8").splitlines()
    full = next(ln.split(";") for ln in lines[1:] if ln.split(";")[4] != "")
    assert full[0] == "0" and full[5:8] == ["1", "1", "1"] and full[3].isdigit()
    assert len(load_ecg(e)) == len(ecg)


def test_runtime_default_config(default_run):
    cfg, ecg, _, _, seconds = default_run
    print(f"\nruntime default config: {seconds:.2f}s, rows={len(ecg)}")
    assert ecg["Name"].nunique() == cfg.n_patients == 40
    assert seconds < 15


def test_works_without_real_file(tmp_path, real_ecg_path):
    """Online / frischer Checkout: keine echte Datei vorhanden -> aggregierte Kennwerte, keine echten Patienten."""
    from kardio.config import load_generator_config
    cfg = load_generator_config()
    cfg.n_patients, cfg.days = 4, 60
    missing = tmp_path / "gibt_es_nicht.csv"
    ecg, bx, _ = generate_demo(cfg, missing, include_real=True)
    assert set(ecg["Name"]) == {"D001", "D002", "D003", "D004"}
    assert set(bx["Patient"]) <= set(ecg["Name"])
    if real_ecg_path.exists():  # gleiche Kennwerte wie aus der echten Datei abgeleitet
        ref, _, _ = generate_demo(cfg, real_ecg_path, include_real=False)
        pd.testing.assert_frame_equal(ecg.reset_index(drop=True), ref.reset_index(drop=True),
                                      check_exact=False, rtol=1e-9)
