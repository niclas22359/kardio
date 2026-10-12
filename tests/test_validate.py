import io
import numpy as np
import pandas as pd
import pytest
from kardio.config import load_generator_config
from kardio.generator.demo import generate_demo
from kardio.ingest.biopsies import load_biopsies
from kardio.ingest.ecg import load_ecg
from kardio.pipeline import run_analysis_frames
from kardio.validate import lead_time_table, run_validation_frames


def frames(seed, n, pattern=True, prefix="D"):
    cfg = load_generator_config()
    cfg.seed, cfg.n_patients, cfg.pattern_enabled = seed, n, pattern
    e, b, _ = generate_demo(cfg, include_real=False)
    e, b = e.copy(), b.copy()
    e["Name"] = e["Name"].str.replace("D", prefix, n=1)
    b["Patient"] = b["Patient"].str.replace("D", prefix, n=1)
    buf = io.StringIO(e.to_csv(sep=";", index=False))
    return load_ecg(buf), load_biopsies(b)


@pytest.fixture(scope="module")
def trained():
    return run_analysis_frames(*frames(101, 40, prefix="A"), data_source_label="DEMO A")


def test_frozen_model_works_on_independent_patients(trained):
    v = run_validation_frames(trained, *frames(202, 30, prefix="B"), data_source_label="DEMO B")
    m = v.metrics
    assert m.status == "ok" and m.auc > 0.85 and m.auc_ci[0] <= m.auc <= m.auc_ci[1]
    assert m.tp + m.fn == m.n_rejections and m.tp + m.fn + m.fp + m.tn == m.n_windows
    assert m.sensitivity > 0.6 and m.specificity > 0.8
    assert v.windows["risk"].dropna().between(0, 1).all()
    assert v.trained_on == "DEMO A" and v.metrics.train_auc == trained.model_report.auc_cv
    lead = lead_time_table(v)
    assert len(lead) == int((v.biopsies["grade"] == 2).sum())
    assert lead["first_red_days_before"].notna().mean() > 0.5


def test_weights_stay_frozen(trained):
    before = trained.model.named_steps["lr"].coef_.copy()
    run_validation_frames(trained, *frames(202, 10, prefix="B"))
    assert np.array_equal(before, trained.model.named_steps["lr"].coef_)


def test_null_dataset_is_not_recognised(trained):
    v = run_validation_frames(trained, *frames(303, 30, pattern=False, prefix="N"))
    assert v.metrics.status == "ok" and 0.35 < v.metrics.auc < 0.65


def test_without_biopsies_only_risks(trained):
    ecg, _ = frames(202, 5, prefix="B")
    v = run_validation_frames(trained, ecg, None)
    assert v.metrics.status == "unlabeled" and len(v.windows) == 0
    assert v.timeline["risk"].notna().sum() > 100
    assert len(lead_time_table(v)) == 0


def test_too_few_findings_is_flagged(trained):
    ecg, bx = frames(202, 3, prefix="B")
    v = run_validation_frames(trained, ecg, bx.head(4))
    assert v.metrics.status == "few" and "zu wenige" in v.metrics.note


def test_needs_a_trained_model(trained):
    import dataclasses
    with pytest.raises(ValueError):
        run_validation_frames(dataclasses.replace(trained, model=None), *frames(202, 3, prefix="B"))
