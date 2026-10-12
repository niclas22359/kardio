"""Analyse ohne Dashboard: python -m kardio.run --demo   oder   --ecg X --biopsies Y"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from kardio.config import ROOT, load_generator_config
from kardio.ingest.ecg import IngestError
from kardio.pipeline import run_analysis, save_processed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Kardio-Analyse ohne Dashboard")
    ap.add_argument("--ecg", help="EKG-Tabelle (CSV)")
    ap.add_argument("--biopsies", help="Biopsie-Tabelle (CSV)")
    ap.add_argument("--demo", action="store_true", help="Demo-Daten erzeugen und analysieren")
    ap.add_argument("--no-pattern", action="store_true", help="Demo ohne eingebautes Muster")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    a = ap.parse_args(argv)

    label = "Echte Daten"
    if a.demo:
        from kardio.generator.demo import write_demo
        cfg = load_generator_config()
        cfg.pattern_enabled = not a.no_pattern
        a.ecg, a.biopsies = write_demo(cfg, ROOT / "data" / "demo")
        label = "DEMO-DATEN " + ("mit eingebautem Muster" if cfg.pattern_enabled else "ohne Muster")
    if not a.ecg or not a.biopsies:
        ap.error("Entweder --demo oder --ecg und --biopsies angeben")

    try:
        r = run_analysis(a.ecg, a.biopsies, data_source_label=label)
    except IngestError as e:
        print(f"Fehler beim Einlesen: {e}", file=sys.stderr)
        return 2
    save_processed(r, Path(a.out))

    print(f"Datenbasis: {label}\n")
    for k, v in r.report.as_rows():
        print(f"{k:32} {v}")
    print("\nTop-Merkmale (Abstoßung gegen keine Abstoßung):")
    top = r.ranking.dropna(subset=["p_raw"]).head(10)
    if top.empty:
        print("  keine vergleichbaren Fenster")
    else:
        print(top[["column", "p_adj", "auc", "cliffs_delta", "n_no", "n_rej"]].round(4).to_string(index=False))
    if not r.ranking_reliable:
        print("  Hinweis: zu wenige Abstoßungen, Statistik nicht belastbar.")
    m = r.model_report
    if m.status == "ok":
        print(f"\nRisikomodell: AUC {m.auc_cv:.2f} (Kreuzvalidierung nach Patient), "
              f"{m.n_windows} Fenster, {m.n_rejections} Abstoßungen")
    else:
        print(f"\nRisikomodell: {m.reason}")
    print(f"\nZwischenergebnisse: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
