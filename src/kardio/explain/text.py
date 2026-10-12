"""Station 6: Befundtext fuer den Arzt.

Das Sprachmodell sucht selbst KEINE Muster. Es bekommt nur die bereits berechneten Zahlen
und formuliert daraus einen Text. Jede Zahl im Text wird danach gegen die gelieferten Zahlen
geprueft; taucht eine fremde Zahl auf, wird der Text verworfen und der Vorlagentext genutzt.
"""
from __future__ import annotations
import json
import math
import os
import re
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from kardio.constants import AGG_LABELS, FEATURE_GROUPS, FEATURE_LABELS, METRIC_LABELS
from kardio.model.risk import traffic_light

DEFAULT_BASE_URL = "https://chat.int.bayer.com/api/v2"
DEFAULT_MODEL = "claude-sonnet-5"
LIGHT_LABELS = {"green": "grün", "amber": "gelb", "red": "rot", "grey": "grau (nicht berechenbar)"}
DISCLAIMER = "Die Auswertung ersetzt keine Biopsie und keine ärztliche Beurteilung."

SYSTEM_PROMPT = (
    "Du schreibst für Kardiologen einen kurzen, nüchternen Befundtext auf Deutsch (150 bis 250 Wörter) "
    "zu einer EKG-Auswertung nach Herztransplantation. Gliedere in vier kurze Absätze mit den "
    "Überschriften: Datenlage, Auffällige Merkmale, Risiko, Einschränkungen.\n"
    "Strikte Regeln:\n"
    "1. Verwende ausschließlich die Zahlen aus den gelieferten Fakten, exakt so wie sie dort stehen. "
    "Rechne nichts um, runde nichts, erfinde keine Werte, keine Datumsangaben, keine Aufzählungsnummern.\n"
    "2. Nenne nur Merkmale, die in den Fakten stehen.\n"
    "3. Keine Therapieempfehlung, keine Diagnose.\n"
    "4. Wenn die Fakten sagen, dass die Datenbasis Demo-Daten sind oder die Statistik nicht belastbar ist, "
    "muss das im Absatz Einschränkungen klar stehen.\n"
    "5. Kein Markdown außer den vier Überschriften als fette Zeilen (**Überschrift**).\n"
    "6. Zitiere keine Feldnamen aus den Fakten (z.B. statistik_belastbar); formuliere in normalem Deutsch."
)

# Zahlen im Text. Ziffern in Kuerzeln (D001, JM239) zaehlen nicht, wohl aber "40ms" oder "2R".
_NUM_RE = re.compile(r"(?<![A-Za-z0-9_.,])\d+(?:[.,]\d+)*(?![0-9])")
# Zahlen, die in jedem Text vorkommen duerfen (Fensterlaenge, Befundstufen, 95-%-Intervall)
_ALWAYS_ALLOWED = {0.0, 1.0, 2.0, 3.0, 14.0, 95.0, 100.0}


class _SystemTrustAdapter(HTTPAdapter):
    """Prueft Zertifikate gegen den Zertifikatsspeicher des Betriebssystems.

    Noetig auf Rechnern, deren Virenschutz oder Firmen-Proxy verschluesselte Verbindungen mit einem
    eigenen Zertifikat prueft. Die Zertifikatspruefung bleibt eingeschaltet.
    """

    def init_poolmanager(self, *args, **kwargs):
        import ssl
        import truststore
        kwargs["ssl_context"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return super().init_poolmanager(*args, **kwargs)


def _session() -> requests.Session:
    s = requests.Session()
    try:
        import truststore  # noqa: F401
        s.mount("https://", _SystemTrustAdapter())
    except Exception:  # noqa: BLE001  ohne truststore gilt die Standardliste von requests
        pass
    return s


class MgaClient:
    """Minimaler Client fuer die OpenAI-kompatible myGenAssist-Schnittstelle."""

    def __init__(self, api_key: str, base_url: str | None = None, model: str | None = None,
                 timeout: float = 45.0):
        self.key = api_key
        self.base = (base_url or os.environ.get("MGA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("MGA_MODEL") or DEFAULT_MODEL
        self.timeout = timeout

    def chat(self, messages: list[dict]) -> str:
        r = _session().post(
            f"{self.base}/chat/completions", timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "max_tokens": 900, "temperature": 0.2})
        r.raise_for_status()
        data = r.json()
        # Eine falsche Modell-ID liefert HTTP 200 mit stillem Rueckfall auf ein anderes Modell.
        answered = str(data.get("model") or "")
        if answered and not answered.startswith(self.model):
            raise RuntimeError(f"Fallback-Modell {answered} statt {self.model}")
        return data["choices"][0]["message"]["content"].strip()


def _de(x: float, nd: int = 2) -> str:
    """Zahl in deutscher Schreibweise mit fester Stellenzahl."""
    return f"{x:.{nd}f}".replace(".", ",")


def _p_text(p: float) -> str:
    return "<0,0001" if p < 0.0001 else _de(p, 4)


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def build_facts(result, patient: str | None = None) -> dict:
    """Verdichtete, bereits formatierte Fakten. Nur diese Zahlen duerfen im Text vorkommen."""
    rep, mr, cfg = result.report, result.model_report, result.cfg
    facts: dict = {
        "datenbasis": result.data_source_label,
        "demo_daten": bool(result.is_demo),
        "patienten": int(rep.n_patients_ecg),
        "patienten_mit_biopsie": int(rep.n_patients_matched),
        "messtage": int(rep.n_days),
        "biopsien": int(rep.n_biopsies),
        "abstossungen": int(rep.n_rejections),
        "zwischengruppe": int(rep.n_borderline),
        "fenster_tage": int(cfg.window_days),
        "statistik_belastbar": bool(result.ranking_reliable),
    }

    sig = result.ranking[result.ranking["p_adj"] < 0.05].head(5)
    feats = []
    for _, r in sig.iterrows():
        feats.append({
            "merkmal": FEATURE_LABELS.get(r["group"], r["group"]),
            "kennzahl": METRIC_LABELS.get(r["metric"], r["metric"]),
            "verdichtung": AGG_LABELS.get(r["agg"], r["agg"]),
            "vor_abstossung": "höher" if r["cliffs_delta"] > 0 else "niedriger",
            "trennschaerfe_auc": _de(float(r["auc"])),
            "p_korrigiert": _p_text(float(r["p_adj"])),
        })
    facts["auffaellige_merkmale"] = feats
    facts["anzahl_signifikanter_kennzahlen"] = int((result.ranking["p_adj"] < 0.05).sum())

    model = {"status": "ok" if mr.status == "ok" else "zu wenig Daten"}
    if mr.status == "ok":
        model.update({"auc_kreuzvalidierung": _de(mr.auc_cv),
                      "fenster": int(mr.n_windows), "abstossungen": int(mr.n_rejections)})
        if mr.auc_ci and all(_finite(v) for v in mr.auc_ci):
            model["auc_intervall_95"] = f"{_de(mr.auc_ci[0])} bis {_de(mr.auc_ci[1])}"
        if _finite(mr.sensitivity) and _finite(mr.specificity):
            model["sensitivitaet_prozent"] = int(round(mr.sensitivity * 100))
            model["spezifitaet_prozent"] = int(round(mr.specificity * 100))
    else:
        model["grund"] = mr.reason
    facts["risikomodell"] = model

    if patient is not None:
        p: dict = {"kuerzel": safe_code(patient)}
        bx = result.biopsies[result.biopsies["patient"] == patient]
        p["biopsien"] = int(len(bx))
        p["abstossungen"] = int((bx["grade"] == 2).sum()) if len(bx) else 0
        p["messtage"] = int((result.daily["patient"] == patient).sum())
        t = result.timeline[result.timeline["patient"] == patient].dropna(subset=["risk"])
        risk = float(t.sort_values("date")["risk"].iloc[-1]) if len(t) else float("nan")
        p["risiko_prozent"] = int(round(risk * 100)) if math.isfinite(risk) else None
        p["ampel"] = LIGHT_LABELS[traffic_light(risk, cfg)]
        # staerkste aktuelle Abweichungen vom persoenlichen Normal (letzte 14 Tage)
        dev = result.deviations[(result.deviations["patient"] == patient)
                                & result.deviations["feature"].str.endswith("_mean")]
        if len(dev):
            last = dev["date"].max()
            recent = dev[dev["date"] >= last - pd.Timedelta(days=cfg.window_days)]
            z = recent.groupby("feature")["z"].mean().dropna()
            top = z.reindex(z.abs().sort_values(ascending=False).index).head(3)
            p["staerkste_abweichungen_vom_eigenen_normal"] = [
                {"merkmal": FEATURE_LABELS.get(f[:-5], f), "richtung": "erhöht" if v > 0 else "erniedrigt",
                 "abweichung_in_streuungen": _de(abs(float(v)), 1)}
                for f, v in top.items() if abs(v) >= 1.0]
        facts["patient"] = p
    return facts


def _numbers_in(obj) -> set[float]:
    """Alle Zahlen einsammeln, die in den Fakten vorkommen (auch in formatierten Texten)."""
    found: set[float] = set()
    if isinstance(obj, bool) or obj is None:
        return found
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            found.add(float(obj))
    elif isinstance(obj, str):
        for tok in _NUM_RE.findall(obj):
            v = _parse_number(tok)
            if v is not None:
                found.add(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found |= _numbers_in(v)
    return found


def _parse_number(tok: str) -> float | None:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", tok):       # 14.213 als Tausenderpunkt
        return float(tok.replace(".", ""))
    if tok.count(",") + tok.count(".") > 1:
        return None
    try:
        return float(tok.replace(",", "."))
    except ValueError:
        return None


def verify_numbers(text: str, facts: dict) -> bool:
    """True, wenn jede Zahl im Text aus den Fakten stammt."""
    allowed = _numbers_in(facts) | _ALWAYS_ALLOWED
    for tok in _NUM_RE.findall(text):
        v = _parse_number(tok)
        if v is None or not any(abs(v - a) < 1e-9 for a in allowed):
            return False
    return True


_UNSAFE_RE = re.compile(r"https?://|www\.|!\[|\]\(|<\s*/?\s*[A-Za-z!]|javascript:|data:", re.IGNORECASE)


def is_plain_text(text: str) -> bool:
    """Der KI-Text wird als formatierter Text angezeigt. Links, Bilder und HTML haben darin nichts
    verloren: ueber ein manipuliertes Patientenkuerzel koennte sonst Fremdinhalt nachgeladen werden."""
    return not _UNSAFE_RE.search(text)


def safe_code(patient) -> str:
    """Patientenkuerzel fuer den Prompt: nur Buchstaben, Ziffern, Binde- und Unterstrich."""
    return re.sub(r"[^A-Za-z0-9ÄÖÜäöüß_\-]", "", str(patient))[:24] or "unbekannt"


def template_text(f: dict) -> str:
    """Befundtext ohne Sprachmodell. Wird immer dann genutzt, wenn kein geprüfter KI-Text vorliegt."""
    lines = [
        "**Datenlage**",
        f"{f['datenbasis']}: {f['patienten']} Patienten mit {f['messtage']} Messtagen, "
        f"{f['biopsien']} Biopsien, davon {f['abstossungen']} mit Abstoßung und "
        f"{f['zwischengruppe']} in der Zwischengruppe. Verglichen wird das EKG der "
        f"{f['fenster_tage']} Tage vor jeder Biopsie mit dem persönlichen Normal des Patienten.",
        "",
        "**Auffällige Merkmale**",
    ]
    feats = f["auffaellige_merkmale"]
    if feats:
        for t in feats:
            lines.append(f"- {t['merkmal']} ({t['kennzahl']}, {t['verdichtung']}): vor Abstoßung "
                         f"{t['vor_abstossung']}, Trennschärfe {t['trennschaerfe_auc']}, "
                         f"p = {t['p_korrigiert']}")
    else:
        lines.append("Kein Merkmal unterscheidet sich nach Korrektur für Mehrfachtests zwischen "
                     "Biopsien mit und ohne Abstoßung.")
    lines += ["", "**Risiko**"]
    m = f["risikomodell"]
    if m["status"] == "ok":
        ci = f" (95-%-Intervall {m['auc_intervall_95']})" if "auc_intervall_95" in m else ""
        lines.append(f"Das Risikomodell erreicht in der Prüfung an nicht gesehenen Patienten eine "
                     f"Trennschärfe von {m['auc_kreuzvalidierung']}{ci}.")
    else:
        lines.append(m.get("grund") or "Für ein Risikomodell liegen zu wenig Daten vor.")
    p = f.get("patient")
    if p:
        if p["risiko_prozent"] is not None:
            lines.append(f"Patient {p['kuerzel']}: aktuelles Risiko {p['risiko_prozent']} %, Ampel {p['ampel']}.")
        else:
            lines.append(f"Patient {p['kuerzel']}: Risiko nicht berechenbar.")
        for d in p.get("staerkste_abweichungen_vom_eigenen_normal", []):
            lines.append(f"- {d['merkmal']} {d['richtung']} um {d['abweichung_in_streuungen']} "
                         f"Streuungen gegenüber dem eigenen Normal")
    lines += ["", "**Einschränkungen**"]
    limits = []
    if f["demo_daten"]:
        limits.append("Es handelt sich um erzeugte Demo-Daten. Das Ergebnis ist keine medizinische Aussage.")
    if not f["statistik_belastbar"]:
        limits.append("Für eine belastbare Statistik gibt es zu wenige Abstoßungen.")
    limits.append(DISCLAIMER)
    lines.append(" ".join(limits))
    return "\n".join(lines)


def _default_client():
    key = (os.environ.get("MGA_API_KEY") or "").strip()
    return MgaClient(key) if key else None


def llm_text(facts: dict, client=None) -> str | None:
    """Geprüfter KI-Text oder None (kein Schluessel, Netzfehler, fremde Zahl im Text)."""
    client = client or _default_client()
    if client is None:
        return None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Fakten (JSON):\n" + json.dumps(facts, ensure_ascii=False, indent=1)},
    ]
    for _ in range(2):  # ein zweiter Versuch, falls der erste Text die Pruefung nicht besteht
        try:
            text = client.chat(messages)
        except Exception:  # noqa: BLE001  Netz, Schluessel, Antwortformat: still auf die Vorlage zurueckfallen
            return None
        if text and len(text) >= 80 and is_plain_text(text) and verify_numbers(text, facts):
            return text
    return None


def explain(result, patient: str | None = None, client=None) -> tuple[str, str]:
    """(Text, Quelle). Quelle ist 'llm' (geprüfter KI-Text) oder 'template' (Vorlagentext)."""
    facts = build_facts(result, patient)
    text = llm_text(facts, client)
    if text:
        if DISCLAIMER not in text:
            text = text.rstrip() + "\n\n" + DISCLAIMER
        return text, "llm"
    return template_text(facts), "template"
