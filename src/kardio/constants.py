"""Gemeinsame Bezeichner fuer alle Stationen."""

FEATURE_GROUPS = [
    "R_amplitude", "R_upslope", "P_amplitude", "T_amplitude",
    "R_duration", "P_duration", "T_duration",
    "RT_interval", "qRT_interval", "PR_interval", "R_rate",
]
STATS = ["mean", "sd", "min", "max"]
VALUE_COLUMNS = [f"{g}_{s}" for g in FEATURE_GROUPS for s in STATS]
MEAN_COLUMNS = [f"{g}_mean" for g in FEATURE_GROUPS]

# Drei Kennzahlen je Merkmal wie in der Prism-Vorlage der Klinik:
# mean = Absolutwert, cv = sd/|mean| (Unruhe), dev = relative Abweichung vom persoenlichen Normal
METRICS = ["mean", "cv", "dev"]
METRIC_LABELS = {"mean": "Absolutwert", "cv": "CV", "dev": "Abweichung"}
DEV_FEATURES = [f"{g}_{m}" for g in FEATURE_GROUPS for m in METRICS]

# Verdichtung je Fenster vor einer Biopsie
AGGS = ["zmean", "zmaxabs", "slope", "alertfrac"]
WINDOW_FEATURE_COLUMNS = [f"{f}__{a}" for f in DEV_FEATURES for a in AGGS]

COL_PATIENT = "patient"
COL_DATE = "date"
COL_INCLUDED = "included_no"
COL_CC = "cc"
COL_BIOPSY_NO = "biopsy_no"
COL_GRADE = "grade"          # 0 keine Abstossung, 1 Zwischengruppe, 2 Abstossung
COL_REJECTION = "rejection"  # grade == 2

GRADE_LABELS = {0: "keine Abstoßung", 1: "Zwischengruppe", 2: "Abstoßung"}
GRADE_COLORS = {0: "#2ecc40", 1: "#ffdc00", 2: "#ff4136"}

# Lesbare Namen fuer Texte und Diagramme
FEATURE_LABELS = {
    "R_amplitude": "R-Amplitude", "R_upslope": "R-Anstieg", "P_amplitude": "P-Amplitude",
    "T_amplitude": "T-Amplitude", "R_duration": "R-Dauer (QRS)", "P_duration": "P-Dauer",
    "T_duration": "T-Dauer", "RT_interval": "RT-Intervall", "qRT_interval": "qRT-Intervall",
    "PR_interval": "PR-Intervall", "R_rate": "Herzrate",
}
AGG_LABELS = {"zmean": "mittlere Abweichung", "zmaxabs": "größte Abweichung",
              "slope": "Trend", "alertfrac": "Anteil auffälliger Tage"}
