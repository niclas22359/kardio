# Kardio: EKG-basierte Früherkennung von Abstoßungen nach Herztransplantation

Stand: 2026-10-10 (Hackathon-Tag). Status: Entwurf, zur Freigabe durch Niclas.

## 1. Ziel

Ein lokal laufendes Dashboard, in das zwei Tabellen hochgeladen werden
(EKG-Messungen und Biopsiebefunde). Nach Klick auf "Analysieren" prüft das
System, ob sich in den EKG-Werten der Tage vor einer Biopsie mit Befund
"Abstoßung" ein Muster zeigt, das sich von Tagen vor "keine Abstoßung"
unterscheidet, und gibt pro Patient ein Risiko aus.

Nicht Ziel heute: Therapie-Erfolg, Datenübertragung von der Apple Watch,
Benutzerverwaltung, Cloud-Betrieb.

## 2. Ehrliche Rahmenbedingung

Es gibt heute nur 3 echte Patienten mit 70 Messungen und keine echten
Biopsiedaten. Deshalb beweist das System heute, dass die Maschine
funktioniert (sie findet ein bewusst eingebautes Muster in generierten
Daten und findet nichts, wenn keins drin ist). Ob das EKG in der Realität
Abstoßungen ankündigt, kann erst die echte Biopsietabelle zeigen. Das
Dashboard zeigt diese Einschränkung jederzeit sichtbar an.

## 3. Eingangsdaten

### 3.1 Tabelle 1: EKG-Messungen (vorhanden, anonymisiert)

Datei `data/raw/ecg_real.csv`. Trennzeichen Semikolon, Dezimalpunkt.
Spalten:

- `Name`: Patientenkürzel (MC, MW, JM239)
- `Date`: Dateiname der Form `ecg_YYYY-MM-DD[_n].csv`, Datum muss daraus
  extrahiert werden
- `included_no`: Anzahl verwerteter Herzschläge (0 bei leerer Messung)
- `cc`: Qualitätsmaß (0,87 bis 0,97), leer bei leerer Messung
- `hr`, `rr`, `rr_int`: Flags, immer 1, werden ignoriert
- 12 EKG-Merkmale, je als `_mean`, `_sd`, `_min`, `_max` (48 Zahlenspalten):
  R_amplitude, R_upslope, P_amplitude, T_amplitude, R_duration, P_duration,
  T_duration, RT_interval, qRT_interval, PR_interval, R_rate
  (zählt man die Gruppen, sind es 11 Merkmalsgruppen; die Tafel nennt
  zusätzlich QRS, das in R_duration steckt)

Bekannte Eigenheiten, die der Einleser abfangen muss:
- 4 komplett leere Messungen (nur Name, Datum, included_no)
- 2 Messungen ohne T-Welle und RT-Abstände
- Mehrere Messungen am selben Tag (MC: 42 am 28.08.2025)
- Große Lücken (JM239: 4 Termine über 5 Monate)

### 3.2 Tabelle 2: Biopsien (nicht vorhanden, wird generiert)

Erwartete Spalten laut Niclas, Namen können abweichen:
- Patientenkürzel
- Biopsie-Nummer (fortlaufend pro Patient)
- Biopsiedatum
- Befund: Abstoßung ja/nein

Spaltenabgleich ist tolerant: Groß/Kleinschreibung, Leerzeichen,
Deutsch/Englisch (z.B. `Patient`, `Kuerzel`, `Name`, `ID`; `Befund`,
`Abstossung`, `rejection`, `result`), Werte wie `ja/nein`, `yes/no`,
`1/0`, `positiv/negativ`. Unbekannte Spalten werden ignoriert. Wenn eine
Pflichtspalte nicht zugeordnet werden kann, zeigt das Dashboard eine
Auswahlbox, in der der Nutzer die Spalte manuell zuweist.

### 3.3 Bestehende Auswertung der Klinik (Prism-Vorlage)

`docs/referenz/prism_vorlage/` zeigt, wie heute ausgewertet wird (GraphPad
Prism). Daraus übernommen:

- **Drei Befundgruppen:** `0 = keine Abstoßung` (grün), `1 = Zwischengruppe`
  (gelb), `2 = Abstoßung` (rot). Die Biopsiespalte wird auf diese Stufe
  abgebildet (`grade`); zusätzlich `rejection = (grade == 2)` für das
  Modell. Erkannte Schreibweisen: `0/1/2`, `nein/zwischen/ja`,
  `keine/Zwischengruppe/Abstoßung`, `none/borderline/rejection`,
  ISHLT-Grade `0R/1R/2R/3R` (0R→0, 1R→1, 2R und 3R→2).
- **Drei Kennzahlen je Merkmal:** Absolutwert (`_mean`), CV
  (`_sd / |_mean|`, "Unruhe") und Abweichung (`|_mean - Normal| / |Normal|`
  in Prozent). Station 2 berechnet alle drei, Station 3 und 4 vergleichen
  alle drei.
- **Darstellung:** Boxplots je Merkmal mit drei farbigen Gruppen,
  Kruskal-Wallis über alle drei Gruppen plus paarweise Mann-Whitney-U
  mit p-Werten als Klammern über den Boxen. Die Merkmalsseite im Dashboard
  baut dieses Raster (4 Spalten x N Zeilen) nach.

## 4. Datengenerator (Demo-Daten)

Erzeugt Dateien im exakt gleichen Format wie die echten, damit alles
dahinter nicht weiß, ob Daten echt oder generiert sind.

- Erzeugt N Demo-Patienten (Standard 40) mit täglichen Messungen über
  12 Monate nach Transplantation, mit zufälligen fehlenden Tagen (ca. 10%)
  und gelegentlich mehreren Messungen pro Tag.
- Wertebereiche und Streuungen pro Merkmal werden aus den echten Daten von
  MW abgeleitet, pro Patient leicht verschoben (jeder hat sein eigenes
  Normal).
- Biopsieplan nach dem Tafelbild: wöchentlich in den ersten 4 Wochen, dann
  alle 2 Wochen bis Monat 3, dann monatlich bis Monat 6, dann alle 3 Monate.
  Nach einer Abstoßung: Behandlung und 3 engmaschige Kontrollbiopsien im
  Wochenabstand.
- Ca. 25% der Biopsien haben Befund "Abstoßung" (Stufe 2), wie von Niclas
  vorgegeben; zusätzlich ca. 15% Zwischengruppe (Stufe 1) mit halb so
  starkem Muster, Rest Stufe 0.
- Eingebautes Muster (per Schalter an/aus): In den 10 Tagen vor einer
  Abstoßungs-Biopsie driften bestimmte Merkmale schrittweise weg vom
  persönlichen Normal. Standard-Annahme (frei gewählt, keine medizinische
  Aussage): R_amplitude_mean sinkt um bis zu 15%, qRT_interval_mean steigt
  um bis zu 8%, R_rate_mean steigt um bis zu 10%, T_amplitude_mean sinkt um
  bis zu 20%. Nach Behandlungsbeginn kehren die Werte über 14 Tage zurück.
- Alle Annahmen stehen in einer Einstellungsdatei (`config/generator.yaml`)
  und werden im Dashboard als "Demo-Annahmen" angezeigt.
- Die 3 echten Patienten werden immer zusätzlich mitgeführt; für sie
  werden ebenfalls Biopsien generiert, damit die Strecke durchläuft.
- Fester Zufalls-Startwert, damit die Demo reproduzierbar ist.

## 5. Verarbeitungsstraße

Jede Station bekommt eine definierte Tabelle rein und gibt eine definierte
Tabelle raus. Alle Zwischenergebnisse werden als CSV in `data/processed/`
abgelegt, damit man sie anschauen kann.

### Station 1: Einlesen und Bereinigen

- EKG: Datum aus `Date` extrahieren, Zahlen parsen, Flags verwerfen.
- Tagesregel: pro Patient und Tag bleibt genau eine Messung. Auswahl:
  (1) die mit den wenigsten leeren Feldern, (2) bei Gleichstand die mit dem
  höchsten `included_no`, (3) bei erneutem Gleichstand die mit dem höchsten
  `cc`. Komplett leere Messungen zählen als "Tag ohne Daten".
- Tage ohne Messung bleiben als Lücke erhalten (werden nicht aufgefüllt).
- Biopsien: Spaltenabgleich nach 3.2, Datum normalisieren, Befund auf
  True/False abbilden, pro Patient nach Datum sortieren und Nummer prüfen.
- Ausgabe: `ecg_daily.csv` (Patient, Datum, 48 Werte, included_no, cc) und
  `biopsies.csv` (Patient, Nummer, Datum, Abstoßung).
- Qualitätsbericht: Anzahl Patienten, Tage, verworfene Doppelmessungen,
  leere Tage, nicht zuordenbare Patienten (Kürzel nur in einer Tabelle).

### Station 2: Persönliche Normallinie und Abweichungen

- Pro Patient und Merkmal: Normallinie = Median und robuste Streuung
  (MAD) über die ersten 14 Tage mit Daten, die nicht in einem
  Abstoßungsfenster liegen. Hat ein Patient weniger als 5 solcher Tage,
  wird die Normallinie aus der Gesamtpopulation genommen und das als
  "schwache Normallinie" markiert.
- Pro Patient, Tag, Merkmal (nur `_mean`-Spalten, 11 Stück, plus `_sd` als
  Unruhemaß): Abweichung in Einheiten der Streuung (robuster z-Wert),
  Veränderung zum letzten vorhandenen Tag, gleitender 7-Tage-Trend.
- Ausgabe: `deviations.csv` im Langformat (Patient, Datum, Merkmal,
  z_wert, delta_vortag, trend_7d).

### Station 3: Fenster um Biopsien

- Pro Biopsie: Fenster = die 14 Tage vor dem Biopsiedatum (einstellbar).
  Der Biopsietag selbst zählt mit.
- Je Merkmal wird aus dem Fenster verdichtet: mittlerer z-Wert, maximaler
  |z-Wert|, Steigung über das Fenster, Anteil Tage mit |z| > 2.
- Fenster mit weniger als 3 Datentagen werden markiert und aus der
  Statistik ausgeschlossen (bleiben aber im Dashboard sichtbar).
- Fenster, die sich mit einer vorherigen Abstoßung plus Behandlungsphase
  überlappen, werden markiert ("unter Behandlung"), damit der
  Rückgang der Werte nicht als Vorbote gelesen wird.
- Ausgabe: `windows.csv` (eine Zeile pro Biopsie, Etikett, 11 Merkmale x
  4 Kennzahlen, Datentage, Markierungen).

### Station 4: Statistischer Vergleich

- Je Merkmal und Kennzahl: Kruskal-Wallis über die drei Gruppen, dazu
  paarweise Mann-Whitney-U (0 vs 2, 0 vs 1, 1 vs 2) wie in der
  Prism-Vorlage. Hauptkriterium für die Rangliste ist der Vergleich
  0 vs 2. Effektstärke als Cliff's Delta.
- Korrektur für Mehrfachtests (Benjamini-Hochberg), damit bei 44 Tests nicht
  zufällig etwas "signifikant" wird.
- Je Merkmal zusätzlich: Trennschärfe als AUC, wenn man nur dieses eine
  Merkmal als Alarm nähme.
- Ausgabe: `feature_ranking.csv` (Merkmal, Kennzahl, p-Wert roh,
  p-Wert korrigiert, Effektstärke, AUC, Anzahl ja/nein).
- Bei weniger als 5 Abstoßungs-Fenstern: Rangliste wird berechnet, aber
  als "nicht belastbar" markiert.

### Station 5: Lernendes Modell

- Eingabe: `windows.csv`. Modell: logistische Regression mit
  Regularisierung (erklärbar, Gewichte pro Merkmal lesbar). Zweites Modell
  zum Vergleich: Gradient Boosting (kleine Bäume), nur als Referenz.
- Validierung: Gruppierte Kreuzvalidierung nach Patient (ein Patient ist
  nie gleichzeitig im Lern- und Testteil). Kennzahlen: AUC,
  Sensitivität und Spezifität bei einem wählbaren Schwellwert, Kalibrierung.
- Ausgabe: `model_report.json` (Kennzahlen, Gewichte, Konfidenzintervalle)
  und `risk_timeline.csv` (Patient, Datum, Risiko 0 bis 1), wobei das
  Risiko für jeden Tag aus einem rollierenden 14-Tage-Fenster berechnet
  wird, nicht nur an Biopsietagen.
- Belastbarkeitsregel: Bei weniger als 20 Fenstern insgesamt oder weniger
  als 8 Abstoßungs-Fenstern gibt das Modell keine Risikozahlen aus, sondern
  den Status "zu wenig Daten". Die Statistik aus Station 4 läuft trotzdem.
- Das Modell wird nicht gespeichert und bei jeder Analyse neu gelernt
  (Datenmengen sind klein, Laufzeit Sekunden).

### Station 6: Erklärtext (LLM)

- Eingabe: Qualitätsbericht, Rangliste, Modellbericht, Risiko des
  gewählten Patienten. Ausgabe: deutscher Fließtext für den Arzt, 150 bis
  250 Wörter, mit fester Struktur: Datenlage, auffällige Merkmale, Risiko,
  Einschränkungen.
- Das LLM bekommt nur verdichtete Zahlen, nie Rohdaten, und darf keine
  Merkmale nennen, die nicht in der Rangliste stehen. Jeder genannte
  Zahlenwert wird nach der Antwort gegen die Eingabe geprüft; bei
  Abweichung wird der Text verworfen und ein Vorlagentext ohne LLM
  ausgegeben.
- Schnittstelle: myGenAssist (MGA), OpenAI-kompatibel, Basisadresse
  `https://chat.int.bayer.com/api/v2`, Modell `claude-sonnet-5` (Details in
  `docs/referenz/MGA-API.md`). Schlüssel aus Umgebungsvariable `MGA_API_KEY`,
  Basisadresse und Modell über `MGA_BASE_URL` / `MGA_MODEL` änderbar. Das
  `model`-Feld der Antwort wird geprüft (stiller Fallback bei falscher ID).
  Ohne Schlüssel oder bei Fehler läuft das System mit dem Vorlagentext.

### Station 7: Dashboard

Streamlit, läuft lokal unter `http://localhost:8501` und online auf Render
(siehe Abschnitt 6). Seiten:

1. **Upload**: zwei Dateifelder (EKG, Biopsien), Knopf "Demo-Daten laden"
   (mit Schalter "Muster eingebaut ja/nein"), Spaltenzuordnung falls nötig,
   Qualitätsbericht nach dem Laden. Großer roter Knopf "Analysieren".
2. **Übersicht**: Ampel pro Patient (grün/gelb/rot nach Risiko, grau bei
   "zu wenig Daten"), Datenbasis-Banner ("DEMO-DATEN mit eingebautem
   Muster" / "DEMO-DATEN ohne Muster" / "Echte Daten, n Patienten").
3. **Merkmale**: Rangliste aus Station 4 als Tabelle und Balkendiagramm,
   pro Merkmal Boxplot ja/nein.
4. **Patient**: Verlaufskurven der Merkmale mit Normalband, Biopsien als
   Markierungen (rot = Abstoßung), Risikokurve, Erklärtext.
5. **Modell**: Kennzahlen, Gewichte, Hinweis zur Belastbarkeit.

Zustände: Ladeanzeige während der Analyse, verständliche Fehlermeldung bei
unlesbarer Datei (welche Spalte fehlt), leere Zustände vor dem ersten
Upload. Mobil nicht Ziel, aber Seiten dürfen nicht brechen.

## 6. Technik

- Python 3.13, pandas, numpy, scipy, scikit-learn, streamlit, plotly,
  pyyaml, anthropic. Abhängigkeiten in `requirements.txt`, Umgebung über
  `venv`.
- Struktur:
  - `src/kardio/ingest/` (Station 1), `src/kardio/features/` (2, 3),
    `src/kardio/stats/` (4), `src/kardio/model/` (5),
    `src/kardio/explain/` (6), `src/kardio/generator/`, `app/` (7)
  - `tests/` mit pytest, `config/` für Einstellungen, `data/raw`,
    `data/processed`, `data/demo`
- Jede Station ist eine Funktion "Tabelle rein, Tabelle raus" und hat
  eigene Tests mit kleinen handgebauten Beispielen.
- Ein Kommandozeilen-Lauf `python -m kardio.run --ecg X --biopsies Y`
  führt alle Stationen ohne Dashboard aus (für Tests und Hackathon-Notfall).
- Online-Betrieb: Render (Weg B aus `docs/referenz/VEROEFFENTLICHEN.md`),
  `render.yaml` mit Python-Runtime, Startbefehl `streamlit run app/main.py`,
  Umgebungsvariablen `MGA_API_KEY` und `APP_PASSWORD` mit `sync: false`
  (werden im Render-Dashboard eingetragen, nie im Repo). Passwortschutz: wenn
  `APP_PASSWORD` gesetzt ist, zeigt das Dashboard zuerst eine Kennwortabfrage;
  lokal ohne Variable keine Abfrage.
- Repo: privat unter dem GitHub-Konto `niclas22359`, Name `kardio`.
  Vor jedem Push Schlüssel-Kontrolle (kein `mga-` im Repo).

## 7. Nachweis, dass es funktioniert (Tests)

1. Einleser: echte Datei lädt ohne Fehler; 4 leere Messungen werden als
   leere Tage erkannt; MC bleibt mit genau 1 Tag; Spaltenvarianten der
   Biopsietabelle werden korrekt zugeordnet.
2. Tagesregel: handgebautes Beispiel mit 3 Messungen am Tag, erwartete
   Gewinnerzeile.
3. Lücken: Patient mit 2 Messungen in 5 Monaten läuft durch alle Stationen
   ohne Absturz und endet mit Status "zu wenig Daten".
4. Muster-Wiederfindung: Generator mit Muster an, 40 Patienten: die im
   Generator veränderten Merkmale stehen in der Rangliste oben mit
   korrigiertem p < 0,05, Modell-AUC > 0,8.
5. Null-Test: Generator mit Muster aus: kein Merkmal mit korrigiertem
   p < 0,05, Modell-AUC zwischen 0,4 und 0,6.
6. LLM-Prüfung: Antwort mit erfundener Zahl wird verworfen, Vorlagentext
   erscheint.

## 8. Entscheidungen, die Niclas noch ändern kann

Alle als Standardwerte gesetzt, in `config/analysis.yaml` änderbar:
Fensterlänge 14 Tage, Normallinie aus 14 Tagen, Belastbarkeitsgrenzen
(20 Fenster, 8 Abstoßungen), Ampelschwellen (gelb ab 30%, rot ab 60%),
Demo-Patientenzahl 40, Abstoßungsanteil 25%, eingebautes Muster laut
Abschnitt 4.

## 9. Was nach heute kommt (nicht Teil dieser Spezifikation)

Echte Biopsietabelle einspielen, Schwellwerte mit Ärzten abstimmen,
Betrieb auf dem Playground-Server, Benutzeranmeldung, Anbindung an die
automatische Apple-Watch-Übertragung.
