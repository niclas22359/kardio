# Kardio – Übergabe

Stand: 12.10.2026

## Links

| Was | Adresse |
|---|---|
| Code (öffentlich, kein GitHub-Konto nötig) | https://github.com/niclas22359/kardio |
| Anwendung online (Render) | *wird nach dem Render-Setup eingetragen* – Kennwort separat |
| Lokal starten | ZIP von GitHub laden, entpacken, `start.bat` doppelklicken (Windows, Python 3.12) |

## Worum es geht (Zusammenfassung)

Rund 400 Menschen erhalten in Deutschland jedes Jahr ein neues Herz. Bis zu jeder Fünfte erlebt im
ersten Jahr eine Abstoßung. Erkannt wird sie heute nur per Biopsie, sieben- bis zwölfmal im ersten
Jahr, jedes Mal ein Katheter im Herzen. Die Biopsie sieht den Moment; zwischen zwei Terminen ist man blind.

**Die Idee:** Patienten messen täglich 30 Sekunden EKG mit der Apple Watch. Das Werkzeug prüft, ob sich
das EKG in den 14 Tagen vor einer Biopsie mit Abstoßung anders verhält als vor einer Biopsie ohne Abstoßung,
und berechnet daraus pro Patient ein tagesaktuelles Risiko.

**Was wir beim Hackathon gebaut haben:**

1. **Daten einlesen** – EKG-Tabelle (Apple-Watch-Export) und Biopsietabelle hochladen; Spaltennamen und
   Schreibweisen dürfen abweichen, nicht erkannte Spalten lassen sich per Hand zuordnen.
2. **Persönliche Normallinie** – jeder Patient wird mit sich selbst verglichen (Absolutwert, CV, Abweichung,
   wie in der bisherigen Prism-Auswertung). Gerätefehler kürzen sich so heraus.
3. **Statistik** – Kruskal-Wallis und paarweise Mann-Whitney-U über die drei Befundgruppen
   (keine Abstoßung / Zwischengruppe / Abstoßung), korrigiert für Mehrfachtests. Ersetzt die manuelle Prism-Arbeit.
4. **Risikomodell** – erklärbare logistische Regression, geprüft an Patienten, die das Modell nie gesehen hat.
   Bei zu wenig Daten gibt es bewusst keine Risikozahl.
5. **Dashboard** – Übersicht mit Ampel pro Patient, Patientenansicht mit Verlauf und Biopsien, Merkmalsvergleich,
   Modell-Einblick, und ein Reiter „Prüfen“ für einen zweiten, unabhängigen Datensatz.
6. **Befundtext** – ein Sprachmodell formuliert die Zahlen aus; jede Zahl wird gegen die Berechnung geprüft.

**Ehrlichkeitshinweis:** Bisher gibt es keine echte Biopsietabelle. Die Demo nutzt erzeugte Patienten mit
eingebautem Muster. Sie belegt, dass die Maschine ein Muster findet (Modellgüte 0,97 auf ungesehenen Patienten)
und ohne Muster nichts findet (0,52). Sie belegt **nicht**, dass das EKG Abstoßungen vorhersagt. Dafür braucht es
die echten Biopsiebefunde: Datei hochladen, Modell lernt neu, Prüfung an unabhängigen Patienten.

**Nächste Schritte:** 1. echte Biopsiebefunde einspielen, 2. Validierung und Schwellen mit Kardiologen festlegen,
3. Pilot in einem Zentrum.

## Screenshots

![Übersicht](pitch/shot_overview.png)

![Patient](pitch/shot_patient.png)

Weitere: `pitch/shot_features.png` (Merkmale), `pitch/shot_upload.png` (Daten laden).

## Render einrichten (einmalig, ca. 5 Minuten)

1. https://dashboard.render.com → **New** → **Blueprint** → GitHub-Repo `niclas22359/kardio` wählen.
2. Render liest `render.yaml`. Zwei Werte eintragen: `MGA_API_KEY` (aus der lokalen `.env`) und
   `APP_PASSWORD` (frei wählen, wird an Frau Moog weitergegeben).
3. **Apply**. Erster Build 3–5 Minuten. Die Webadresse steht **auf der Service-Seite** (Service `kardio-ekg`
   anklicken), nicht auf der Blueprint-Seite. Falls der Name vergeben ist, hängt Render vier Zufallszeichen an.
4. Adresse einmal selbst öffnen (Kaltstart ca. 1 Minute im Free-Plan), dann in die E-Mail eintragen.

## E-Mail-Entwurf an Frau Moog

**Betreff:** Kardio – EKG-Früherkennung von Abstoßungen: Code und Anwendung zum Ausprobieren

Liebe Frau Moog,

vielen Dank noch einmal für den Austausch beim Hackathon. Wie besprochen schicke ich Ihnen hiermit den Stand
unseres Prototyps „Kardio“, damit Sie und Ihr Team ihn in Ruhe anschauen und weitergeben können.

**Kurz zusammengefasst:** Das Werkzeug liest die EKG-Messungen der Apple Watch und eine Biopsietabelle ein,
bildet für jeden Patienten seine persönliche Normallinie und prüft, ob sich das EKG in den 14 Tagen vor einer
Biopsie mit Abstoßung anders verhält als vor einer Biopsie ohne Abstoßung. Daraus entsteht pro Patient ein
tagesaktuelles Risiko mit Ampel, dazu die statistische Auswertung über die drei Befundgruppen, wie Sie sie
bisher in Prism von Hand erstellen, nur automatisch.

Wichtig zur Einordnung: Die Demo läuft mit künstlich erzeugten Patientendaten mit eingebautem Muster. Sie zeigt,
dass die Auswertung funktioniert, nicht, dass das EKG Abstoßungen tatsächlich vorhersagt. Das lässt sich erst mit
der echten Biopsietabelle beantworten, und genau dafür ist das Werkzeug gebaut: Tabelle hochladen, Modell lernt
neu, Prüfung an Patienten, die es nicht gesehen hat.

**Anwendung online:** [RENDER-ADRESSE]
Kennwort: [KENNWORT] (bitte nicht weiterleiten; die Seite „schläft“ bei Nichtnutzung, der erste Aufruf dauert dann rund eine Minute)

**Code und Dokumentation (öffentlich, kein Konto nötig):** https://github.com/niclas22359/kardio
Dort finden Sie auch eine Beschreibung der Methode, der erwarteten Tabellenformate und der Grenzen des Prototyps.
Wer die Anwendung lokal auf dem eigenen Rechner laufen lassen möchte: grüner Button „Code“ → „Download ZIP“,
entpacken, `start.bat` starten.

Zwei Screenshots hänge ich an: die Übersicht aller Patienten mit Ampel und die Patientenansicht mit Verlauf
und Biopsien.

Als nächsten Schritt würden wir gern mit echten (pseudonymisierten) Biopsiebefunden testen. Wenn das für Sie
denkbar ist, melde ich mich gern für ein kurzes Gespräch, wie wir die Daten sicher austauschen können.

Herzliche Grüße
Niclas Krieg
Beyondles.ai · info@beyondles.ai

Anhänge: shot_overview.png, shot_patient.png
