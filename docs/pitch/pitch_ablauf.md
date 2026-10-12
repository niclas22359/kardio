# Kardio – Pitch-Ablauf (5 Minuten)

Zwei Sprecher: eine Person aus der Medizin (Pain Point, Mehrwert) und eine aus dem Bau (Produkt, Nachweis).

## 0:00 – 0:30  Pain Point (Medizin)
„Rund 400 Menschen bekommen in Deutschland jedes Jahr ein neues Herz. Und dann beginnt das Warten:
Bis zu jeder Fünfte erlebt im ersten Jahr eine Abstoßung. Entdeckt wird sie heute mit einer Biopsie,
sieben bis zwölf Mal im ersten Jahr, jedes Mal ein Katheter im Herzen. Die Biopsie sieht nur den Moment.
Zwischen zwei Terminen sind wir blind.“

## 0:30 – 0:45  Die Idee in einem Satz
„Was wäre, wenn das Herz uns Tage vorher sagen könnte, dass etwas nicht stimmt? Mit 30 Sekunden EKG am Tag,
von der Apple Watch, zu Hause.“

## 0:45 – 2:05  Video (Version 2, 77 Sekunden, nur Musik)
Niemand spricht. Danach kurze Pause.

## 2:05 – 3:20  Live-Demo (Bau), nicht hochladen, Ergebnis ist vorbereitet
1. Reiter Übersicht: „Jeder Patient, sein aktuelles Risiko, sortiert. Rot heißt: heute anrufen.“
2. Reiter Patient, D035 oder den mit dem höchsten Risiko: „Das Risiko steigt Tage vor der Biopsie, die dann
   Abstoßung zeigt. Darunter der Befundtext, dessen Zahlen das System selbst prüft.“
3. Reiter Merkmale, 5 Sekunden: „Das ist die Auswertung, die ihr aus Prism kennt. Automatisch.“
4. Reiter Prüfen: „Trainiert an 40 Patienten, geprüft an 30, die es nie gesehen hat: 0,97.
   Und die Gegenprobe ohne Muster: 0,52. Es schlägt nicht einfach immer Alarm.“

## 3:20 – 3:50  Ehrlichkeit (Bau)
„Das sind erzeugte Demo-Daten mit eingebautem Muster. Sie beweisen, dass die Maschine funktioniert,
nicht, dass das EKG Abstoßungen vorhersagt. Dafür brauchen wir die echte Biopsietabelle. Das System ist
dafür gebaut: Datei rein, Modell lernt neu, Prüfung an unabhängigen Patienten.“

## 3:50 – 4:30  Mehrwert (Medizin)
- Patient: Warnung vor Symptomen, Therapie früher, weniger Schaden.
- Arzt: eine Ampel pro Patient statt Datenberge, Befundtext zum Gegenlesen.
- Klinik: Biopsien nach Risiko statt nach Kalender, weniger Eingriffe.
- Alltag: 30 Sekunden am Handgelenk, Daten fließen automatisch.

## 4:30 – 5:00  Nächste Schritte und Schluss
1. Echte Biopsiebefunde einspielen.  2. Validierung, Schwellen mit Kardiologen.  3. Pilot in einem Zentrum.
„Kardio. Das Herz hört man, bevor es laut wird.“

## Zahlen und Quellen (für Rückfragen)
- ~400 Herztransplantationen pro Jahr: DGTHG, „nur knapp 400 pro Jahr“; 15.000ste Transplantation 2024 (Ärzteblatt).
- Abstoßung im ersten Jahr: Studien nennen 13 % bis 19,8 % klinisch relevante Abstoßungen (≥2R), je nach
  Zentrum und Zeitraum; „bis zu jeder Fünfte“ ist die Obergrenze.
- 7 bis 12 Biopsien im ersten Jahr: Literatur zu Überwachungs-Protokollen, Mittel 7,1 ± 2,5.
- Modellzahlen 0,97 / 0,52: eigene Demo-Daten, Reiter Prüfen.

## Mögliche Fragen der Jury
- „Warum logistische Regression?“ Erklärbar, Gewichte lesbar, robust bei wenigen Patienten.
- „Wie viele Patienten braucht ihr?“ Mindestens 20 Biopsien mit und ohne Abstoßung, verteilt auf mehrere
  Patienten; sonst sagt das System ehrlich „zu wenig Daten“.
- „Was ist mit der Apple-Watch-Genauigkeit?“ Wir nutzen nicht den Absolutwert, sondern die Abweichung vom
  persönlichen Normal desselben Geräts. Systematische Gerätefehler kürzen sich heraus.
