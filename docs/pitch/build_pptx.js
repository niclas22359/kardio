// Kardio Pitch-Deck (erster Entwurf). Dunkles, medizinisch-futuristisches Layout mit EKG-Motiv.
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const SKILL = "C:/Users/Niclas Krieg/.claude/skills/synced/6927eafb-420d-437c-91e3-ea2afe1ae39b_0ebbd26e-810f-4c56-9147-760f7231dd6f/pptx";
const { applyTheme } = require(SKILL + "/scripts/apply_theme.js");
const OUT = process.argv[2] || "Kardio_Pitch_Entwurf.pptx";
const SHOT = "C:/Users/Niclas Krieg/Claude Code_Projects/kardio/docs/pitch/shot_features.png";

const HEX = { bg: "070B14", bg2: "0F1828", card: "111B2E", text: "E8F1FF", muted: "9FB2C8",
              cyan: "22D3EE", red: "FF4D6D", green: "2DD4A7", amber: "FFB020", line: "24344F" };
const THEME = {
  name: "Kardio Pulse", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: HEX.text, lt1: HEX.bg, dk2: HEX.muted, lt2: HEX.bg2,
            accent1: HEX.cyan, accent2: HEX.red, accent3: HEX.green, accent4: HEX.amber,
            accent5: HEX.line, accent6: HEX.card, hlink: HEX.cyan, folHlink: HEX.muted },
};
const W = 13.333, H = 7.5, PX = 200;                     // Folie in Zoll, Grafiken mit 200 px je Zoll
const png = async (svg) => "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");

// ---------- Grafiken (ohne Text; Beschriftungen sind echte, editierbare Textfelder) ----------
function backgroundSvg() {
  const w = W * 120, h = H * 120;
  let grid = "";
  for (let x = 0; x <= w; x += 40) grid += `<line x1="${x}" y1="0" x2="${x}" y2="${h}" stroke="#1B2A44" stroke-width="${x % 200 === 0 ? 1.2 : 0.5}" opacity="0.55"/>`;
  for (let y = 0; y <= h; y += 40) grid += `<line x1="0" y1="${y}" x2="${w}" y2="${y}" stroke="#1B2A44" stroke-width="${y % 200 === 0 ? 1.2 : 0.5}" opacity="0.55"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">
    <defs><radialGradient id="g" cx="78%" cy="18%" r="85%">
      <stop offset="0" stop-color="#12233F"/><stop offset="0.55" stop-color="#0A1222"/><stop offset="1" stop-color="#070B14"/>
    </radialGradient>
    <radialGradient id="v" cx="50%" cy="50%" r="75%"><stop offset="0.6" stop-color="#070B14" stop-opacity="0"/><stop offset="1" stop-color="#070B14" stop-opacity="0.9"/></radialGradient></defs>
    <rect width="100%" height="100%" fill="url(#g)"/>${grid}<rect width="100%" height="100%" fill="url(#v)"/></svg>`;
}
const BEAT = [[0.22, 0.10, 0.030], [0.37, -0.07, 0.010], [0.40, 1.0, 0.012], [0.43, -0.16, 0.012], [0.66, 0.22, 0.050]];
const beatY = (t) => BEAT.reduce((s, [c, a, w]) => s + a * Math.exp(-((t - c) ** 2) / (2 * w * w)), 0);
function ecgPath(x0, beatW, beats, base, amp, shrink = () => 1) {
  let d = "";
  const n = Math.round(beatW * beats / 2);
  for (let i = 0; i <= n; i++) {
    const u = (i / n) * beats, b = Math.floor(Math.min(u, beats - 1e-9));
    const y = base - amp * shrink(b) * beatY(u - b);
    d += (i ? "L" : "M") + (x0 + u * beatW).toFixed(1) + " " + y.toFixed(1);
  }
  return d;
}
function ecgSvg(wIn, hIn, { beats = 3, x0 = 120, base, amp, color = "#22D3EE", fade = false, shrink } = {}) {
  const w = wIn * PX, h = hIn * PX, beatW = (w - 2 * x0) / beats;
  const d = ecgPath(x0, beatW, beats, base, amp, shrink);
  return { beatW, x0, svg: `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">
    <defs><filter id="glow" x="-10%" y="-40%" width="120%" height="180%"><feGaussianBlur stdDeviation="9"/></filter>
    <linearGradient id="f" x1="0" x2="1"><stop offset="0" stop-color="${color}" stop-opacity="${fade ? 0 : 1}"/><stop offset="0.25" stop-color="${color}"/><stop offset="1" stop-color="${color}"/></linearGradient></defs>
    <line x1="0" y1="${base}" x2="${w}" y2="${base}" stroke="#24344F" stroke-width="2"/>
    <path d="${d}" fill="none" stroke="${color}" stroke-width="16" opacity="0.45" filter="url(#glow)"/>
    <path d="${d}" fill="none" stroke="url(#f)" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/></svg>` };
}
// Zeitleiste: unsichtbarer Angriff auf das neue Herz, Biopsien als Punkte
const TL = { wIn: 11.93, hIn: 3.7, x0: 110, perMonth: 262, base: 520, top: 150, dotY: 610 };
const tlX = (m) => TL.x0 + m * TL.perMonth;
function attack(m) {                                      // 0..1
  if (m < 3.45) return 0;
  if (m < 4.0) { const u = (m - 3.45) / 0.55; return 0.82 * u * u * (3 - 2 * u); }
  if (m < 4.5) return 0.82 + 0.18 * (m - 4.0) / 0.5;
  return Math.exp(-(m - 4.5) / 0.55);
}
function timelineSvg() {
  const w = TL.wIn * PX, h = TL.hIn * PX;
  let d = "", area = `M${tlX(0)} ${TL.base}`;
  for (let m = 0; m <= 8.2; m += 0.02) {
    const p = `${tlX(m).toFixed(1)} ${(TL.base - (TL.base - TL.top) * attack(m)).toFixed(1)}`;
    d += (m ? "L" : "M") + p; area += "L" + p;
  }
  area += `L${tlX(8.2)} ${TL.base}Z`;
  let grid = "", dots = "";
  for (const m of [0, 2, 4, 6, 8]) grid += `<line x1="${tlX(m)}" y1="40" x2="${tlX(m)}" y2="${h - 40}" stroke="#24344F" stroke-width="2"/>`;
  for (const [m, rej] of [[0.5, 0], [1, 0], [2, 0], [3, 0], [4.5, 1], [5.25, 0]])
    dots += `<circle cx="${tlX(m)}" cy="${TL.dotY}" r="${rej ? 26 : 22}" fill="${rej ? "#FF4D6D" : "#22D3EE"}" opacity="0.25"/><circle cx="${tlX(m)}" cy="${TL.dotY}" r="13" fill="${rej ? "#FF4D6D" : "#22D3EE"}"/>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">
    <defs><linearGradient id="a" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FF4D6D" stop-opacity="0.55"/><stop offset="1" stop-color="#FF4D6D" stop-opacity="0.03"/></linearGradient>
    <filter id="glow" x="-10%" y="-40%" width="120%" height="180%"><feGaussianBlur stdDeviation="10"/></filter></defs>
    ${grid}<path d="${area}" fill="url(#a)"/>
    <path d="${d}" fill="none" stroke="#FF4D6D" stroke-width="18" opacity="0.4" filter="url(#glow)"/>
    <path d="${d}" fill="none" stroke="#FF4D6D" stroke-width="6" stroke-linejoin="round"/>
    <line x1="${tlX(0)}" y1="${h - 40}" x2="${tlX(8.2)}" y2="${h - 40}" stroke="#3A4E70" stroke-width="3"/>${dots}</svg>`;
}
function playSvg() {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400"><circle cx="200" cy="200" r="170" fill="#22D3EE" opacity="0.12"/>
    <circle cx="200" cy="200" r="130" fill="none" stroke="#22D3EE" stroke-width="8"/><path d="M165 130 L165 270 L275 200 Z" fill="#22D3EE"/></svg>`;
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Kardio – Pitch";
  pres.author = "Beyondles.ai";
  const C = pres.SchemeColor;
  const bg = await png(backgroundSvg());

  const footer = [
    { text: { text: "KARDIO  ·  EKG-Früherkennung nach Herztransplantation", options: { x: 0.7, y: 6.95, w: 8, h: 0.3, fontSize: 10, color: C.text2, charSpacing: 1.5, margin: 0 } } },
  ];
  pres.defineSlideMaster({
    title: "KARDIO_TITLE", background: { data: bg },
    objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.25, w: 11.9, h: 2.1, fontSize: 46, bold: true, color: C.text1, align: "left", valign: "top", margin: 0 }, text: "Titel" } },
              { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.45, w: 9.2, h: 1.0, fontSize: 20, color: C.text2, align: "left", valign: "top", margin: 0 }, text: "Untertitel" } }],
  });
  pres.defineSlideMaster({
    title: "KARDIO_CONTENT", background: { data: bg }, slideNumber: { x: 12.1, y: 6.93, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
    objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.45, w: 11.93, h: 0.85, fontSize: 34, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "Titel" } }, ...footer],
  });
  const T = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, color: C.text1, fontSize: 16, valign: "top", ...o });
  const card = (s, x, y, w, h, name) => s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color: C.accent6, transparency: 12 }, line: { color: C.accent5, width: 1 }, objectName: name });
  const badge = (s, x, y, n, color) => {
    s.addShape(pres.ShapeType.ellipse, { x, y, w: 0.62, h: 0.62, fill: { color, transparency: 80 }, line: { color, width: 1.5 }, objectName: "Nummer " + n });
    T(s, String(n), { x, y, w: 0.62, h: 0.62, align: "center", valign: "middle", bold: true, fontSize: 20, color });
  };

  // ---------- 1 Titel ----------
  pres.addSection({ title: "Pitch" });
  let s = pres.addSlide({ masterName: "KARDIO_TITLE", sectionTitle: "Pitch" });
  T(s, "KARDIO  ·  HACKATHON 2026", { x: 0.7, y: 0.7, w: 8, h: 0.35, fontSize: 13, color: C.accent1, bold: true, charSpacing: 4 });
  s.addText("Erkennt eine Smartwatch, wenn ein transplantiertes Herz in Gefahr ist?", { placeholder: "title" });
  s.addText("Abstoßung früher sehen: mit dem EKG, das die Patienten ohnehin jeden Tag am Handgelenk aufzeichnen.", { placeholder: "body" });
  const e1 = ecgSvg(W, 2.6, { beats: 4, x0: 0, base: 360, amp: 300, fade: true });
  s.addImage({ data: await png(e1.svg), x: 0, y: 4.55, w: W, h: 2.6, objectName: "EKG-Linie" });
  s.addNotes("Einstieg mit der Frage. Ein Satz zum Team und zum Hackathon, dann direkt zum Problem.");

  // ---------- 2 Problem ----------
  s = pres.addSlide({ masterName: "KARDIO_CONTENT", sectionTitle: "Pitch" });
  s.addText("Das Problem: Eine Abstoßung kommt leise", { placeholder: "title" });
  T(s, "Die rote Kurve zeigt, wie stark der Körper das neue Herz angreift. Niemand sieht sie direkt. Gefunden wird sie nur, wenn gerade eine Gewebeprobe entnommen wird.", { x: 0.7, y: 1.35, w: 9.6, h: 0.75, fontSize: 16, color: C.text2 });
  const IX = 0.7, IY = 2.25;
  s.addImage({ data: await png(timelineSvg()), x: IX, y: IY, w: TL.wIn, h: TL.hIn, objectName: "Zeitleiste Abstoßung" });
  const ix = (m) => IX + tlX(m) / PX, iy = (px) => IY + px / PX;
  [["Transplantation", 0], ["2 Monate", 2], ["4 Monate", 4], ["6 Monate", 6], ["8 Monate", 8]].forEach(([t, m]) =>
    T(s, t, { x: ix(m) - 0.9, y: iy(TL.hIn * PX - 30), w: 1.8, h: 0.3, fontSize: 12, color: C.text2, align: "center" }));
  T(s, "Jeder Punkt ist eine Biopsie. Sie werden mit der Zeit seltener.", { x: ix(0.35), y: iy(TL.dotY + 30) - 0.02, w: 5.2, h: 0.3, fontSize: 12, color: C.text2 });
  s.addShape(pres.ShapeType.roundRect, { x: ix(2.75) - 1.75, y: iy(55), w: 3.5, h: 0.5, rectRadius: 0.25, fill: { color: C.accent4 }, line: { color: C.accent4, width: 0 }, objectName: "Kernfrage" });
  T(s, "Sieht die Uhr das kommen?", { x: ix(2.75) - 1.75, y: iy(55), w: 3.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
  T(s, "Das Herz gerät in Not:\nstill und ohne Symptome", { x: ix(2.75) - 1.75, y: iy(190), w: 3.5, h: 0.6, fontSize: 13, color: C.accent4, align: "center" });
  T(s, "Eine Biopsie findet es", { x: ix(4.5) + 0.25, y: iy(TL.top - 10), w: 2.6, h: 0.3, fontSize: 13, bold: true, color: C.accent2 });
  T(s, "Behandlung drängt es zurück", { x: ix(5.0) + 0.1, y: iy(330), w: 3.2, h: 0.3, fontSize: 13, color: C.accent3 });
  s.addNotes("Painpoint: Biopsien sind invasiv und finden nur in Abständen statt. Dazwischen ist man blind. Genau in diese Lücke zielt das Projekt.");

  // ---------- 3 Herzschlag ----------
  s = pres.addSlide({ masterName: "KARDIO_CONTENT", sectionTitle: "Pitch" });
  s.addText("Ein Herzschlag, wie die Uhr ihn aufzeichnet", { placeholder: "title" });
  T(s, "Jeder Schlag zeichnet dieselbe Form aus fünf Teilen. Unsere Annahme: Eine Abstoßung verändert diese Form um wenige Prozent, lange bevor der Patient etwas spürt.", { x: 0.7, y: 1.35, w: 9.8, h: 0.75, fontSize: 16, color: C.text2 });
  const EX = 0.7, EY = 2.15, EW = 11.93, EH = 2.95, base = 400, amp = 300;
  const e3 = ecgSvg(EW, EH, { beats: 3, x0: 120, base, amp, shrink: (b) => [1, 0.93, 0.86][b] });
  s.addImage({ data: await png(e3.svg), x: EX, y: EY, w: EW, h: EH, objectName: "Drei Herzschläge" });
  const bx = (t) => EX + (e3.x0 + t * e3.beatW) / PX, by = (v) => EY + (base - amp * v) / PX;
  const lab = (ch, t, v, dy, note, noteW = 2.2) => {
    T(s, ch, { x: bx(t) - 0.3, y: by(v) + dy, w: 0.6, h: 0.4, fontSize: 22, bold: true, color: C.accent1, align: "center" });
    if (note) T(s, note, { x: bx(t) - noteW / 2, y: by(v) + dy - 0.52, w: noteW, h: 0.5, fontSize: 11, color: C.text2, align: "center", valign: "bottom" });
  };
  lab("P", 0.22, 0.10, -0.5, "Vorhöfe füllen\ndie Kammern");
  lab("R", 0.40, 1.0, -0.45, null);
  T(s, "Der große Schlag:\nBlut verlässt das Herz", { x: bx(0.40) + 0.3, y: by(1.0) - 0.3, w: 2.2, h: 0.5, fontSize: 11, color: C.text2 });
  lab("T", 0.66, 0.22, -0.5, "Das Herz\nerholt sich");
  lab("Q", 0.345, -0.07, 0.05, null);
  lab("S", 0.455, -0.16, 0.08, null);
  T(s, "normal", { x: bx(0.4) - 0.8, y: EY + EH - 0.1, w: 1.6, h: 0.28, fontSize: 11, color: C.text2, align: "center" });
  T(s, "R-Zacke wird niedriger …", { x: bx(1.4) - 1.3, y: EY + EH - 0.1, w: 2.6, h: 0.28, fontSize: 11, color: C.accent4, align: "center" });
  T(s, "… wenige Prozent, kaum sichtbar", { x: bx(2.4) - 1.5, y: EY + EH - 0.1, w: 3.0, h: 0.28, fontSize: 11, color: C.accent4, align: "center" });
  const feats = [["Höhen", "Amplituden von R, P und T, Anstieg von R"], ["Dauern", "Wie lange P, R und T jeweils brauchen"], ["Abstände", "PR-, RT- und qRT-Intervall, Herzrate"]];
  feats.forEach(([h, t], i) => {
    const x = 0.7 + i * 4.06;
    card(s, x, 5.45, 3.81, 1.25, "Merkmal " + h);
    T(s, h, { x: x + 0.25, y: 5.58, w: 3.3, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
    T(s, t, { x: x + 0.25, y: 6.0, w: 3.3, h: 0.55, fontSize: 14, color: C.text1 });
  });
  s.addNotes("11 Merkmale je Messung, jeweils Mittelwert und Streuung über die Herzschläge. Die Verkleinerung in der Grafik ist stark übertrieben dargestellt.");

  // ---------- 4 Lösung ----------
  s = pres.addSlide({ masterName: "KARDIO_CONTENT", sectionTitle: "Pitch" });
  s.addText("Unsere Lösung: Jeder Patient ist sein eigener Maßstab", { placeholder: "title" });
  const steps = [["Messen", "Die Apple Watch zeichnet täglich ein kurzes EKG auf. Kein Klinikbesuch nötig.", C.accent1],
                 ["Normal lernen", "Für jeden Patienten entsteht sein persönliches Normal je Merkmal.", C.accent1],
                 ["Abweichung sehen", "Die Tage vor jeder Biopsie werden verdichtet und mit dem Befund verglichen.", C.accent4],
                 ["Warnen", "Ein erklärbares Modell macht daraus eine Ampel je Patient.", C.accent2]];
  steps.forEach(([h, t, col], i) => {
    const x = 0.7 + i * 3.04;
    card(s, x, 1.75, 2.8, 3.05, "Schritt " + (i + 1));
    badge(s, x + 0.25, 1.98, i + 1, col);
    T(s, h, { x: x + 0.25, y: 2.8, w: 2.3, h: 0.45, fontSize: 20, bold: true });
    T(s, t, { x: x + 0.25, y: 3.3, w: 2.3, h: 1.35, fontSize: 14, color: C.text2 });
    if (i < 3) T(s, "›", { x: x + 2.8, y: 2.95, w: 0.24, h: 0.6, fontSize: 28, color: C.accent1, align: "center", valign: "middle" });
  });
  card(s, 0.7, 5.1, 11.93, 1.5, "Was dahinter steckt");
  const facts = [["132", "Kennzahlen je Biopsie"], ["3", "statistische Prüfungen, auch je Patient"], ["296", "automatische Tests der Anwendung"]];
  facts.forEach(([n, t], i) => {
    const x = 1.0 + i * 3.95;
    T(s, n, { x, y: 5.22, w: 1.45, h: 1.25, fontSize: 48, bold: true, color: C.accent1, valign: "middle", align: "right" });
    T(s, t, { x: x + 1.7, y: 5.22, w: 2.1, h: 1.25, fontSize: 14, color: C.text1, valign: "middle" });
  });
  s.addNotes("Kernidee: nicht Patienten untereinander vergleichen, sondern jeden mit sich selbst. Das Modell ist eine einfache logistische Regression, bewusst erklärbar.");

  // ---------- 5 Video ----------
  s = pres.addSlide({ masterName: "KARDIO_CONTENT", sectionTitle: "Pitch" });
  s.addText("Die Anwendung in 90 Sekunden", { placeholder: "title" });
  const VW = 9.6, VH = 5.4, VX = (W - VW) / 2, VY = 1.4;
  s.addShape(pres.ShapeType.roundRect, { x: VX, y: VY, w: VW, h: VH, rectRadius: 0.15, fill: { color: C.background2, transparency: 10 }, line: { color: C.accent1, width: 1.5, dashType: "dash" }, objectName: "Videorahmen (Platzhalter)" });
  s.addImage({ data: await png(playSvg()), x: W / 2 - 0.9, y: VY + 1.35, w: 1.8, h: 1.8, objectName: "Play-Symbol" });
  T(s, "Video hier einfügen", { x: VX, y: VY + 3.4, w: VW, h: 0.5, fontSize: 22, bold: true, align: "center" });
  T(s, "Einfügen › Video › Dieses Gerät. Rahmen und Symbol danach löschen.", { x: VX, y: VY + 3.95, w: VW, h: 0.4, fontSize: 13, color: C.text2, align: "center" });
  s.addNotes("Platzhalter für das Erklärvideo. Format 16:9, der Rahmen hat die passende Größe.");

  // ---------- 6 Live-Demo ----------
  s = pres.addSlide({ masterName: "KARDIO_CONTENT", sectionTitle: "Pitch" });
  s.addText("Jetzt live: Zwei Tabellen, ein Knopf", { placeholder: "title" });
  const live = [["Hochladen", "EKG-Tabelle und Biopsie-Tabelle. Abweichende Spaltennamen erkennt die Anwendung selbst."],
                ["Analysieren", "Der rote Knopf startet Bereinigung, Statistik und Modell. Dauer: rund zehn Sekunden."],
                ["Prüfen", "Ein zweiter Datensatz mit anderen Patienten zeigt, ob das Gelernte auch dort trägt."]];
  live.forEach(([h, t], i) => {
    const y = 1.65 + i * 1.6;
    badge(s, 0.7, y, i + 1, i === 1 ? C.accent2 : C.accent1);
    T(s, h, { x: 1.55, y: y - 0.02, w: 3.9, h: 0.4, fontSize: 20, bold: true });
    T(s, t, { x: 1.55, y: y + 0.42, w: 3.9, h: 0.95, fontSize: 14, color: C.text2 });
  });
  const SW = 6.75, SH = SW * 1250 / 2000;
  s.addShape(pres.ShapeType.roundRect, { x: 5.8, y: 1.55, w: SW + 0.16, h: SH + 0.16, rectRadius: 0.08, fill: { color: C.accent5 }, line: { color: C.accent1, width: 1 }, shadow: { type: "outer", color: HEX.cyan, opacity: 0.25, blur: 18, offset: 0, angle: 90 }, objectName: "Rahmen Bildschirmfoto" });
  s.addImage({ path: SHOT, x: 5.88, y: 1.63, w: SW, h: SH, objectName: "Bildschirmfoto Merkmalsseite" });
  T(s, "Merkmalsseite: derselbe Gruppenvergleich wie bisher in der Klinik, nur automatisch.", { x: 5.8, y: 1.55 + SH + 0.3, w: SW + 0.16, h: 0.35, fontSize: 12, color: C.text2 });
  s.addNotes("Übergang in die Live-Demo. Reihenfolge: Demo-Datensatz A hochladen, analysieren, Merkmale zeigen, Patient zeigen, dann Datensatz B im Reiter Prüfen.");

  // ---------- 7 Stand und nächste Schritte ----------
  s = pres.addSlide({ masterName: "KARDIO_CONTENT", sectionTitle: "Pitch" });
  s.addText("Stand: Die Maschine läuft, jetzt fehlen echte Daten", { placeholder: "title" });
  const stats = [["0,92", "Trennschärfe im Training", "geprüft an zurückgehaltenen Patienten", C.accent1],
                 ["0,97", "an einem zweiten Datensatz", "30 Patienten, die das Modell nie gesehen hat", C.accent3],
                 ["0,52", "in der Gegenprobe ohne Muster", "so gut wie ein Münzwurf, genau wie es sein soll", C.accent4]];
  stats.forEach(([n, h, t, col], i) => {
    const x = 0.7 + i * 4.06;
    card(s, x, 1.6, 3.81, 2.55, "Kennzahl " + n);
    T(s, n, { x: x + 0.3, y: 1.75, w: 3.2, h: 1.0, fontSize: 60, bold: true, color: col, valign: "middle" });
    T(s, h, { x: x + 0.3, y: 2.82, w: 3.2, h: 0.4, fontSize: 16, bold: true });
    T(s, t, { x: x + 0.3, y: 3.24, w: 3.2, h: 0.7, fontSize: 13, color: C.text2 });
  });
  T(s, "Alle drei Zahlen stammen aus erzeugten Demo-Daten mit eingebautem Muster. Sie zeigen, dass die Auswertung ein vorhandenes Muster findet und ohne Muster nichts erfindet. Eine medizinische Aussage sind sie nicht.", { x: 0.7, y: 4.4, w: 11.93, h: 0.75, fontSize: 14, italic: true, color: C.text2 });
  T(s, "Nächste Schritte", { x: 0.7, y: 5.3, w: 4, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
  T(s, [{ text: "Echte Biopsietabelle einspielen und mit den EKG-Daten verknüpfen", options: { bullet: true, breakLine: true } },
        { text: "An einem zweiten, unabhängigen Patientenkollektiv prüfen", options: { bullet: true, breakLine: true } },
        { text: "Zeitraum vor der Biopsie und Alarmschwellen mit den Ärzten festlegen", options: { bullet: true } }],
    { x: 0.7, y: 5.72, w: 11.9, h: 1.05, fontSize: 14, paraSpaceAfter: 4 });
  s.addNotes("Ehrlich bleiben: Es gibt heute drei echte Patienten und keine echte Biopsietabelle. Das Werkzeug ist fertig, sobald die Daten kommen.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("geschrieben:", path.resolve(OUT));
})().catch((e) => { console.error(e); process.exit(1); });
