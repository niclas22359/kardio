"""Ton fuer das Pitch-Video: deutsche Sprecherstimme je Szene + ruhige Musik mit leisem Herzschlag.
Aufruf: .venv/Scripts/python docs/pitch/audio.py   (nach render.py; mischt in kardio_pitch.mp4)"""
import shutil
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SR = 44100

# (Szenendauer in s, gesprochener Text) in der Reihenfolge der Szenen in video.html
SCENES = [
    (9, "Rund vierhundert Menschen bekommen in Deutschland jedes Jahr ein neues Herz. Und dann beginnt das Warten: "
        "Bis zu jeder Fünfte erlebt im ersten Jahr eine Abstoßung."),
    (11, "Entdeckt wird sie heute mit der Biopsie: ein Katheter im Herzen, sieben bis zwölf Mal im ersten Jahr. "
         "Sie zeigt nur den Moment. Zwischen zwei Terminen ist die Klinik blind."),
    (13, "Unsere Idee: dreißig Sekunden EKG täglich mit der Apple Watch, zu Hause. Daraus entsteht für jeden Patienten "
         "sein persönliches Normal. Driftet das EKG davon weg, erkennt das Modell das Muster und schaltet die Ampel."),
    (9, "So funktioniert Kardio. Zwei Tabellen hochladen, EKG-Daten und Biopsiebefunde, dann der rote Knopf."),
    (9, "Die Übersicht zeigt jeden Patienten mit seinem aktuellen Risiko, sortiert nach Dringlichkeit."),
    (10, "Die Merkmalsseite zeigt die Auswertung, die Kardiologen kennen: drei Befundgruppen, p-Werte, Trennschärfe. "
         "Automatisch statt in Handarbeit."),
    (10, "Beim einzelnen Patienten sieht man, wie das Risiko Tage vor der Biopsie steigt. Dazu ein Befundtext, "
         "dessen Zahlen das System selbst prüft."),
    (12, "Funktioniert die Maschine? Trainiert an vierzig Patienten, geprüft an dreißig anderen: Trennschärfe null Komma "
         "neun sieben. Ohne eingebautes Muster: null Komma fünf zwei, also nichts. Das sind Demo-Daten. "
         "Sie beweisen die Maschine, noch nicht die Medizin."),
    (12, "Was das bringt: Patienten werden gewarnt, bevor Symptome da sind. Ärzte sehen eine Ampel statt Datenberge. "
         "Kliniken setzen Biopsien nach Risiko statt nach Kalender."),
    (10, "Die Maschine steht. Jetzt braucht sie echte Biopsiedaten, eine Validierung mit Kardiologen "
         "und einen Pilot in einem Transplantationszentrum."),
    (7, "Kardio. Das Herz hört man, bevor es laut wird."),
]
VOICE = "Microsoft Katja"


def speak(text: str, path: Path) -> None:
    ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('{VOICE}')
$s.Rate = 1
$s.SetOutputToWaveFile('{path}')
$s.Speak(@'
{text}
'@)
$s.Dispose()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)


def music(seconds: float) -> np.ndarray:
    """Ruhiges Pad (Dur-Akkorde, langsam wechselnd) plus leiser Herzschlag bei 60 Schlaegen pro Minute."""
    t = np.arange(int(seconds * SR)) / SR
    chords = [[110, 164.81, 220, 277.18], [98, 146.83, 196, 246.94], [87.31, 130.81, 174.61, 220],
              [98, 146.83, 196, 261.63]]            # A, G, F, G-Dur-Varianten, tief und warm
    pad = np.zeros_like(t)
    seg = 16.0
    for i, ch in enumerate(chords * 20):
        start = i * seg
        if start >= seconds:
            break
        w = np.clip((t - start) / 4, 0, 1) * np.clip((start + seg + 4 - t) / 4, 0, 1)   # weiche Blenden
        for f in ch:
            pad += w * (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * (f * 1.003) * t)
                        + 0.25 * np.sin(2 * np.pi * (f * 2) * t)) / len(ch)
    pad *= 0.12 * (0.8 + 0.2 * np.sin(2 * np.pi * 0.05 * t))
    # Herzschlag: "lub-dub" alle 1 s, sehr leise, nur in den ersten 25 s und den letzten 8 s
    beat = np.zeros_like(t)
    env = lambda tt: np.exp(-tt * 18) * (tt >= 0)
    for k in range(int(seconds)):
        for off, amp in ((0.0, 1.0), (0.18, 0.6)):
            tt = t - (k + off)
            beat += amp * env(tt) * np.sin(2 * np.pi * 55 * tt)
    gate = ((t < 25) | (t > seconds - 8)).astype(float)
    beat *= 0.25 * gate
    mix = pad + beat
    fade = np.clip(t / 2, 0, 1) * np.clip((seconds - t) / 3, 0, 1)
    return (mix * fade).astype(np.float32)


def write_wav(path: Path, data: np.ndarray) -> None:
    pcm = np.clip(data, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())


def main():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    video = OUT / "kardio_pitch.mp4"
    # 1) Sprecherin je Szene erzeugen und messen
    voices = []
    for i, (dur, text) in enumerate(SCENES):
        p = OUT / f"voice_{i:02d}.wav"
        speak(text, p)
        with wave.open(str(p), "rb") as w:
            voices.append(w.getnframes() / w.getframerate())
    # 2) Szenen so lang machen, dass die Stimme hineinpasst (plus Luft), und ins Video schreiben
    durations = [max(d, round(v + 1.6)) for (d, _), v in zip(SCENES, voices)]
    html = (HERE / "video.html").read_text(encoding="utf-8")
    parts = html.split('<section class="scene" data-t="')
    assert len(parts) == len(SCENES) + 1
    html = parts[0] + "".join(f'<section class="scene" data-t="{d * 1000}"' + part.split('"', 1)[1]
                              for d, part in zip(durations, parts[1:]))
    (HERE / "video.html").write_text(html, encoding="utf-8")
    import sys
    subprocess.run([sys.executable, str(HERE / "render.py"), "--no-shots"], check=True)
    total = sum(durations) + 1.0
    offsets, start = [], 0.6
    for i, d in enumerate(durations):
        offsets.append((OUT / f"voice_{i:02d}.wav", start))
        start += d
    print("szenen (s):", durations, "gesamt", total)
    def read_wav(path: Path) -> np.ndarray:
        with wave.open(str(path), "rb") as w:
            sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
            data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32767
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        if sr != SR:   # einfache Umtastung
            data = np.interp(np.arange(0, len(data), sr / SR), np.arange(len(data)), data)
        return data

    track = music(total)
    voice = np.zeros_like(track)
    for p, off in offsets:
        v = read_wav(p) * 1.5
        i = int(off * SR)
        n = min(len(v), len(voice) - i)
        voice[i:i + n] += v[:n]
        # Musik unter der Stimme leicht absenken (Ducking)
        track[i:i + n] *= 0.55
    mix = np.clip(track + voice, -0.98, 0.98)
    write_wav(OUT / "mix.wav", mix)
    out = OUT / "kardio_pitch_ton.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-i", str(OUT / "mix.wav"),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(out)], check=True)
    dest = Path.home() / "Downloads" / "kardio_pitch.mp4"
    shutil.copy(out, dest)
    print("fertig mit Ton:", dest)


if __name__ == "__main__":
    main()
