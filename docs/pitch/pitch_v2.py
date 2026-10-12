"""Version 2 des Pitch-Videos: kurz, filmisch, Sternenhimmel, ein Satz pro Szene, nur Musik (keine Stimme).

Bilder werden mit virtueller Uhr Bild fuer Bild gerendert (30 Bilder je Sekunde), nicht in Echtzeit.
Dadurch ist jeder Szenenwechsel auf das Bild genau bekannt, und die Musik wird exakt darauf gebaut.
Aufruf: .venv/Scripts/python docs/pitch/pitch_v2.py  (nutzt shot_overview.png aus render.py)"""
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FRAMES = OUT / "v2_frames24"
HTML = HERE / "video2.html"
SR = 44100
FPS = 24
WORKERS = 3


def scene_durations() -> list[float]:
    html = HTML.read_text(encoding="utf-8")
    return [int(p.split('"', 1)[0]) / 1000 for p in html.split('<section class="scene" data-t="')[1:]]


def _render_range(args):
    first, last, total = args
    step = 1000 / FPS
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        page = b.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(HTML.as_uri())
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(800)
        cdp = page.context.new_cdp_session(page)
        cdp.send("Emulation.setVirtualTimePolicy", {"policy": "pause"})
        page.evaluate("window.start()")

        def advance(ms):
            done = {"ok": False}
            cdp.once("Emulation.virtualTimeBudgetExpired", lambda _=None: done.__setitem__("ok", True))
            cdp.send("Emulation.setVirtualTimePolicy", {"policy": "advance", "budget": ms})
            for _ in range(400):
                if done["ok"]:
                    return
                page.wait_for_timeout(5)

        # bis zum ersten eigenen Bild vorspulen (in kleinen Schritten, damit alle Timer sauber feuern)
        for _ in range(first):
            advance(step)
        for i in range(first, last):
            advance(step)
            page.screenshot(path=FRAMES / f"f{i:05d}.jpg", type="jpeg", quality=92, scale="css")
        b.close()
    return last - first


def render_frames(total: float) -> int:
    from concurrent.futures import ProcessPoolExecutor
    FRAMES.mkdir(parents=True, exist_ok=True)
    for f in FRAMES.glob("*"):
        f.unlink()
    n = int(round(total * FPS))
    bounds = [round(k * n / WORKERS) for k in range(WORKERS + 1)]
    jobs = [(bounds[k], bounds[k + 1], total) for k in range(WORKERS)]
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(_render_range, jobs))
    return n


# ---------- Musik, exakt auf die Szenen gebaut ----------

def _env(t, t0, attack, decay):
    tt = t - t0
    return np.where(tt < 0, 0, np.where(tt < attack, tt / max(attack, 1e-6), np.exp(-(tt - attack) / decay)))


def music(starts: list[float], durs: list[float], total: float) -> np.ndarray:
    """Szenen: 0 Zahl, 1 Abstossung (EKG), 2 Biopsie (Bedrohung), 3 Idee (Hoffnung), 4 Drift (Alarm),
    5 Produkt, 6 Beweis, 7 Logo. Alle Zeiten kommen aus den Szenenlaengen des Videos."""
    t = np.arange(int(total * SR)) / SR
    out = np.zeros_like(t)
    rng = np.random.default_rng(7)
    noise = rng.normal(size=len(t)).astype(np.float32)
    s = starts
    end = total

    # 1) Drone in D, ueber das ganze Video, langsam anschwellend bis zum Beweis, dann haltend
    swell = np.clip(t / s[6], 0, 1) ** 1.4
    for f, a in ((36.71, 1.0), (73.42, .55), (110.0, .3)):
        out += a * np.sin(2 * np.pi * f * t + .3 * np.sin(2 * np.pi * .06 * t)) * (0.07 + 0.09 * swell)

    # 2) Herzschlag 60 bpm: Szene 0 bis Ende Szene 1 deutlich, ab Biopsie-Szene aussetzend (Stille = Bedrohung),
    #    ab Idee leise zurueck, in der Logo-Szene ein letzter kraeftiger Schlag
    beat_gain = np.zeros_like(t)
    beat_gain += 1.0 * ((t >= 0) & (t < s[2]))
    beat_gain += 0.35 * ((t >= s[3]) & (t < s[6]))
    for k in range(int(total)):
        for off, amp in ((0.0, 1.0), (0.17, 0.55)):
            out += 0.22 * amp * _env(t, k + off, 0.004, 0.06) * np.sin(2 * np.pi * 52 * (t - k - off)) * beat_gain

    # 3) Akkordflaeche: dunkel (Dm) bis Szene 2, Hoffnung (F / C) ab Idee, gross (Bb -> F) im Beweis
    def pad(f_list, t0, t1, gain):
        w = np.clip((t - t0) / 2.0, 0, 1) * np.clip((t1 + 1.5 - t) / 2.0, 0, 1)
        for f in f_list:
            out[:] += gain * w * (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 1.004 * t)
                                  + 0.2 * np.sin(2 * np.pi * f * 2 * t))
    pad([146.8, 174.6, 220.0], 0, s[2], 0.030)                    # Dm
    pad([146.8, 174.6, 207.7], s[2], s[3], 0.034)                 # Dm mit kleiner Sexte: Bedrohung
    pad([174.6, 220.0, 261.6], s[3], s[4], 0.036)                 # F-Dur: Hoffnung
    pad([130.8, 164.8, 196.0], s[4], s[5], 0.036)                 # C-Dur
    pad([116.5, 146.8, 174.6, 233.1], s[5], s[6], 0.040)          # Bb-Dur
    pad([174.6, 220.0, 261.6, 349.2], s[6], end, 0.050)           # F-Dur gross: Beweis + Logo

    # 4) Pulsierende Achtel ab der Idee, lauter im Drift, stark im Beweis
    eighth = (t * 4) % 1
    pulse = np.exp(-eighth * 9) * np.sin(2 * np.pi * 146.8 * t)
    pulse_gain = (0.05 * ((t >= s[3]) & (t < s[4])) + 0.09 * ((t >= s[4]) & (t < s[6]))
                  + 0.12 * ((t >= s[6]) & (t < s[7])))
    out += pulse * pulse_gain * np.clip((t - s[3]) / 3, 0, 1)

    # 5) Szene 2 (Biopsie): drei tiefe Schlaege auf die Ringe (alle 2 s), Szene 4 (Drift): Alarm-Ton ab Sekunde 4,5
    for k in range(3):
        t0 = s[2] + 0.2 + 2.0 * k
        out += 0.45 * _env(t, t0, 0.003, 0.35) * np.sin(2 * np.pi * (48 + 60 * np.exp(-(t - t0) * 14)) * (t - t0))
    alarm_t0 = s[4] + 4.5
    alarm = _env(t, alarm_t0, 0.4, 2.5) * np.sin(2 * np.pi * 880 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    out += 0.06 * alarm

    # 6) Riser + Schlag an jedem Szenenwechsel; grosser Schlag beim Beweis und beim Logo
    for i, t0 in enumerate(s[1:], start=1):
        big = 1.7 if i in (6, 7) else 1.0
        tt = t - t0
        kick = _env(t, t0, 0.003, 0.22) * np.sin(2 * np.pi * (55 + 90 * np.exp(-np.clip(tt, 0, None) * 18)) * np.clip(tt, 0, None))
        hiss = _env(t, t0, 0.002, 0.25) * noise
        out += big * (0.55 * kick + 0.035 * hiss)
        r = np.clip((tt + 2.0) / 2.0, 0, 1) * (tt < 0)       # 2 s Riser vor dem Schlag
        out += 0.07 * big * r ** 3 * noise * (0.5 + 0.5 * np.sin(2 * np.pi * (300 + 2500 * r) * t))

    # 7) Logo-Szene: Hochton-Schimmer, danach alles ausklingen
    shimmer = _env(t, s[7], 0.6, 3.0) * (np.sin(2 * np.pi * 1760 * t) + np.sin(2 * np.pi * 2637 * t)) * 0.02
    out += shimmer
    fade = np.clip(t / 1.0, 0, 1) * np.clip((end - t) / 2.5, 0, 1)
    out = out * fade
    peak = np.max(np.abs(out)) or 1.0
    return (0.95 * out / peak).astype(np.float32)


def write_wav(path: Path, data: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    OUT.mkdir(exist_ok=True)
    durs = scene_durations()
    starts = [float(x) for x in np.cumsum([0] + durs[:-1])]
    total = float(sum(durs))
    print("szenen (s):", durs, "start bei", [round(x, 1) for x in starts], "gesamt", total)
    if "--audio-only" not in sys.argv:
        n = render_frames(total)
        print("bilder:", n)
    write_wav(OUT / "v2_music.wav", music(starts, durs, total))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = OUT / "kardio_pitch_v2.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(FRAMES / "f%05d.jpg"),
                    "-i", str(OUT / "v2_music.wav"), "-map", "0:v", "-map", "1:a", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", str(out)], check=True)
    dest = Path.home() / "Downloads" / "kardio_pitch_v2.mp4"
    shutil.copy(out, dest)
    print("fertig:", dest)


if __name__ == "__main__":
    main()
