"""Pitch-Video rendern: Dashboard fotografieren, animierte Seite aufnehmen, als MP4 speichern.
Aufruf: .venv/Scripts/python docs/pitch/render.py   (Dashboard muss auf localhost:8501 laufen)"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def shots(pw):
    b = pw.chromium.launch()
    page = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.25)
    page.goto("http://localhost:8501")
    page.get_by_text("Demo-Daten laden und analysieren").first.wait_for(timeout=60000)
    page.screenshot(path=HERE / "shot_upload.png")
    page.locator('button:has-text("Demo-Daten laden und analysieren")').click()
    page.get_by_text("DEMO-DATEN mit eingebautem Muster").first.wait_for(timeout=600000)
    page.wait_for_timeout(2000)
    for tab, name in [("Übersicht", "overview"), ("Merkmale", "features"), ("Patient", "patient")]:
        page.locator(f'button[role="tab"]:has-text("{tab}")').click()
        page.wait_for_timeout(5000)
        if name == "patient":
            page.mouse.wheel(0, 250)
            page.wait_for_timeout(800)
        page.screenshot(path=HERE / f"shot_{name}.png")
        print("foto", name)
    b.close()


def record(pw):
    b = pw.chromium.launch()
    ctx = b.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(OUT),
                        record_video_size={"width": 1920, "height": 1080})
    t0 = time.time()
    page = ctx.new_page()
    page.goto((HERE / "video.html").as_uri())
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(1500)
    total = page.evaluate("window.totalMs")
    lead = time.time() - t0          # Vorlauf der Aufnahme bis zum Start der Animation
    page.evaluate("window.start()")
    page.wait_for_function("document.body.dataset.done === '1'", timeout=total + 20000)
    page.wait_for_timeout(1200)
    video = page.video
    ctx.close(); b.close()
    return Path(video.path()), total, lead


def main():
    with sync_playwright() as pw:
        if "--no-shots" not in sys.argv:
            shots(pw)
        webm, total, lead = record(pw)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = OUT / "kardio_pitch.mp4"
    # 1,5 s Vorlauf (Seite laedt) abschneiden, H.264, kompatibel mit PowerPoint/Teams
    subprocess.run([ffmpeg, "-y", "-ss", f"{lead + 0.3:.2f}", "-i", str(webm), "-t", str(total / 1000 + 1.0),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", "30",
                    "-movflags", "+faststart", str(mp4)], check=True, capture_output=True)
    webm.unlink(missing_ok=True)
    dest = Path.home() / "Downloads" / "kardio_pitch.mp4"
    shutil.copy(mp4, dest)
    print("fertig:", dest, f"{total/1000:.0f} s")


if __name__ == "__main__":
    main()
