# Piano Pitch Detector

A web app that estimates how far a piano's overall pitch sits from standard by
listening to the eight **A** notes (A0–A7). Record the A's on your phone, upload
the clip, and the app reports each A's deviation in cents against a stretched
tuning curve plus a single traffic-light verdict.

It measures **overall pitch level only** — it cannot judge unison quality or
fine tuning. Even a piano reading right on level still needs a regular tuning,
so the closest verdict is yellow ("no pitch raise needed — regular tuning
recommended"), never a reassuring green "on pitch".

## How it works

- **`app.py`** — a Flask backend. `POST /analyze` takes an audio/video file,
  extracts mono 44.1 kHz audio with `ffmpeg`, finds the eight A onsets by RMS
  segmentation, and measures each one:
  - **Bass (A0–A2):** `harmonic_f0` — fits the inharmonic partial series, since
    the fundamental is weak on phone mics.
  - **Mid (A3–A5):** direct fundamental FFT peak.
  - **Treble (A6–A7):** FFT peak in a wide band (YIN octave-errors up here).
  - Notes are matched to slots with an order-aware, monotonic walk so a very
    flat note is still identified without being mis-snapped to a neighbor.
  - A weighted A2–A6 core average drives the green/yellow/red verdict.
- **`Static/index.html`** — the single-page UI Flask serves at `/`. Handles
  upload, renders the per-note results and verdict, and includes a monitoring
  mode that tracks drift against a saved baseline (in `localStorage`).
- **`Static/sw.js` / `Static/manifest.json`** — PWA bits so the app installs to
  the home screen and works offline for review.

## Running locally

```bash
pip install -r requirements-dev.txt   # runtime deps + pytest
python app.py                         # serves on http://localhost:5000
```

`ffmpeg` must be on the `PATH` for `/analyze` to work (the upload path shells
out to it). `/health` and the static UI work without it.

## Tests

Two suites:

```bash
pytest                  # Python: backend DSP, scoring, routes (see pytest.ini)
npm install && npm test # JavaScript: the shipping frontend logic, via jsdom
```

The JS tests load the real `Static/index.html` in jsdom and call its functions
directly, so they exercise exactly what ships (no duplicated logic to drift).
Both suites run in CI (`.github/workflows/tests.yml`) on every push and PR.

## Deployment

Configured for a PaaS (Procfile + `build.sh` + `runtime.txt`); currently runs on
Render, deploying from `main`:

```
web: gunicorn app:app --timeout 300 --workers 1
```

The host **must provide `ffmpeg`** (e.g. an apt step or a Docker image) or audio
analysis returns "Could not read audio from that file".

## Project layout

```
app.py                      Flask backend + DSP
Static/index.html           single-page UI (served at /)
Static/sw.js, manifest.json PWA assets
tests/                      pytest suite (Python)
tests/frontend/             vitest suite (JS, jsdom)
.github/workflows/tests.yml CI
.claude/                    SessionStart hook for Claude Code on the web
```
