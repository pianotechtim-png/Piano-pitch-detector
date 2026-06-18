"""Area 4: Flask routes and /analyze input validation.

The DSP itself is exercised elsewhere; here we mock extract_audio and
detect_8_a_notes so we can drive the request-handling logic in isolation:
file presence, preset aliasing/fallback, a4 clamping, and error mapping.
"""
import io
import subprocess

import pytest

import app


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace the heavy audio pipeline with a recorder.

    extract_audio becomes a no-op; detect_8_a_notes records the kwargs it was
    called with so tests can assert what the route passed through.
    """
    calls = {}

    def fake_extract(src_path, max_seconds=45):
        return src_path + ".wav"

    def fake_detect(audio_path, min_gap_ms=300, preset="standard", a4=440.0):
        calls["min_gap_ms"] = min_gap_ms
        calls["preset"] = preset
        calls["a4"] = a4
        return [], {"notes_detected": 0}

    monkeypatch.setattr(app, "extract_audio", fake_extract)
    monkeypatch.setattr(app, "detect_8_a_notes", fake_detect)
    monkeypatch.setattr(app.os, "unlink", lambda *a, **k: None)
    return calls


def _post(client, data=None, filename="a.wav", file_bytes=b"RIFFfake"):
    payload = {"file": (io.BytesIO(file_bytes), filename)}
    if data:
        payload.update(data)
    return client.post("/analyze", data=payload, content_type="multipart/form-data")


# --------------------------------------------------------------------------- #
# Static / simple routes
# --------------------------------------------------------------------------- #
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.content_type


def test_service_worker_is_javascript_and_uncached(client):
    resp = client.get("/sw.js")
    assert resp.status_code == 200
    assert resp.content_type.startswith("application/javascript")
    # The SW must never be cached or clients get stuck on a stale worker.
    assert resp.headers["Cache-Control"] == "no-cache"


def test_manifest_has_manifest_mimetype(client):
    resp = client.get("/manifest.json")
    assert resp.status_code == 200
    assert resp.content_type.startswith("application/manifest+json")


# --------------------------------------------------------------------------- #
# /analyze validation
# --------------------------------------------------------------------------- #
def test_analyze_missing_file_returns_400(client):
    resp = client.post("/analyze", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_analyze_happy_path_returns_summary(client, stub_pipeline):
    resp = _post(client)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["count"] == 0
    assert body["summary"]["preset"] == "standard"
    assert body["summary"]["a4"] == 440.0


@pytest.mark.parametrize("alias, expected", [
    ("average", "standard"),
    ("grand", "standard"),
    ("standard", "standard"),
    ("spinet", "spinet"),
])
def test_analyze_preset_aliasing(client, stub_pipeline, alias, expected):
    _post(client, data={"preset": alias})
    assert stub_pipeline["preset"] == expected


def test_analyze_unknown_preset_falls_back_to_default(client, stub_pipeline):
    _post(client, data={"preset": "bananas"})
    assert stub_pipeline["preset"] == app.DEFAULT_PRESET


@pytest.mark.parametrize("a4_in, expected", [
    ("440", 440.0),
    ("442", 442.0),
    ("415", 415.0),
    ("399", 440.0),        # below 400 -> clamp to default
    ("481", 440.0),        # above 480 -> clamp to default
    ("notanumber", 440.0),  # non-numeric -> default
])
def test_analyze_a4_clamping(client, stub_pipeline, a4_in, expected):
    _post(client, data={"a4": a4_in})
    assert stub_pipeline["a4"] == expected


def test_analyze_min_gap_passed_through(client, stub_pipeline):
    _post(client, data={"minGap": "500"})
    assert stub_pipeline["min_gap_ms"] == 500


def test_analyze_ffmpeg_failure_returns_500(client, monkeypatch):
    def boom(src_path, max_seconds=45):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(app, "extract_audio", boom)
    monkeypatch.setattr(app.os, "unlink", lambda *a, **k: None)
    resp = _post(client)
    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Could not read audio from that file"


def test_analyze_unexpected_error_returns_500(client, monkeypatch):
    def boom(src_path, max_seconds=45):
        return src_path + ".wav"

    def detect_boom(*a, **k):
        raise ValueError("kaboom")

    monkeypatch.setattr(app, "extract_audio", boom)
    monkeypatch.setattr(app, "detect_8_a_notes", detect_boom)
    monkeypatch.setattr(app.os, "unlink", lambda *a, **k: None)
    resp = _post(client)
    assert resp.status_code == 500
    assert resp.get_json()["error"] == "kaboom"
