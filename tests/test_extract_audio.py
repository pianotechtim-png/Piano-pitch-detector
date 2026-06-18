"""extract_audio: the ffmpeg wrapper.

ffmpeg isn't available in CI, and we don't want to shell out anyway, so we
mock subprocess.run and assert the command line. The flags matter: -vn drops
video, -t caps duration, -ac/-ar force mono 44.1k -- a regression in any of
these silently changes every measurement downstream.
"""
import subprocess

import pytest

import app


def test_extract_audio_builds_expected_ffmpeg_command(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    out = app.extract_audio("/tmp/input.mov", max_seconds=30)

    assert out == "/tmp/input.mov.wav"
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "-vn" in cmd                       # drop video stream
    assert "-i" in cmd and "/tmp/input.mov" in cmd
    assert cmd[cmd.index("-t") + 1] == "30"   # duration cap honored
    assert cmd[cmd.index("-ac") + 1] == "1"   # mono
    assert cmd[cmd.index("-ar") + 1] == "44100"
    assert cmd[-1] == "/tmp/input.mov.wav"    # output path is last
    assert captured["kwargs"]["check"] is True


def test_extract_audio_default_max_seconds(monkeypatch):
    captured = {}
    monkeypatch.setattr(app.subprocess, "run",
                        lambda cmd, **kw: captured.update(cmd=cmd))
    app.extract_audio("/tmp/clip.mp4")
    assert captured["cmd"][captured["cmd"].index("-t") + 1] == "45"


def test_extract_audio_propagates_ffmpeg_failure(monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(app.subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        app.extract_audio("/tmp/bad.mov")
