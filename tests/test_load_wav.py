"""Area 2: load_wav_mono dtype normalization and channel handling.

Each integer dtype is normalized to float32 in [-1, 1] by a different constant,
and stereo input is averaged to mono. A wrong constant is a silent one-line bug.
"""
import numpy as np
import pytest
from scipy.io import wavfile

import app


def _write_wav(tmp_path, data, sr=44100):
    path = str(tmp_path / "sample.wav")
    wavfile.write(path, sr, data)
    return path


def test_load_wav_int16_normalized(tmp_path):
    data = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    path = _write_wav(tmp_path, data)
    sr, out = app.load_wav_mono(path)
    assert sr == 44100
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(16384 / 32768.0)
    assert out[2] == pytest.approx(-16384 / 32768.0)
    assert np.max(np.abs(out)) <= 1.0


def test_load_wav_int32_normalized(tmp_path):
    data = np.array([0, 2147483647, -2147483648], dtype=np.int32)
    path = _write_wav(tmp_path, data)
    _, out = app.load_wav_mono(path)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(2147483647 / 2147483648.0, abs=1e-6)
    assert out[2] == pytest.approx(-1.0)


def test_load_wav_uint8_centered(tmp_path):
    # uint8 WAV is unsigned, centered on 128.
    data = np.array([128, 255, 0], dtype=np.uint8)
    path = _write_wav(tmp_path, data)
    _, out = app.load_wav_mono(path)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx((255 - 128) / 128.0)
    assert out[2] == pytest.approx((0 - 128) / 128.0)


def test_load_wav_float32_passthrough(tmp_path):
    data = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    path = _write_wav(tmp_path, data)
    _, out = app.load_wav_mono(path)
    np.testing.assert_allclose(out, data, atol=1e-7)


def test_load_wav_stereo_averaged_to_mono(tmp_path):
    # Two channels; left=+full, right=-full should average to ~0.
    left = np.full(100, 10000, dtype=np.int16)
    right = np.full(100, -10000, dtype=np.int16)
    stereo = np.stack([left, right], axis=1)
    path = _write_wav(tmp_path, stereo)
    _, out = app.load_wav_mono(path)
    assert out.ndim == 1
    assert out.shape == (100,)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_load_wav_stereo_distinct_channels(tmp_path):
    left = np.full(50, 16384, dtype=np.int16)
    right = np.zeros(50, dtype=np.int16)
    stereo = np.stack([left, right], axis=1)
    path = _write_wav(tmp_path, stereo)
    _, out = app.load_wav_mono(path)
    # mean of (0.5, 0.0) == 0.25
    np.testing.assert_allclose(out, (16384 / 32768.0) / 2, atol=1e-6)
