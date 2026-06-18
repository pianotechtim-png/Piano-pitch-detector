"""Area 1: pure math and DSP helpers in app.py.

These functions are deterministic and underpin every result the app produces,
so they are the highest-value / lowest-effort things to lock down.
"""
import numpy as np
import pytest

import app


# --------------------------------------------------------------------------- #
# et_freq
# --------------------------------------------------------------------------- #
def test_et_freq_a4_is_concert_pitch():
    assert app.et_freq(69) == pytest.approx(440.0)


@pytest.mark.parametrize("midi, expected", [
    (21, 27.5),     # A0
    (33, 55.0),     # A1
    (45, 110.0),    # A2
    (57, 220.0),    # A3
    (69, 440.0),    # A4
    (81, 880.0),    # A5
    (93, 1760.0),   # A6
    (105, 3520.0),  # A7
])
def test_et_freq_all_a_octaves(midi, expected):
    assert app.et_freq(midi) == pytest.approx(expected, rel=1e-9)


def test_et_freq_matches_module_nominal_table():
    # NOMINAL_A is a hand-written table; et_freq must agree with it exactly.
    for midi, nominal in app.NOMINAL_A.items():
        assert app.et_freq(midi) == pytest.approx(nominal, rel=1e-9)


def test_et_freq_one_octave_is_double():
    assert app.et_freq(81) == pytest.approx(2 * app.et_freq(69))


def test_et_freq_respects_a4_reference():
    # Raising the A4 reference scales every note by the same ratio.
    assert app.et_freq(69, a4=442.0) == pytest.approx(442.0)
    assert app.et_freq(81, a4=442.0) == pytest.approx(884.0)


def test_et_freq_semitone_ratio():
    # One semitone up is the twelfth root of two.
    assert app.et_freq(70) / app.et_freq(69) == pytest.approx(2 ** (1 / 12))


# --------------------------------------------------------------------------- #
# stretched_target
# --------------------------------------------------------------------------- #
def test_stretched_target_zero_offset_equals_et():
    for midi in app.A_MIDI_ORDER:
        assert app.stretched_target(midi, 0.0) == pytest.approx(app.et_freq(midi))


def test_stretched_target_positive_offset_is_sharp():
    base = app.et_freq(81)
    assert app.stretched_target(81, 10.0) > base


def test_stretched_target_negative_offset_is_flat():
    base = app.et_freq(21)
    assert app.stretched_target(21, -28.0) < base


def test_stretched_target_offset_is_in_cents():
    # +1200 cents == one octave == double the frequency.
    assert app.stretched_target(69, 1200.0) == pytest.approx(2 * app.et_freq(69))


# --------------------------------------------------------------------------- #
# cents_from_target
# --------------------------------------------------------------------------- #
def test_cents_from_target_identical_is_zero():
    assert app.cents_from_target(440.0, 440.0) == pytest.approx(0.0)


def test_cents_from_target_octave_up_is_1200():
    assert app.cents_from_target(880.0, 440.0) == pytest.approx(1200.0)


def test_cents_from_target_octave_down_is_minus_1200():
    assert app.cents_from_target(220.0, 440.0) == pytest.approx(-1200.0)


def test_cents_from_target_one_semitone():
    assert app.cents_from_target(app.et_freq(70), app.et_freq(69)) == pytest.approx(100.0)


@pytest.mark.parametrize("measured, target", [
    (0.0, 440.0),
    (-5.0, 440.0),
    (440.0, 0.0),
    (440.0, -5.0),
    (0.0, 0.0),
])
def test_cents_from_target_non_positive_returns_none(measured, target):
    # Guard clause in the code: cannot take a log of a non-positive ratio.
    assert app.cents_from_target(measured, target) is None


# --------------------------------------------------------------------------- #
# nearest_a
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("midi", app.A_MIDI_ORDER)
def test_nearest_a_exact_pitches(midi):
    found_midi, cents = app.nearest_a(app.NOMINAL_A[midi])
    assert found_midi == midi
    assert cents == pytest.approx(0.0, abs=1e-6)


def test_nearest_a_slightly_sharp():
    # A4 +30 cents should still map to A4 with positive cents.
    freq = app.et_freq(69) * 2 ** (30 / 1200.0)
    found_midi, cents = app.nearest_a(freq)
    assert found_midi == 69
    assert cents == pytest.approx(30.0, abs=0.5)


def test_nearest_a_picks_closer_octave():
    # Just above A3 (220) should be A3, not A4.
    found_midi, _ = app.nearest_a(230.0)
    assert found_midi == 57


def test_nearest_a_returns_signed_cents():
    flat_midi, flat_cents = app.nearest_a(app.et_freq(69) * 2 ** (-20 / 1200.0))
    assert flat_midi == 69
    assert flat_cents < 0


# --------------------------------------------------------------------------- #
# yin_pitch — synthetic pure tones
# --------------------------------------------------------------------------- #
def _sine(freq, sr, dur, amp=0.5):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("freq", [110.0, 220.0, 440.0, 880.0])
def test_yin_pitch_on_pure_sine(freq):
    sr = 44100
    seg = _sine(freq, sr, 0.5)
    detected = app.yin_pitch(seg, sr, fmin=20, fmax=3800)
    assert detected is not None
    cents = 1200 * np.log2(detected / freq)
    assert abs(cents) < 5  # within 5 cents of truth


def test_yin_pitch_silence_returns_none():
    sr = 44100
    seg = np.zeros(sr // 2, dtype=np.float32)
    assert app.yin_pitch(seg, sr, fmin=20, fmax=3800) is None


def test_yin_pitch_degenerate_range_returns_none():
    sr = 44100
    seg = _sine(440.0, sr, 0.5)
    # fmin > fmax makes tau_min >= tau_max -> None
    assert app.yin_pitch(seg, sr, fmin=4000, fmax=20) is None


# --------------------------------------------------------------------------- #
# fundamental_fft / refined_mid_pitch / treble_fft_pitch
# --------------------------------------------------------------------------- #
def test_fundamental_fft_locks_onto_peak():
    sr = 44100
    freq = 220.0
    seg = _sine(freq, sr, 0.6)
    f = app.fundamental_fft(seg, sr, rough=freq)
    assert f == pytest.approx(freq, abs=1.0)


def test_fundamental_fft_handles_none_rough():
    sr = 44100
    seg = _sine(220.0, sr, 0.6)
    assert app.fundamental_fft(seg, sr, rough=None) is None


def test_refined_mid_pitch_falls_back_to_rough_on_silence():
    sr = 44100
    seg = np.zeros(sr // 2, dtype=np.float32)
    # No usable peak -> returns the rough estimate it was given.
    assert app.refined_mid_pitch(seg, sr, rough=440.0) == 440.0


def test_treble_fft_pitch_finds_high_tone():
    sr = 44100
    freq = 1760.0  # A6
    seg = _sine(freq, sr, 0.4)
    f = app.treble_fft_pitch(seg, sr, f_expected=freq)
    assert f == pytest.approx(freq, abs=2.0)


def test_treble_fft_pitch_silence_returns_none():
    sr = 44100
    seg = np.zeros(sr // 2, dtype=np.float32)
    assert app.treble_fft_pitch(seg, sr, f_expected=1760.0) is None


# --------------------------------------------------------------------------- #
# harmonic_f0 — inharmonic partial fit
# --------------------------------------------------------------------------- #
def _harmonic_tone(f1, sr, dur, n_partials=10, B=0.0, amp=0.5):
    """Sum of partials with optional inharmonicity coefficient B.

    Real piano partials sit sharp of n*f1 by sqrt(1 + B*n^2).
    """
    t = np.arange(int(sr * dur)) / sr
    sig = np.zeros_like(t)
    for n in range(1, n_partials + 1):
        fn = n * f1 * np.sqrt(1 + B * n * n)
        sig += (amp / n) * np.sin(2 * np.pi * fn * t)
    return sig.astype(np.float32)


def test_harmonic_f0_recovers_fundamental_harmonic_series():
    sr = 44100
    f1 = 55.0  # A1
    seg = _harmonic_tone(f1, sr, 0.6, n_partials=12, B=0.0)
    f0 = app.harmonic_f0(seg, sr, f0_seed=f1)
    assert f0 is not None
    assert abs(1200 * np.log2(f0 / f1)) < 10


def test_harmonic_f0_recovers_fundamental_with_inharmonicity():
    sr = 44100
    f1 = 55.0
    seg = _harmonic_tone(f1, sr, 0.6, n_partials=12, B=0.0008)
    f0 = app.harmonic_f0(seg, sr, f0_seed=f1)
    assert f0 is not None
    # The inharmonicity fit should recover f1 better than naive n*f1 spacing.
    assert abs(1200 * np.log2(f0 / f1)) < 20


def test_harmonic_f0_none_seed_returns_none():
    sr = 44100
    seg = _harmonic_tone(55.0, sr, 0.6)
    assert app.harmonic_f0(seg, sr, f0_seed=None) is None


def test_harmonic_f0_silence_returns_none():
    sr = 44100
    seg = np.zeros(int(sr * 0.6), dtype=np.float32)
    assert app.harmonic_f0(seg, sr, f0_seed=55.0) is None


# --------------------------------------------------------------------------- #
# _parabolic
# --------------------------------------------------------------------------- #
def test_parabolic_returns_index_at_boundaries():
    spec = np.array([1.0, 2.0, 3.0])
    assert app._parabolic(spec, 0) == 0
    assert app._parabolic(spec, len(spec) - 1) == len(spec) - 1


def test_parabolic_refines_toward_taller_neighbor():
    # Symmetric peak -> no shift.
    spec = np.array([1.0, 4.0, 1.0])
    assert app._parabolic(spec, 1) == pytest.approx(1.0)
    # Right neighbor taller -> estimate shifts right of the sampled bin.
    spec = np.array([1.0, 4.0, 3.0])
    assert app._parabolic(spec, 1) > 1.0
