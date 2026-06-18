"""Area 3: detect_8_a_notes, the order-aware detection/scoring core.

This is the most intricate logic in the app: RMS event segmentation, the
monotonic slot-assignment walk (ACCEPT_CENTS), per-register pitch methods
(harmonic for bass, FFT for treble), the verdict thresholds, and the several
empty/degenerate return shapes. We drive it end-to-end with synthesized WAVs.
"""
import numpy as np
import pytest
from scipy.io import wavfile

import app
from tests.synth import SR, build_wav, piano_tone


def _detect(tmp_path, y, **kwargs):
    path = str(tmp_path / "tones.wav")
    wavfile.write(path, SR, y)
    return app.detect_8_a_notes(path, **kwargs)


def _et_all():
    return [app.et_freq(m) for m in app.A_MIDI_ORDER]


def _shifted(cents):
    return [app.et_freq(m) * 2 ** (cents / 1200.0) for m in app.A_MIDI_ORDER]


# --------------------------------------------------------------------------- #
# Detection, labeling, ordering
# --------------------------------------------------------------------------- #
def test_detects_all_eight_a_notes(tmp_path):
    notes, summary = _detect(tmp_path, build_wav(_et_all()), preset="standard")
    assert summary["notes_detected"] == 8
    assert [n["note_label"] for n in notes] == \
        ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    assert summary["missing_notes"] == []
    assert summary["rejected_blobs"] == 0


def test_measured_pitch_matches_input(tmp_path):
    notes, _ = _detect(tmp_path, build_wav(_et_all()), preset="standard")
    by_midi = {n["midi"]: n for n in notes}
    for midi in app.A_MIDI_ORDER:
        assert by_midi[midi]["freq_measured"] == pytest.approx(app.et_freq(midi), rel=2e-3)
        assert abs(by_midi[midi]["cents_from_et"]) < 5


def test_index_and_core_flags(tmp_path):
    notes, _ = _detect(tmp_path, build_wav(_et_all()), preset="standard")
    by_midi = {n["midi"]: n for n in notes}
    # index is the 1-based position in A0..A7 order.
    assert by_midi[21]["index"] == 1
    assert by_midi[105]["index"] == 8
    # core == membership in CORE_WEIGHTS (A2..A6).
    for midi in app.A_MIDI_ORDER:
        assert by_midi[midi]["core"] is (midi in app.CORE_WEIGHTS)


def test_partial_set_only_reports_played_notes(tmp_path):
    # Play just A2 and A3; the rest must be reported missing, not invented.
    freqs = [app.et_freq(45), app.et_freq(57)]
    notes, summary = _detect(tmp_path, build_wav(freqs), preset="standard")
    assert [n["note_label"] for n in notes] == ["A2", "A3"]
    assert summary["notes_detected"] == 2
    assert set(summary["missing_notes"]) == {"A4", "A5", "A6"}


# --------------------------------------------------------------------------- #
# Verdict thresholds (green / yellow / red, and the red sign split)
# --------------------------------------------------------------------------- #
def test_verdict_near_pitch_reads_yellow_not_green(tmp_path):
    # Tones placed exactly on the standard stretch curve -> ~0 deviation.
    # Even on-level the piano still needs a tuning, so this must read yellow
    # and never say "on pitch" -- a reassuring green verdict costs tunings.
    freqs = [app.stretched_target(m, app.STRETCH_PRESETS["standard"][m])
             for m in app.A_MIDI_ORDER]
    _, summary = _detect(tmp_path, build_wav(freqs), preset="standard")
    assert abs(summary["weighted_avg_cents"]) <= app.VERDICT_GREEN
    assert summary["verdict"] == "yellow"
    assert summary["verdict"] != "green"
    label = summary["verdict_label"].lower()
    assert "on pitch" not in label
    assert "fine tuning" not in label and "fine-tuning" not in label
    assert "no pitch raise needed" in label


def test_verdict_yellow_mild_flat(tmp_path):
    _, summary = _detect(tmp_path, build_wav(_shifted(-10)), preset="standard")
    assert app.VERDICT_GREEN < abs(summary["weighted_avg_cents"]) <= app.VERDICT_RED
    assert summary["verdict"] == "yellow"
    assert summary["verdict_label"] == "Tuning recommended, no pitch raise necessary"


def test_verdict_red_flat_is_pitch_raise(tmp_path):
    _, summary = _detect(tmp_path, build_wav(_shifted(-30)), preset="standard")
    assert summary["weighted_avg_cents"] < -app.VERDICT_RED
    assert summary["verdict"] == "red"
    assert summary["verdict_label"] == "Pitch raise recommended"


def test_verdict_red_sharp_is_pitch_correction(tmp_path):
    _, summary = _detect(tmp_path, build_wav(_shifted(30)), preset="standard")
    assert summary["weighted_avg_cents"] > app.VERDICT_RED
    assert summary["verdict"] == "red"
    assert summary["verdict_label"] == "Pitch correction recommended"


def test_verdict_none_with_too_few_core_notes(tmp_path):
    # Only two core notes -> below the 3-note minimum for a verdict.
    freqs = [app.et_freq(45), app.et_freq(57)]
    _, summary = _detect(tmp_path, build_wav(freqs), preset="standard")
    assert summary["core_notes_detected"] == 2
    assert summary["weighted_avg_cents"] is None
    assert summary["verdict"] == "none"


# --------------------------------------------------------------------------- #
# Stretch presets and a4 reference
# --------------------------------------------------------------------------- #
def test_railsback_offset_follows_preset(tmp_path):
    y = build_wav([app.et_freq(21)], tone_dur=1.2)
    notes_spinet, _ = _detect(tmp_path, y, preset="spinet")
    notes_std, _ = _detect(tmp_path, y, preset="standard")
    assert notes_spinet[0]["railsback_offset"] == app.STRETCH_PRESETS["spinet"][21]
    assert notes_std[0]["railsback_offset"] == app.STRETCH_PRESETS["standard"][21]
    # Deeper bass stretch on a spinet -> different stretched target.
    assert notes_spinet[0]["freq_stretched"] != notes_std[0]["freq_stretched"]


def test_unknown_preset_falls_back_to_default_curve(tmp_path):
    y = build_wav([app.et_freq(45)])
    notes, _ = _detect(tmp_path, y, preset="does-not-exist")
    assert notes[0]["railsback_offset"] == app.STRETCH_PRESETS[app.DEFAULT_PRESET][45]


def test_a4_reference_shifts_targets(tmp_path):
    # A4 played at 442 and scored against a4=442 reads ~0 cents from ET.
    y = build_wav([app.et_freq(69, 442.0)])
    notes, _ = _detect(tmp_path, y, preset="standard", a4=442.0)
    assert notes[0]["freq_et"] == pytest.approx(442.0, abs=0.5)
    assert abs(notes[0]["cents_from_et"]) < 5


# --------------------------------------------------------------------------- #
# Garbage rejection and the >8-event cap
# --------------------------------------------------------------------------- #
def test_rejects_blob_that_fits_no_slot(tmp_path):
    # ~311 Hz sits >230 cents from both A3 (220) and A4 (440): pure garbage.
    sil = np.zeros(int(SR * 0.5), dtype=np.float32)
    y = np.concatenate([
        sil, piano_tone(311.0, 1.0), sil,
        piano_tone(app.et_freq(69), 1.0), sil,
    ]).astype(np.float32)
    notes, summary = _detect(tmp_path, y, preset="standard")
    assert [n["note_label"] for n in notes] == ["A4"]
    assert summary["rejected_blobs"] == 1


def test_more_than_eight_events_keeps_strongest_eight(tmp_path):
    # Nine tones (A0..A7 + a quiet extra A4). The cap trims to 8 by energy
    # before assignment, so the trimmed event is not counted as rejected.
    sil = np.zeros(int(SR * 0.5), dtype=np.float32)
    parts = [sil]
    for f in _et_all():
        parts += [piano_tone(f, 1.0), sil]
    parts += [piano_tone(app.et_freq(69), 1.0) * 0.2, sil]  # quiet 9th
    y = np.concatenate(parts).astype(np.float32)
    _, summary = _detect(tmp_path, y, preset="standard")
    assert summary["notes_detected"] == 8
    assert summary["rejected_blobs"] == 0


# --------------------------------------------------------------------------- #
# Empty / degenerate audio return shapes
# --------------------------------------------------------------------------- #
def test_empty_audio_returns_empty_summary(tmp_path):
    notes, summary = _detect(tmp_path, np.zeros(0, dtype=np.float32))
    assert notes == []
    assert summary["notes_detected"] == 0
    # The empty-array early return lists ALL eight A's as missing.
    assert summary["missing_notes"] == [app.A_LABELS[m] for m in app.A_MIDI_ORDER]
    assert summary["avg_cents_from_stretch"] is None


def test_too_short_audio_returns_empty_summary(tmp_path):
    # Shorter than one analysis frame (~0.09s) -> no RMS frames.
    notes, summary = _detect(tmp_path, np.ones(100, dtype=np.float32) * 0.1)
    assert notes == []
    assert summary["notes_detected"] == 0
    assert summary["missing_notes"] == [app.A_LABELS[m] for m in app.A_MIDI_ORDER]


def test_silence_returns_no_notes(tmp_path):
    # Long pure silence: frames exist but none clear the energy threshold.
    notes, summary = _detect(tmp_path, np.zeros(SR, dtype=np.float32))
    assert notes == []
    assert summary["notes_detected"] == 0
    assert summary["verdict"] == "none"


# --------------------------------------------------------------------------- #
# Inharmonic bass (exercises the harmonic_f0 path inside the pipeline)
# --------------------------------------------------------------------------- #
def test_inharmonic_bass_still_detected(tmp_path):
    # A0/A1/A2 carry realistic inharmonicity; they must still land on-pitch
    # via the harmonic-partial method rather than the weak fundamental.
    freqs = [app.et_freq(21), app.et_freq(33), app.et_freq(45)]
    notes, summary = _detect(tmp_path, build_wav(freqs, B=0.0004), preset="standard")
    assert [n["note_label"] for n in notes] == ["A0", "A1", "A2"]
    for n in notes:
        assert abs(n["cents_from_et"]) < 25
