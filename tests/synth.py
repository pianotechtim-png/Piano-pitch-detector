"""Synthetic audio helpers for exercising the detection pipeline.

Real fixtures would require committing binary WAV/MP3 files; instead we
generate piano-like tones on the fly. detect_8_a_notes is extremely accurate
on clean synthetic signals (measured pitch matches the input to <0.1 cent),
which lets the area-3 tests assert tight numbers.
"""
import numpy as np

SR = 44100


def piano_tone(f1, dur, n_partials=12, B=0.0, sr=SR):
    """A decaying tone built from `n_partials` partials.

    B is the inharmonicity coefficient: partial n sits at
    n*f1*sqrt(1 + B*n^2), matching how real piano partials run sharp.
    A short onset ramp + exponential decay give the attack/sustain shape the
    detector skips past and then measures.
    """
    t = np.arange(int(sr * dur)) / sr
    sig = np.zeros_like(t)
    for n in range(1, n_partials + 1):
        fn = n * f1 * np.sqrt(1 + B * n * n)
        if fn > sr / 2 - 100:
            break
        sig += (1.0 / n) * np.sin(2 * np.pi * fn * t)
    env = np.exp(-3.0 * t / dur)
    onset = np.minimum(1.0, t / 0.01)
    return (sig * env * onset).astype(np.float32)


def build_wav(freqs, tone_dur=1.0, gap=0.5, sr=SR, B=0.0):
    """Concatenate tones at `freqs`, separated by silence.

    The 0.5s gaps sit comfortably above the default min-gap (~0.29s) so each
    tone is detected as its own event rather than merged with its neighbor.
    """
    silence = np.zeros(int(sr * gap), dtype=np.float32)
    parts = [silence]
    for f in freqs:
        parts.append(piano_tone(f, tone_dur, B=B, sr=sr))
        parts.append(silence)
    y = np.concatenate(parts)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.8
    return y.astype(np.float32)
