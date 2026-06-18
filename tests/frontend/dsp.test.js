import { describe, it, expect, beforeAll } from 'vitest';
import { loadApp } from './load.js';

// NOTE: detectFFT / detectAC / fftRadix2 / noteET are currently unused by the
// shipping app (the live-mic mode was removed; analysis runs server-side). They
// are pure and self-contained, so these tests lock their behavior in case the
// on-device live mode returns. They also cover the FFT/autocorrelation math.

let w;
beforeAll(() => { w = loadApp(); });

const SR = 44100;

function sine(freq, n, sr = SR) {
  const buf = new Float64Array(n);
  for (let i = 0; i < n; i++) buf[i] = Math.sin(2 * Math.PI * freq * i / sr);
  return buf;
}

const centsErr = (got, want) => 1200 * Math.log2(got / want);

describe('noteET', () => {
  it('maps A4 to the reference pitch', () => {
    expect(w.noteET('A4', 440)).toBeCloseTo(440, 6);
    expect(w.noteET('A4', 442)).toBeCloseTo(442, 6);
  });

  it('is an octave apart per A step', () => {
    expect(w.noteET('A5', 440)).toBeCloseTo(880, 6);
    expect(w.noteET('A2', 440)).toBeCloseTo(110, 6);
  });
});

describe('fftRadix2', () => {
  it('transforms an impulse to a flat unit spectrum', () => {
    const n = 8;
    const re = new Float64Array(n); const im = new Float64Array(n);
    re[0] = 1;
    w.fftRadix2(re, im);
    for (let k = 0; k < n; k++) {
      expect(Math.hypot(re[k], im[k])).toBeCloseTo(1, 6);
    }
  });
});

describe('detectFFT', () => {
  it.each([220, 440, 880])('locks onto a %i Hz sine within a few cents', (f) => {
    const got = w.detectFFT(sine(f, 8192), SR, f);
    expect(got).not.toBeNull();
    expect(Math.abs(centsErr(got, f))).toBeLessThan(10);
  });

  it('returns null when the search band is empty', () => {
    // fExp above Nyquist -> no valid bins.
    expect(w.detectFFT(sine(440, 8192), SR, 1e9)).toBeNull();
  });
});

describe('detectAC', () => {
  it.each([110, 220, 440])('locks onto a %i Hz sine within a few cents', (f) => {
    const got = w.detectAC(sine(f, 8192), SR, f);
    expect(got).not.toBeNull();
    expect(Math.abs(centsErr(got, f))).toBeLessThan(10);
  });

  it('returns null on silence (no periodicity)', () => {
    expect(w.detectAC(new Float64Array(8192), SR, 440)).toBeNull();
  });
});
