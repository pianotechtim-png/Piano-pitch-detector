import { describe, it, expect, beforeAll } from 'vitest';
import { loadApp, hueOf } from './load.js';

let w;
beforeAll(() => { w = loadApp(); });

describe('centsColor', () => {
  it('returns the dim color for null / undefined / NaN', () => {
    expect(w.centsColor(null)).toBe('var(--dim)');
    expect(w.centsColor(undefined)).toBe('var(--dim)');
    expect(w.centsColor(NaN)).toBe('var(--dim)');
  });

  it('is green (hue 142) exactly on target', () => {
    expect(hueOf(w.centsColor(0))).toBe(142);
  });

  it('runs toward red below pitch and blue above', () => {
    // -15c -> hue 8 (red), +15c -> hue 212 (blue), 0c -> 142 (green).
    expect(hueOf(w.centsColor(-15))).toBe(8);
    expect(hueOf(w.centsColor(15))).toBe(212);
    expect(hueOf(w.centsColor(-7))).toBeLessThan(142);
    expect(hueOf(w.centsColor(7))).toBeGreaterThan(142);
  });

  it('clamps beyond the +-15c verdict threshold', () => {
    expect(w.centsColor(-100)).toBe(w.centsColor(-15));
    expect(w.centsColor(100)).toBe(w.centsColor(15));
  });

  it('always emits a well-formed hsl with 75% sat / 55% light', () => {
    for (const c of [-30, -10, -1, 0, 1, 10, 30]) {
      expect(w.centsColor(c)).toMatch(/^hsl\(\d+,75%,55%\)$/);
    }
  });
});

describe('fillBar', () => {
  it('grows from the center to the left when flat', () => {
    const html = w.fillBar(-25);
    expect(html).toContain('right:50%');
    expect(html).toContain('width:25%');   // |clamp(-25,-50,50)|/50*50
  });

  it('grows from the center to the right when sharp', () => {
    const html = w.fillBar(25);
    expect(html).toContain('left:50%');
    expect(html).toContain('width:25%');
  });

  it('clamps the bar width at +-50 cents', () => {
    expect(w.fillBar(100)).toContain('width:50%');
    expect(w.fillBar(-100)).toContain('width:50%');
  });

  it('paints the bar with the matching centsColor', () => {
    expect(w.fillBar(10)).toContain(w.centsColor(10));
  });
});
