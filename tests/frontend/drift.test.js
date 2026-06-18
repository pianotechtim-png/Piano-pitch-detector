import { describe, it, expect, beforeAll } from 'vitest';
import { loadApp } from './load.js';

let w;
beforeAll(() => { w = loadApp(); });

// cents helper for building "measured" frequencies relative to a baseline.
const bend = (f, cents) => f * Math.pow(2, cents / 1200);

describe('driftVerdict', () => {
  it('is green while holding tuning (|w| <= 2)', () => {
    expect(w.driftVerdict(0)).toEqual(['green', 'Holding its tuning']);
    expect(w.driftVerdict(2)[0]).toBe('green');
    expect(w.driftVerdict(-2)[0]).toBe('green');
  });

  it('is yellow for moderate drift (2 < |w| <= 15)', () => {
    expect(w.driftVerdict(2.1)[0]).toBe('yellow');
    expect(w.driftVerdict(15)[0]).toBe('yellow');
    expect(w.driftVerdict(-8)).toEqual(['yellow', 'Drifted — tuning recommended']);
  });

  it('is red for major drift, split by direction', () => {
    expect(w.driftVerdict(-15.1)).toEqual(['red', 'Major drift — pitch raise likely']);
    expect(w.driftVerdict(15.1)).toEqual(['red', 'Major drift — pitch correction likely']);
  });
});

describe('computeDrift', () => {
  it('reports zero drift when measured equals baseline', () => {
    const r = w.computeDrift([{ note_label: 'A4', freq_measured: 440 }], { A4: 440 });
    expect(r.weighted).toBe(0);
    expect(r.rows).toEqual([{ label: 'A4', drift: 0, core: true }]);
  });

  it('measures drift in cents', () => {
    const r = w.computeDrift(
      [{ note_label: 'A4', freq_measured: bend(440, 10) }], { A4: 440 });
    expect(r.weighted).toBeCloseTo(10, 1);
  });

  it('half-weights A6 in the core average', () => {
    // A4 = 0c (weight 1), A6 = +12c (weight 0.5) -> (0*1 + 12*0.5)/1.5 = 4.
    const r = w.computeDrift([
      { note_label: 'A4', freq_measured: 440 },
      { note_label: 'A6', freq_measured: bend(1760, 12) },
    ], { A4: 440, A6: 1760 });
    expect(r.weighted).toBeCloseTo(4, 1);
  });

  it('ignores notes missing from the baseline', () => {
    const r = w.computeDrift([
      { note_label: 'A4', freq_measured: 440 },
      { note_label: 'A5', freq_measured: 880 },   // no baseline entry
    ], { A4: 440 });
    expect(r.rows.map(x => x.label)).toEqual(['A4']);
  });

  it('ignores non-positive measured frequencies', () => {
    const r = w.computeDrift([{ note_label: 'A4', freq_measured: 0 }], { A4: 440 });
    expect(r.rows).toEqual([]);
    expect(r.weighted).toBeNull();
  });

  it('includes non-core notes as rows but not in the weighted average', () => {
    const r = w.computeDrift([
      { note_label: 'A0', freq_measured: bend(27.5, 20) },
    ], { A0: 27.5 });
    expect(r.rows[0]).toMatchObject({ label: 'A0', core: false });
    expect(r.weighted).toBeNull();   // A0 has no core weight
  });
});
