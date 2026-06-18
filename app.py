from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from scipy.io import wavfile
import tempfile
import os
import subprocess

app = Flask(__name__, static_folder='Static')
CORS(app)

# Note labels per MIDI
A_LABELS = {21:'A0', 33:'A1', 45:'A2', 57:'A3', 69:'A4', 81:'A5', 93:'A6', 105:'A7'}

# Two stretch presets (cents from equal temperament), derived from 13 real
# PianoScope tunings (June 2026). Consoles/spinets need a much deeper bass;
# grands and uprights share one curve (within ~3c at every A). A2-A6 are
# nearly size-independent, which is why they form the core verdict.
STRETCH_PRESETS = {
    'spinet':   {21:-58.0, 33:-21.0, 45:-8.0, 57:-3.5, 69:0.0, 81:3.6, 93:12.0, 105:30.0},
    'standard': {21:-28.0, 33:-10.0, 45:-4.0, 57:-2.0, 69:0.0, 81:3.4, 93:10.5, 105:28.0},
}
PRESET_ALIASES = {'average': 'standard', 'grand': 'standard'}  # legacy clients
DEFAULT_PRESET = 'standard'

# Core verdict: A2-A6 weighted (A6 half-weight). A0/A1/A7 are informational
# only -- their "correct" stretch varies too much piano-to-piano to judge.
CORE_WEIGHTS = {45: 1.0, 57: 1.0, 69: 1.0, 81: 1.0, 93: 0.5}
# This test reads overall pitch LEVEL only -- it cannot judge unisons or
# tuning quality, so even a piano sitting right at the target level still
# needs a tuning. We deliberately never show a green / "on pitch" verdict:
# that reads as "nothing is wrong" and costs tunings. The closest tier reads
# YELLOW -- no pitch raise needed, but a regular tuning is still recommended.
VERDICT_GREEN = 2.0   # |weighted avg| <= this -> on level: no pitch raise needed
VERDICT_RED = 15.0    # |weighted avg| >  this -> pitch raise/correction

# Backwards-compat: RAILSBACK_A holds (label, offset) for the selected curve.
# Built per-request now; this default keeps module-level helpers working.
RAILSBACK_A = {m: (A_LABELS[m], STRETCH_PRESETS[DEFAULT_PRESET][m]) for m in A_LABELS}
A_MIDI_ORDER = [21, 33, 45, 57, 69, 81, 93, 105]
NOMINAL_A = {21:27.5, 33:55.0, 45:110.0, 57:220.0, 69:440.0,
             81:880.0, 93:1760.0, 105:3520.0}
HARMONIC_NOTES = {21, 33, 45}   # use harmonic method (weak fundamental)

# The first part of a piano tone is hammer/attack junk (broadband noise,
# longitudinal string modes) -- never measure it. Skip into the sustain, then
# analyze a longer window (better bass resolution: 0.6s of A1 = ~33 cycles).
ATTACK_SKIP_S = 0.18
ANALYSIS_WIN_S = 0.60

def nearest_a(freq):
    best, best_c = None, None
    for midi, nom in NOMINAL_A.items():
        c = 1200.0 * np.log2(freq / nom)
        if best is None or abs(c) < abs(best_c):
            best, best_c = midi, c
    return best, best_c

def et_freq(midi, a4=440.0):
    return a4 * (2 ** ((midi - 69) / 12.0))

def stretched_target(midi, offset_cents, a4=440.0):
    return et_freq(midi, a4) * (2 ** (offset_cents / 1200.0))

def cents_from_target(measured, target):
    if measured <= 0 or target <= 0:
        return None
    return 1200.0 * np.log2(measured / target)

def extract_audio(src_path, max_seconds=45):
    """Extract mono 44.1k WAV. -vn drops the video stream entirely (fast for
    .mov/.mp4); -t caps duration so a long clip can't exceed the time budget."""
    wav_path = src_path + '.wav'
    subprocess.run(
        ['ffmpeg', '-y', '-vn', '-i', src_path,
         '-t', str(max_seconds),
         '-ac', '1', '-ar', '44100',
         '-f', 'wav', wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def load_wav_mono(path):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data

def yin_pitch(seg, sr, fmin, fmax, thresh=0.15):
    seg = seg.astype(np.float64); seg = seg - np.mean(seg)
    if np.max(np.abs(seg)) < 1e-6:
        return None
    N = len(seg)
    tau_min = max(int(sr / fmax), 2)
    tau_max = min(int(sr / fmin), N // 2)
    if tau_min >= tau_max:
        return None
    fft = np.fft.rfft(seg, 2 * N)
    acf = np.fft.irfft(fft * np.conj(fft))[:tau_max]
    sq = seg ** 2
    csum = np.concatenate([[0], np.cumsum(sq)])
    d = np.zeros(tau_max)
    for tau in range(1, tau_max):
        d[tau] = csum[N - tau] + (csum[N] - csum[tau]) - 2 * acf[tau]
    cmnd = np.zeros(tau_max); cmnd[0] = 1; running = 0.0
    for tau in range(1, tau_max):
        running += d[tau]
        cmnd[tau] = d[tau] * tau / running if running > 0 else 1
    tau = tau_min; tau_est = None
    while tau < tau_max:
        if cmnd[tau] < thresh:
            while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_est = tau; break
        tau += 1
    if tau_est is None:
        tau_est = tau_min + int(np.argmin(cmnd[tau_min:tau_max]))
    if 1 <= tau_est < tau_max - 1:
        a, b, c = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
        den = (a - 2 * b + c)
        if den != 0:
            tau_est += 0.5 * (a - c) / den
    return sr / tau_est

def _parabolic(spec, k):
    """Refine a spectral peak position. Interpolates on LOG magnitude --
    far more accurate for windowed sinusoids than linear interpolation."""
    if k <= 0 or k >= len(spec) - 1:
        return k
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    if a > 0 and b > 0 and c > 0:
        a, b, c = np.log(a), np.log(b), np.log(c)
    den = (a - 2 * b + c)
    return k + 0.5 * (a - c) / den if den != 0 else k

def harmonic_f0(seg, sr, f0_seed, n_max=14):
    """Bass fundamental from its partials (the ETD way: measure what's loud,
    not the weak/inaudible fundamental).

    Seeded with the ROUGH MEASURED pitch, not nominal ET, so a piano 50c+
    flat keeps its partials inside the search windows. Piano partials are
    inharmonic (f_n = n*f1*sqrt(1+B*n^2), sharp of n*f1), so with >=4 partials
    we fit B directly:  (f_n/n)^2 = f1^2 + (f1^2*B)*n^2  is linear in n^2.
    The intercept gives an inharmonicity-corrected fundamental. Falls back to
    an amplitude-weighted median of implied fundamentals when the fit can't
    be trusted."""
    seg = seg.astype(np.float64); seg = seg - np.mean(seg)
    if f0_seed is None or f0_seed <= 0 or np.max(np.abs(seg)) < 1e-6:
        return None
    seg = seg * np.hanning(len(seg))
    nfft = 1 << int(np.ceil(np.log2(len(seg) * 4)))
    spec = np.abs(np.fft.rfft(seg, nfft))
    smax = np.max(spec)
    if smax <= 0:
        return None
    found = []
    for n in range(2, n_max + 1):
        target = n * f0_seed
        if target > sr / 2 - 50:
            break
        # +-45c seed tolerance, plus sharp headroom for inharmonicity
        lo = target * (2 ** (-45 / 1200.0))
        hi = target * (2 ** (45 / 1200.0)) * np.sqrt(1 + 0.001 * n * n)
        klo, khi = int(lo / sr * nfft), int(hi / sr * nfft)
        if khi <= klo or khi >= len(spec):
            continue
        kpk = klo + int(np.argmax(spec[klo:khi]))
        if spec[kpk] < smax * 0.02:
            continue
        fpk = _parabolic(spec, kpk) / nfft * sr
        found.append((n, fpk, float(spec[kpk])))
    if len(found) < 3:
        return None
    ns = np.array([f[0] for f in found], dtype=float)
    fs = np.array([f[1] for f in found], dtype=float)
    am = np.array([f[2] for f in found], dtype=float)
    if len(found) >= 4:
        x = ns * ns
        yv = (fs / ns) ** 2
        w = np.sqrt(am)                      # trust louder partials more
        W = np.sum(w)
        mx = np.sum(w * x) / W
        my = np.sum(w * yv) / W
        var = np.sum(w * (x - mx) ** 2)
        if var > 0:
            slope = np.sum(w * (x - mx) * (yv - my)) / var
            intercept = my - slope * mx
            if intercept > 0:
                B = slope / intercept
                f0 = float(np.sqrt(intercept))
                # sane piano range for B; result must agree with the seed
                if -1e-5 <= B <= 0.004 and \
                   abs(1200.0 * np.log2(f0 / f0_seed)) < 150:
                    return f0
    # Fallback: amplitude-weighted median of f_n/n, favoring low partials
    # (least inharmonic). Slight sharp bias, but robust.
    implied = fs / ns
    w = am / ns
    order = np.argsort(implied)
    cw = np.cumsum(w[order])
    return float(implied[order][np.searchsorted(cw, cw[-1] / 2)])

def fundamental_fft(seg, sr, rough, search_cents=60):
    """Measure the FUNDAMENTAL partial directly as a spectral peak near the
    rough estimate. YIN on an inharmonic piano tone reads a few cents SHARP
    (sharp upper partials pull the periodicity compromise); the fundamental
    peak itself doesn't lie."""
    seg = seg.astype(np.float64); seg = seg - np.mean(seg)
    if rough is None or rough <= 0 or np.max(np.abs(seg)) < 1e-7:
        return None
    seg = seg * np.hanning(len(seg))
    nfft = 1 << int(np.ceil(np.log2(max(len(seg), 1) * 8)))
    spec = np.abs(np.fft.rfft(seg, nfft))
    lo = rough * 2 ** (-search_cents / 1200.0)
    hi = rough * 2 ** (search_cents / 1200.0)
    klo, khi = int(lo / sr * nfft), min(int(hi / sr * nfft), len(spec) - 2)
    if khi <= klo:
        return None
    kpk = klo + int(np.argmax(spec[klo:khi]))
    if spec[kpk] <= 0:
        return None
    return _parabolic(spec, kpk) / nfft * sr

def refined_mid_pitch(seg, sr, rough):
    """A3-A5: fundamental is strong on phone recordings up here -- measure
    that peak directly (no YIN inharmonicity bias, no octave hops since the
    search window is locked to the rough estimate)."""
    f = fundamental_fft(seg, sr, rough)
    return f if (f and np.isfinite(f)) else rough

TREBLE_NOTES = {93, 105}   # A6, A7: YIN octave-errors here; use direct FFT peak
TREBLE_MAX_MIDI = 105        # highest slot; A7

def treble_fft_pitch(seg, sr, f_expected, search_cents=200):
    """For A6/A7 the fundamental is strong in the spectrum but short/quiet, and
    YIN tends to lock onto a sub-octave. Pick the strongest FFT peak within a
    window around the expected fundamental instead -- no octave error."""
    seg = seg.astype(np.float64); seg = seg - np.mean(seg)
    if np.max(np.abs(seg)) < 1e-7:
        return None
    seg = seg * np.hanning(len(seg))
    nfft = 1 << int(np.ceil(np.log2(max(len(seg), 1) * 8)))
    spec = np.abs(np.fft.rfft(seg, nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    lo = f_expected * 2 ** (-search_cents / 1200.0)
    hi = f_expected * 2 ** (search_cents / 1200.0)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return None
    idx = np.where(band)[0]
    kpk = idx[int(np.argmax(spec[idx]))]
    # The peak must actually stand out in the whole spectrum, otherwise any
    # broadband noise blob would "find" a peak in every band it's probed for.
    if spec[kpk] < 0.05 * np.max(spec):
        return None
    return _parabolic(spec, kpk) / nfft * sr

def detect_8_a_notes(audio_path, min_gap_ms=300, preset='standard', a4=440.0):
    sr, y = load_wav_mono(audio_path)
    curve = STRETCH_PRESETS.get(preset, STRETCH_PRESETS[DEFAULT_PRESET])
    nominal = {m: et_freq(m, a4) for m in A_MIDI_ORDER}
    if len(y) == 0:
        return [], {'notes_detected':0,'missing_notes':[A_LABELS[m] for m in A_MIDI_ORDER],'rejected_blobs':0,'avg_cents_from_stretch':None}
    hop = 512
    frame_len = int(sr * 0.09)
    frame_dur = hop / sr
    n_frames = max(0, (len(y) - frame_len) // hop)
    rms = np.array([
        np.sqrt(np.mean(y[i*hop:i*hop+frame_len] ** 2))
        for i in range(n_frames)
    ])
    if len(rms) == 0:
        return [], {'notes_detected':0,'missing_notes':[A_LABELS[m] for m in A_MIDI_ORDER],'rejected_blobs':0,'avg_cents_from_stretch':None}
    thresh = max(rms.max() * 0.08, 1e-4)
    min_gap_frames = int((min_gap_ms / 1000.0) / frame_dur)

    groups, cur = [], []
    for i, e in enumerate(rms):
        if e > thresh:
            cur.append(i)
        else:
            if cur:
                groups.append(cur); cur = []
    if cur:
        groups.append(cur)

    merged = []
    for g in groups:
        if merged and (g[0] - merged[-1][-1]) < min_gap_frames:
            merged[-1].extend(g)
        else:
            merged.append(g)
    merged = [g for g in merged if len(g) >= 2]

    events = []
    for g in merged:
        s = g[0] * hop
        e = min(g[-1] * hop + frame_len, len(y))
        peak_e = float(rms[g].max())
        events.append({'start': s, 'end': e,
                       'time': g[0] * frame_dur,
                       'dur': (g[-1]-g[0]) * frame_dur,
                       'energy': peak_e})
    events.sort(key=lambda x: x['time'])
    if len(events) > 8:
        strong = sorted(events, key=lambda x: x['energy'], reverse=True)[:8]
        strong.sort(key=lambda x: x['time'])
        events = strong

    max_e = max((e['energy'] for e in events), default=1.0) or 1.0

    # --- Measure each blob's rough pitch (no labels yet) ---
    # Skip the attack transient, then analyze a long sustain window. Blobs
    # where YIN finds no period are KEPT (rough=None): a quiet A6/A7 can fail
    # YIN yet still show a clean spectral peak for the treble FFT probe.
    blobs = []  # {rough, energy, time, dur, seg}
    for ev in events:
        seg_full = y[ev['start']:ev['end']]
        L = len(seg_full)
        skip = min(int(sr * ATTACK_SKIP_S), L // 3)
        seg = seg_full[skip: skip + int(sr * ANALYSIS_WIN_S)]
        if len(seg) < 256:
            seg = seg_full
        rough = yin_pitch(seg, sr, 20, 3800)
        if rough is not None and (not np.isfinite(rough) or rough <= 0):
            rough = None
        blobs.append({'rough': (float(rough) if rough else None),
                      'energy': ev['energy'],
                      'time': ev['time'], 'dur': ev['dur'], 'seg': seg})

    # --- Order-aware assignment ---------------------------------------------
    # Clients play the A's low->high in order. We exploit that: walk the blobs
    # in time order and the A-slots in pitch order together. A blob is accepted
    # into the current A-slot if it is plausibly that note -- allowing it to be
    # VERY flat (or sharp) -- WITHOUT requiring it to be the literal nearest A.
    # This lets a note sit 150c+ flat and still be identified, while still
    # rejecting true garbage (sounds that fit no upcoming slot).
    #
    # ACCEPT_CENTS: how far from a slot's ET pitch a blob may sit and still be
    # that note. Wide enough for badly neglected pianos, but bounded so a random
    # noise blob doesn't masquerade as a note. Because matching is sequential and
    # monotonic, a flat note can't be mis-snapped to its lower neighbor.
    ACCEPT_CENTS = 230.0

    blobs.sort(key=lambda b: b['time'])
    assignments = {}
    rejected = 0
    slot = 0  # index into A_MIDI_ORDER
    for b in blobs:
        placed = False
        for s in range(slot, len(A_MIDI_ORDER)):
            midi = A_MIDI_ORDER[s]
            et = et_freq(midi, a4)

            if midi in TREBLE_NOTES:
                # YIN is unreliable up here (it can read an A7 as ~20 Hz). Don't
                # trust rough at all: probe the spectrum directly for a strong
                # peak near this slot's expected pitch. If found, that IS the
                # measurement and the seating decision in one step.
                f_fft = treble_fft_pitch(b['seg'], sr, nominal[midi],
                                         search_cents=ACCEPT_CENTS)
                if f_fft is None:
                    # no peak in this slot's band; let a later treble slot try,
                    # but only if the rough pitch isn't clearly a lower note.
                    rough = b['rough']
                    if rough and 1200.0 * np.log2(rough / et) < -ACCEPT_CENTS \
                       and midi != TREBLE_MAX_MIDI:
                        continue
                    if rough is None and midi != TREBLE_MAX_MIDI:
                        continue
                    break
                c = 1200.0 * np.log2(f_fft / et)
                if abs(c) <= ACCEPT_CENTS:
                    assignments[midi] = {'freq': float(f_fft), 'energy': b['energy'],
                                         'time': b['time'], 'dur': b['dur']}
                    slot = s + 1; placed = True; break
                continue

            # Non-treble slots need a rough pitch to seat; a no-period blob
            # can only ever be a treble note (or garbage).
            rough = b['rough']
            if rough is None:
                continue
            c = 1200.0 * np.log2(rough / et)
            if c > ACCEPT_CENTS:
                continue
            if abs(c) <= ACCEPT_CENTS:
                if midi in HARMONIC_NOTES:
                    # Measure the partials (loud) instead of the fundamental
                    # (weak/rolled off on phone mics). Seed with the rough
                    # MEASURED pitch so flat pianos stay in-window.
                    f0 = harmonic_f0(b['seg'], sr, rough) or rough
                else:
                    f0 = refined_mid_pitch(b['seg'], sr, rough)
                assignments[midi] = {'freq': float(f0), 'energy': b['energy'],
                                     'time': b['time'], 'dur': b['dur']}
                slot = s + 1; placed = True; break
            break
        if not placed:
            rejected += 1

    # --- Build results in fixed A0..A7 order; gaps stay gaps ---
    results = []
    for idx, midi in enumerate(A_MIDI_ORDER):
        if midi not in assignments:
            continue
        m = assignments[midi]
        f0 = m['freq']
        label = A_LABELS[midi]
        rb = curve[midi]
        et_f = et_freq(midi, a4)
        st_f = stretched_target(midi, rb, a4)
        results.append({
            'index': idx + 1,
            'midi': midi,
            'core': midi in CORE_WEIGHTS,
            'note_label': label,
            'freq_measured': round(f0, 3),
            'freq_et': round(et_f, 3),
            'freq_stretched': round(st_f, 3),
            'railsback_offset': rb,
            'cents_from_et': round(cents_from_target(f0, et_f), 1),
            'cents_from_stretch': round(cents_from_target(f0, st_f), 1),
            'confidence': round(min(m['energy'] / max_e, 1.0), 3),
            'time': round(m['time'], 3),
            'duration': round(m['dur'], 3),
        })

    missing = [A_LABELS[m] for m in CORE_WEIGHTS if m not in assignments]

    # Overall average deviation (vs stretched target), all detected notes
    devs = [r['cents_from_stretch'] for r in results if r['cents_from_stretch'] is not None]
    avg_dev = round(float(np.mean(devs)), 1) if devs else None

    # Traffic-light verdict from the A2-A6 core, weighted
    core_pts = [(CORE_WEIGHTS[r['midi']], r['cents_from_stretch'])
                for r in results
                if r['core'] and r['cents_from_stretch'] is not None]
    if len(core_pts) >= 3:
        wsum = sum(w for w, _ in core_pts)
        wavg = round(float(sum(w * c for w, c in core_pts) / wsum), 1)
        if abs(wavg) <= VERDICT_GREEN:
            # On the right pitch level: no pitch raise needed, but the piano
            # still needs a regular tuning. Deliberately yellow, never green.
            verdict, vlabel = 'yellow', 'No pitch raise needed — regular tuning recommended'
        elif abs(wavg) <= VERDICT_RED:
            verdict, vlabel = 'yellow', 'Tuning recommended'
        elif wavg < 0:
            verdict, vlabel = 'red', 'Pitch raise recommended'
        else:
            verdict, vlabel = 'red', 'Pitch correction recommended'
    else:
        wavg, verdict, vlabel = None, 'none', 'Not enough core notes for a verdict'

    summary = {
        'notes_detected': len(results),
        'core_notes_detected': len(core_pts),
        'missing_notes': missing,
        'rejected_blobs': rejected,
        'avg_cents_from_stretch': avg_dev,
        'weighted_avg_cents': wavg,
        'verdict': verdict,
        'verdict_label': vlabel,
    }
    return results, summary

@app.route('/')
def index():
    return send_from_directory('Static', 'index.html')

@app.route('/sw.js')
def sw():
    r = send_from_directory('Static', 'sw.js', mimetype='application/javascript')
    r.headers['Cache-Control'] = 'no-cache'
    return r

@app.route('/manifest.json')
def manifest():
    return send_from_directory('Static', 'manifest.json',
                               mimetype='application/manifest+json')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    min_gap = int(request.form.get('minGap', 300))
    preset = request.form.get('preset', DEFAULT_PRESET)
    preset = PRESET_ALIASES.get(preset, preset)
    if preset not in STRETCH_PRESETS:
        preset = DEFAULT_PRESET
    try:
        a4 = float(request.form.get('a4', 440.0))
    except (TypeError, ValueError):
        a4 = 440.0
    if not (400.0 <= a4 <= 480.0):
        a4 = 440.0
    ext = os.path.splitext(file.filename)[1] if file.filename else '.tmp'
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    wav_path = None
    try:
        wav_path = extract_audio(tmp_path)
        notes, summary = detect_8_a_notes(wav_path, min_gap_ms=min_gap,
                                          preset=preset, a4=a4)
        summary['preset'] = preset
        summary['a4'] = a4
        return jsonify({'notes': notes, 'count': len(notes), 'summary': summary})
    except subprocess.CalledProcessError:
        return jsonify({'error': 'Could not read audio from that file'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
