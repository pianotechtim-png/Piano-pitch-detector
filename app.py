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

# Three stretch presets (cents from equal temperament), keyed by piano size.
STRETCH_PRESETS = {
    'spinet':  {21:-40.0, 33:-24.0, 45:-11.0, 57:-3.0, 69:0.0, 81:9.0,  93:20.0, 105:40.0},
    'average': {21:-32.0, 33:-18.0, 45:-8.0,  57:-2.0, 69:0.0, 81:6.0,  93:14.0, 105:28.0},
    'grand':   {21:-22.0, 33:-12.0, 45:-5.0,  57:-1.0, 69:0.0, 81:4.0,  93:9.0,  105:18.0},
}
DEFAULT_PRESET = 'average'

# Backwards-compat: RAILSBACK_A holds (label, offset) for the selected curve.
# Built per-request now; this default keeps module-level helpers working.
RAILSBACK_A = {m: (A_LABELS[m], STRETCH_PRESETS[DEFAULT_PRESET][m]) for m in A_LABELS}
A_MIDI_ORDER = [21, 33, 45, 57, 69, 81, 93, 105]
NOMINAL_A = {21:27.5, 33:55.0, 45:110.0, 57:220.0, 69:440.0,
             81:880.0, 93:1760.0, 105:3520.0}
HARMONIC_NOTES = {21, 33, 45}   # use harmonic method (weak fundamental)

def nearest_a(freq):
    best, best_c = None, None
    for midi, nom in NOMINAL_A.items():
        c = 1200.0 * np.log2(freq / nom)
        if best is None or abs(c) < abs(best_c):
            best, best_c = midi, c
    return best, best_c

def et_freq(midi):
    return 440.0 * (2 ** ((midi - 69) / 12.0))

def stretched_target(midi, offset_cents):
    return et_freq(midi) * (2 ** (offset_cents / 1200.0))

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
    if k <= 0 or k >= len(spec) - 1:
        return k
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    den = (a - 2 * b + c)
    return k + 0.5 * (a - c) / den if den != 0 else k

def harmonic_f0(seg, sr, f0_expected, n_search=12):
    seg = seg.astype(np.float64); seg = seg - np.mean(seg)
    if np.max(np.abs(seg)) < 1e-6:
        return None
    seg = seg * np.hanning(len(seg))
    nfft = 1 << int(np.ceil(np.log2(len(seg) * 4)))
    spec = np.abs(np.fft.rfft(seg, nfft))
    smax = np.max(spec)
    if smax <= 0:
        return None
    found = []
    for n in range(2, n_search + 1):
        target = n * f0_expected
        if target > sr / 2 - 50:
            break
        lo, hi = target * 0.97, target * 1.06
        klo, khi = int(lo / sr * nfft), int(hi / sr * nfft)
        if khi <= klo or khi >= len(spec):
            continue
        kpk = klo + int(np.argmax(spec[klo:khi]))
        if spec[kpk] < smax * 0.02:
            continue
        fpk = _parabolic(spec, kpk) / nfft * sr
        found.append((n, fpk))
    if len(found) < 3:
        return None
    ns = np.array([f[0] for f in found], dtype=float)
    fs = np.array([f[1] for f in found], dtype=float)
    implied = fs / ns
    w = 1.0 / ns
    order = np.argsort(implied)
    implied_s, w_s = implied[order], w[order]
    cw = np.cumsum(w_s)
    return float(implied_s[np.searchsorted(cw, cw[-1] / 2)])

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
    if 0 < kpk < len(spec) - 1:
        a, b, c = spec[kpk - 1], spec[kpk], spec[kpk + 1]
        den = (a - 2 * b + c)
        kf = kpk + 0.5 * (a - c) / den if den != 0 else kpk
    else:
        kf = kpk
    return kf / nfft * sr

def detect_8_a_notes(audio_path, min_gap_ms=300, preset='average'):
    sr, y = load_wav_mono(audio_path)
    curve = STRETCH_PRESETS.get(preset, STRETCH_PRESETS[DEFAULT_PRESET])
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
    blobs = []  # {rough, energy, time, dur, seg}
    for ev in events:
        seg_full = y[ev['start']:ev['end']]
        L = len(seg_full)
        seg = seg_full[L // 4: L // 4 + min(int(sr * 0.30), L - L // 4)]
        if len(seg) < 256:
            seg = seg_full
        rough = yin_pitch(seg, sr, 20, 3800)
        if rough is None or not np.isfinite(rough) or rough <= 0:
            continue
        blobs.append({'rough': float(rough), 'energy': ev['energy'],
                      'time': ev['time'], 'dur': ev['dur'], 'seg': seg})

    # --- Order-aware assignment ---------------------------------------------
    # Clients play A0..A7 low->high in order. We exploit that: walk the blobs in
    # time order and the A-slots in pitch order together. A blob is accepted into
    # the current A-slot if it is plausibly that note -- allowing it to be VERY
    # flat (or sharp) -- WITHOUT requiring it to be the literal nearest A. This
    # lets a note sit 150c+ flat and still be identified, while still rejecting
    # true garbage (sounds that fit no upcoming slot within a wide tolerance).
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
            et = et_freq(midi)

            if midi in TREBLE_NOTES:
                # YIN is unreliable up here (it can read an A7 as ~20 Hz). Don't
                # trust rough at all: probe the spectrum directly for a strong
                # peak near this slot's expected pitch. If found, that IS the
                # measurement and the seating decision in one step.
                f_fft = treble_fft_pitch(b['seg'], sr, NOMINAL_A[midi],
                                         search_cents=ACCEPT_CENTS)
                if f_fft is None:
                    # no peak in this slot's band; let a later treble slot try,
                    # but only if the rough pitch isn't clearly a lower note.
                    rough = b['rough']
                    if rough > 0 and 1200.0 * np.log2(rough / et) < -ACCEPT_CENTS \
                       and midi != TREBLE_MAX_MIDI:
                        continue
                    break
                c = 1200.0 * np.log2(f_fft / et)
                if abs(c) <= ACCEPT_CENTS:
                    assignments[midi] = {'freq': float(f_fft), 'energy': b['energy'],
                                         'time': b['time'], 'dur': b['dur']}
                    slot = s + 1; placed = True; break
                continue

            # Non-treble: use rough YIN to seat, refine bass via harmonics.
            rough = b['rough']
            c = 1200.0 * np.log2(rough / et)
            if c > ACCEPT_CENTS:
                continue
            if abs(c) <= ACCEPT_CENTS:
                if midi in HARMONIC_NOTES:
                    f0 = harmonic_f0(b['seg'], sr, NOMINAL_A[midi]) or rough
                else:
                    f0 = rough
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
        et_f = et_freq(midi)
        st_f = stretched_target(midi, rb)
        results.append({
            'index': idx + 1,
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

    missing = [A_LABELS[m] for m in A_MIDI_ORDER if m not in assignments]

    # Overall average deviation (vs stretched target) -> pricing signal
    devs = [r['cents_from_stretch'] for r in results if r['cents_from_stretch'] is not None]
    avg_dev = round(float(np.mean(devs)), 1) if devs else None

    summary = {
        'notes_detected': len(results),
        'missing_notes': missing,
        'rejected_blobs': rejected,
        'avg_cents_from_stretch': avg_dev,
    }
    return results, summary

@app.route('/')
def index():
    return send_from_directory('Static', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    min_gap = int(request.form.get('minGap', 300))
    preset = request.form.get('preset', 'average')
    if preset not in STRETCH_PRESETS:
        preset = 'average'
    ext = os.path.splitext(file.filename)[1] if file.filename else '.tmp'
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    wav_path = None
    try:
        wav_path = extract_audio(tmp_path)
        notes, summary = detect_8_a_notes(wav_path, min_gap_ms=min_gap, preset=preset)
        summary['preset'] = preset
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
