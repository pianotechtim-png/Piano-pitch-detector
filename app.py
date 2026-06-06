from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from scipy.io import wavfile
import tempfile
import os
import subprocess

app = Flask(__name__, static_folder='Static')
CORS(app)

RAILSBACK_A = {
    21: ('A0', -16.0), 33: ('A1', -10.0), 45: ('A2', -5.0),
    57: ('A3', -1.5),  69: ('A4', 0.0),   81: ('A5', 3.0),
    93: ('A6', 8.0),  105: ('A7', 16.0),
}
A_MIDI_ORDER = [21, 33, 45, 57, 69, 81, 93, 105]
NOMINAL_A = {21:27.5, 33:55.0, 45:110.0, 57:220.0, 69:440.0,
             81:880.0, 93:1760.0, 105:3520.0}
HARMONIC_NOTES = {21, 33, 45}

def et_freq(midi):
    return 440.0 * (2 ** ((midi - 69) / 12.0))

def stretched_target(midi):
    return et_freq(midi) * (2 ** (RAILSBACK_A[midi][1] / 1200.0))

def cents_from_target(measured, target):
    if measured <= 0 or target <= 0:
        return None
    return 1200.0 * np.log2(measured / target)

def extract_audio(src_path):
    wav_path = src_path + '.wav'
    subprocess.run(
        ['ffmpeg', '-y', '-i', src_path, '-ac', '1', '-ar', '44100',
         '-vn', '-f', 'wav', wav_path],
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

def detect_8_a_notes(audio_path, min_gap_ms=300):
    sr, y = load_wav_mono(audio_path)
    if len(y) == 0:
        return [], 0
    hop = 512
    frame_len = int(sr * 0.09)
    frame_dur = hop / sr
    n_frames = max(0, (len(y) - frame_len) // hop)
    rms = np.array([
        np.sqrt(np.mean(y[i*hop:i*hop+frame_len] ** 2))
        for i in range(n_frames)
    ])
    if len(rms) == 0:
        return [], 0
    thresh = max(rms.max() * 0.12, 1e-4)
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
    merged = [g for g in merged if len(g) >= 3]

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
    results = []
    for i, ev in enumerate(events):
        if i >= len(A_MIDI_ORDER):
            break
        midi = A_MIDI_ORDER[i]
        seg_full = y[ev['start']:ev['end']]
        L = len(seg_full)
        seg = seg_full[L // 4: L // 4 + min(int(sr * 0.30), L - L // 4)]
        if len(seg) < 256:
            seg = seg_full
        if midi in HARMONIC_NOTES:
            f0 = harmonic_f0(seg, sr, NOMINAL_A[midi])
            if f0 is None:
                f0 = yin_pitch(seg, sr, 24, 200)
        else:
            lo = NOMINAL_A[midi] * 0.85
            hi = NOMINAL_A[midi] * 1.18
            f0 = yin_pitch(seg, sr, lo, hi)
        if not f0 or not np.isfinite(f0):
            continue
        label, rb = RAILSBACK_A[midi]
        et_f = et_freq(midi)
        st_f = stretched_target(midi)
        results.append({
            'index': i + 1,
            'note_label': label,
            'freq_measured': round(f0, 3),
            'freq_et': round(et_f, 3),
            'freq_stretched': round(st_f, 3),
            'railsback_offset': rb,
            'cents_from_et': round(cents_from_target(f0, et_f), 1),
            'cents_from_stretch': round(cents_from_target(f0, st_f), 1),
            'confidence': round(min(ev['energy'] / max_e, 1.0), 3),
            'time': round(ev['time'], 3),
            'duration': round(ev['dur'], 3),
        })
    return results, len(events)

@app.route('/')
def index():
    return send_from_directory('Static', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    min_gap = int(request.form.get('minGap', 300))
    ext = os.path.splitext(file.filename)[1] if file.filename else '.tmp'
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    wav_path = None
    try:
        wav_path = extract_audio(tmp_path)
        notes, found = detect_8_a_notes(wav_path, min_gap_ms=min_gap)
        return jsonify({'notes': notes, 'count': len(notes), 'groups_found': found})
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
