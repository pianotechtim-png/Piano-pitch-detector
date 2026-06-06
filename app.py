from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import librosa
import tempfile
import os

app = Flask(__name__, static_folder='Static')
CORS(app)

RAILSBACK_A = {
    21: ('A0', -16.0),
    33: ('A1', -10.0),
    45: ('A2',  -5.0),
    57: ('A3',  -1.5),
    69: ('A4',   0.0),
    81: ('A5',  +3.0),
    93: ('A6',  +8.0),
   105: ('A7', +16.0),
}

def et_freq(midi):
    return 440.0 * (2 ** ((midi - 69) / 12.0))

def stretched_target(midi):
    rb_cents = RAILSBACK_A[midi][1]
    return et_freq(midi) * (2 ** (rb_cents / 1200.0))

def cents_from_target(measured_freq, target_freq):
    if measured_freq <= 0 or target_freq <= 0:
        return None
    return 1200.0 * np.log2(measured_freq / target_freq)

def detect_8_a_notes(audio_path, min_gap_ms=300):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    hop_length = 256
    frame_duration = hop_length / sr

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('G#0'),
        fmax=librosa.note_to_hz('A#7'),
        hop_length=hop_length,
        sr=sr,
        fill_na=None
    )

    min_gap_frames = int((min_gap_ms / 1000.0) / frame_duration)

    groups = []
    current = []
    for i, (freq, voiced, prob) in enumerate(zip(f0, voiced_flag, voiced_prob)):
        if voiced and prob > 0.5 and freq is not None and np.isfinite(freq):
            current.append({'freq': float(freq), 'prob': float(prob), 'frame': i})
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)

    merged = []
    for g in groups:
        if merged and (g[0]['frame'] - merged[-1][-1]['frame']) < min_gap_frames:
            merged[-1].extend(g)
        else:
            merged.append(g)

    merged = [g for g in merged if len(g) >= 4]

    note_events = []
    for group in merged:
        freqs = np.array([f['freq'] for f in group])
        probs = np.array([f['prob'] for f in group])
        sorted_idx = np.argsort(freqs)
        freqs_s = freqs[sorted_idx]
        probs_s = probs[sorted_idx]
        cumw = np.cumsum(probs_s)
        median_freq = float(freqs_s[np.searchsorted(cumw, cumw[-1] / 2.0)])
        avg_conf = float(np.mean(probs))
        start_time = group[0]['frame'] * frame_duration
        end_time   = group[-1]['frame'] * frame_duration
        s = group[0]['frame'] * hop_length
        e = min(group[-1]['frame'] * hop_length + hop_length, len(y))
        rms = float(np.sqrt(np.mean(y[s:e] ** 2)))
        note_events.append({
            'freq': median_freq,
            'confidence': avg_conf,
            'time': start_time,
            'duration': end_time - start_time,
            'rms': rms
        })

    note_events.sort(key=lambda n: n['time'])

    if len(note_events) > 8:
        scored = sorted(note_events, key=lambda n: n['rms'] * n['confidence'], reverse=True)
        top8 = scored[:8]
        top8.sort(key=lambda n: n['time'])
        note_events = top8

    a_midi_order = [21, 33, 45, 57, 69, 81, 93, 105]
    results = []

    for i, note in enumerate(note_events):
        target_midi = a_midi_order[i] if i < len(a_midi_order) else None
        if target_midi is None:
            continue
        measured = note['freq']
        label, rb_offset = RAILSBACK_A[target_midi]
        et_f      = et_freq(target_midi)
        stretch_f = stretched_target(target_midi)
        cents_from_et      = cents_from_target(measured, et_f)
        cents_from_stretch = cents_from_target(measured, stretch_f)
        results.append({
            'index':              i + 1,
            'note_label':         label,
            'freq_measured':      round(measured, 3),
            'freq_et':            round(et_f, 3),
            'freq_stretched':     round(stretch_f, 3),
            'railsback_offset':   rb_offset,
            'cents_from_et':      round(cents_from_et, 1) if cents_from_et else None,
            'cents_from_stretch': round(cents_from_stretch, 1) if cents_from_stretch else None,
            'confidence':         round(note['confidence'], 3),
            'time':               round(note['time'], 3),
            'duration':           round(note['duration'], 3),
        })

    return results, len(note_events)


@app.route('/')
def index():
    return send_from_directory('Static', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file    = request.files['file']
    min_gap = int(request.form.get('minGap', 300))
    ext = os.path.splitext(file.filename)[1] if file.filename else '.tmp'
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        notes, found = detect_8_a_notes(tmp_path, min_gap_ms=min_gap)
        return jsonify({'notes': notes, 'count': len(notes), 'groups_found': found})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
