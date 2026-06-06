from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import librosa
import tempfile
import os
import subprocess

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

def extract_audio(src_path):
    wav_path = src_path + '.wav'
    subprocess.run(
        ['ffmpeg', '-y', '-i', src_path,
         '-ac', '1', '-ar', '16000', '-vn',
         '-f', 'wav', wav_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return wav_path

def detect_8_a_notes(audio_path, min_gap_ms=300):
    sr = 16000
    y, sr = librosa.load(audio_path, sr=sr, mono=True, dtype=np.float32)

    hop_length = 512
    frame_length = 2048
    frame_duration = hop_length / sr

    # YIN: low-memory fundamental frequency detection (no probability matrix)
    f0 = librosa.yin(
        y,
        fmin=25.0,
        fmax=3600.0,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    ).astype(np.float32)

    # Per-frame RMS energy, used as a voicing gate + confidence proxy
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0].astype(np.float32)

    n = min(len(f0), len(rms))
    f0 = f0[:n]
    rms = rms[:n]

    # Energy threshold: ignore quiet/silent frames
    if rms.max() > 0:
        thresh = max(rms.max() * 0.12, 1e-4)
    else:
        thresh = 1e-4

    min_gap_frames = int((min_gap_ms / 1000.0) / frame_duration)

    groups = []
    current = []
    for i in range(n):
        freq = float(f0[i])
        energy = float(rms[i])
        if energy > thresh and np.isfinite(freq) and 25.0 < freq < 3600.0:
            current.append({'freq': freq, 'energy': energy, 'frame': i})
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
        freqs   = np.array([f['freq'] for f in group])
        weights = np.array([f['energy'] for f in group])
        # energy-weighted median frequency (robust to attack/decay wobble)
        sorted_idx = np.argsort(freqs)
        freqs_s = freqs[sorted_idx]
        w_s     = weights[sorted_idx]
        cumw    = np.cumsum(w_s)
        median_freq = float(freqs_s[np.searchsorted(cumw, cumw[-1] / 2.0)])
        peak_energy = float(weights.max())
        start_time  = group[0]['frame'] * frame_duration
        end_time    = group[-1]['frame'] * frame_duration
        note_events.append({
            'freq': median_freq,
            'energy': peak_energy,
            'time': start_time,
            'duration': end_time - start_time
        })

    note_events.sort(key=lambda x: x['time'])

    # If more than 8 candidates, keep the 8 strongest, then restore time order
    if len(note_events) > 8:
        scored = sorted(note_events, key=lambda x: x['energy'], reverse=True)
        top8 = scored[:8]
        top8.sort(key=lambda x: x['time'])
        note_events = top8

    # Normalize energy to a 0-1 confidence for display
    if note_events:
        max_e = max(e['energy'] for e in note_events) or 1.0
    else:
        max_e = 1.0

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
        conf = round(min(note['energy'] / max_e, 1.0), 3)

        results.append({
            'index':              i + 1,
            'note_label':         label,
            'freq_measured':      round(measured, 3),
            'freq_et':            round(et_f, 3),
            'freq_stretched':     round(stretch_f, 3),
            'railsback_offset':   rb_offset,
            'cents_from_et':      round(cents_from_et, 1) if cents_from_et is not None else None,
            'cents_from_stretch': round(cents_from_stretch, 1) if cents_from_stretch is not None else None,
            'confidence':         conf,
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
