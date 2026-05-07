import os
import json
import uuid
import threading
import subprocess
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

WORK_DIR = '/tmp/barber_sessions'
os.makedirs(WORK_DIR, exist_ok=True)

jobs = {}


def get_session_dir(session_id):
    d = os.path.join(WORK_DIR, session_id)
    os.makedirs(d, exist_ok=True)
    return d


def get_video_info(path):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
           '-show_streams', '-show_format', path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    width = height = None
    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video':
            width = stream.get('width')
            height = stream.get('height')
            break
    duration = float(data.get('format', {}).get('duration', 0))
    return width, height, duration


def analyze_reference(ref_path):
    """Detect scene cuts in reference video using PySceneDetect, fall back to CV2."""
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(ref_path, ContentDetector(threshold=27.0, min_scene_len=10))
        if scenes and len(scenes) >= 2:
            durations = [(end.get_seconds() - start.get_seconds()) for start, end in scenes]
            avg = sum(durations) / len(durations)
            return {
                'avg_clip_duration': max(0.4, min(avg, 5.0)),
                'num_cuts': len(scenes)
            }
    except Exception:
        pass

    # CV2 fallback: histogram-based cut detection
    try:
        import cv2
        import numpy as np
        cap = cv2.VideoCapture(ref_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        cut_times = [0.0]
        prev_hist = None
        frame_idx = 0
        min_gap = int(fps * 0.4)
        last_cut = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            small = cv2.resize(frame, (160, 90))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None and (frame_idx - last_cut) >= min_gap:
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if corr < 0.45:
                    cut_times.append(frame_idx / fps)
                    last_cut = frame_idx
            prev_hist = hist
            frame_idx += 1
        cap.release()
        cut_times.append(duration)

        if len(cut_times) >= 3:
            clip_durs = [cut_times[i + 1] - cut_times[i] for i in range(len(cut_times) - 1)]
            avg = sum(clip_durs) / len(clip_durs)
            return {
                'avg_clip_duration': max(0.4, min(avg, 5.0)),
                'num_cuts': len(clip_durs)
            }
    except Exception:
        pass

    return {'avg_clip_duration': 2.0, 'num_cuts': 8}


def build_crop_filter(w, h):
    """Return FFmpeg vf filter string to produce 9:16 at 1080x1920, or None."""
    if not w or not h:
        return 'scale=1080:1920'
    current = w / h
    target = 9 / 16  # 0.5625
    if abs(current - target) < 0.04:
        return 'scale=1080:1920'
    if current > target:
        # Too wide — crop left/right
        new_w = int(h * target)
        x_off = (w - new_w) // 2
        return f'crop={new_w}:{h}:{x_off}:0,scale=1080:1920'
    # Too tall / narrow — pad sides with black
    return 'scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1'


def process_video(session_id, ref_path, raw_path, output_path):
    try:
        jobs[session_id].update(status='analyzing', message='Analyzing reference video pacing...')

        ref = analyze_reference(ref_path)
        avg_clip = ref['avg_clip_duration']
        num_ref_clips = ref['num_cuts']

        jobs[session_id]['message'] = (
            f'Reference: {num_ref_clips} cuts detected, avg {avg_clip:.1f}s per clip'
        )

        raw_w, raw_h, raw_dur = get_video_info(raw_path)
        if not raw_dur or raw_dur < 5:
            raise ValueError('Raw video is too short (minimum 5 seconds required)')

        # Target: ≤19.5s total, honour reference pacing
        max_total = 19.5
        num_clips = min(num_ref_clips, int(max_total / avg_clip))
        num_clips = max(num_clips, 4)
        clip_dur = avg_clip
        if num_clips * clip_dur > max_total:
            clip_dur = max_total / num_clips

        # Distribute clip start times evenly across usable portion of raw footage
        usable_start = raw_dur * 0.05
        usable_end = min(raw_dur * 0.92, raw_dur - clip_dur)
        usable_range = usable_end - usable_start

        if usable_range < clip_dur:
            raise ValueError('Raw video is too short to extract enough clips')

        # Reduce clip count if footage range is narrow
        while num_clips * (usable_range / num_clips) < clip_dur and num_clips > 4:
            num_clips -= 1

        spacing = usable_range / num_clips
        clip_starts = [
            usable_start + i * spacing
            for i in range(num_clips)
            if usable_start + i * spacing + clip_dur <= raw_dur
        ]

        jobs[session_id].update(status='processing', message=f'Cutting {len(clip_starts)} clips...')

        vf = build_crop_filter(raw_w, raw_h)
        vf_args = ['-vf', vf] if vf else []

        session_dir = get_session_dir(session_id)
        clip_files = []

        for i, start in enumerate(clip_starts):
            clip_path = os.path.join(session_dir, f'clip_{i:03d}.mp4')
            cmd = [
                'ffmpeg', '-y',
                '-ss', f'{start:.3f}',
                '-i', raw_path,
                '-t', f'{clip_dur:.3f}',
                *vf_args,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
                '-r', '30',
                '-an',
                clip_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=180)
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                clip_files.append(clip_path)
            jobs[session_id]['message'] = f'Cut {i + 1}/{len(clip_starts)} clips...'

        if not clip_files:
            raise ValueError('Failed to extract any clips — check that ffmpeg can read your video format')

        concat_path = os.path.join(session_dir, 'concat.txt')
        with open(concat_path, 'w') as f:
            for cp in clip_files:
                f.write(f"file '{cp}'\n")

        jobs[session_id]['message'] = 'Joining clips...'

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', concat_path,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
            '-movflags', '+faststart',
            '-an',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0 or not os.path.exists(output_path):
            raise ValueError(f'FFmpeg concat failed: {result.stderr[-300:]}')

        _, _, out_dur = get_video_info(output_path)

        for cp in clip_files:
            try:
                os.remove(cp)
            except OSError:
                pass
        try:
            os.remove(concat_path)
        except OSError:
            pass

        jobs[session_id].update(
            status='done',
            message=f'Done — {out_dur:.1f}s edited video ready',
            output_path=output_path,
            output_duration=round(out_dur, 1),
            num_clips=len(clip_files),
            avg_clip=round(clip_dur, 2)
        )

    except Exception as e:
        jobs[session_id].update(status='error', message=str(e))


@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(__file__), 'barber-editor.html'))


@app.route('/upload', methods=['POST'])
def upload():
    if 'reference' not in request.files or 'raw' not in request.files:
        return jsonify({'error': 'Both reference and raw videos are required'}), 400

    ref_file = request.files['reference']
    raw_file = request.files['raw']

    session_id = str(uuid.uuid4())
    session_dir = get_session_dir(session_id)

    ref_ext = os.path.splitext(ref_file.filename)[1] or '.mp4'
    raw_ext = os.path.splitext(raw_file.filename)[1] or '.mp4'
    ref_path = os.path.join(session_dir, f'reference{ref_ext}')
    raw_path = os.path.join(session_dir, f'raw{raw_ext}')

    ref_file.save(ref_path)
    raw_file.save(raw_path)

    return jsonify({'session_id': session_id})


@app.route('/process/<session_id>', methods=['POST'])
def process(session_id):
    session_dir = get_session_dir(session_id)
    ref_files = [f for f in os.listdir(session_dir) if f.startswith('reference')]
    raw_files = [f for f in os.listdir(session_dir) if f.startswith('raw')]

    if not ref_files or not raw_files:
        return jsonify({'error': 'Videos not found — upload first'}), 400

    if session_id in jobs and jobs[session_id].get('status') in ('analyzing', 'processing'):
        return jsonify({'error': 'Already processing'}), 400

    ref_path = os.path.join(session_dir, ref_files[0])
    raw_path = os.path.join(session_dir, raw_files[0])
    output_path = os.path.join(session_dir, 'output.mp4')

    jobs[session_id] = {'status': 'queued', 'message': 'Starting...'}

    t = threading.Thread(
        target=process_video,
        args=(session_id, ref_path, raw_path, output_path),
        daemon=True
    )
    t.start()
    return jsonify({'status': 'started'})


@app.route('/status/<session_id>')
def status(session_id):
    if session_id not in jobs:
        return jsonify({'status': 'unknown', 'message': 'Session not found'})
    return jsonify(jobs[session_id])


@app.route('/download/<session_id>')
def download(session_id):
    if session_id not in jobs or jobs[session_id].get('status') != 'done':
        return jsonify({'error': 'Not ready'}), 400
    out = jobs[session_id].get('output_path', '')
    if not os.path.exists(out):
        return jsonify({'error': 'Output file missing'}), 404
    return send_file(out, as_attachment=True, download_name='stebs_edit.mp4', mimetype='video/mp4')


if __name__ == '__main__':
    print('Barber Video Editor running at http://localhost:5555')
    app.run(host='0.0.0.0', port=5555, debug=False, threaded=True)
