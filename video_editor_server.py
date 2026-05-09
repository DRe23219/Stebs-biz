import os
import json
import uuid
import threading
import subprocess
import tempfile
import shutil
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

WORK_DIR = '/tmp/barber_sessions'
os.makedirs(WORK_DIR, exist_ok=True)

jobs = {}


# ---------------------------------------------------------------------------
# Core video utilities
# ---------------------------------------------------------------------------

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
    """Detect scene cuts via PySceneDetect, fall back to CV2 histogram diff."""
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(ref_path, ContentDetector(threshold=27.0, min_scene_len=10))
        if scenes and len(scenes) >= 2:
            durations = [(end.get_seconds() - start.get_seconds()) for start, end in scenes]
            avg = sum(durations) / len(durations)
            return {'avg_clip_duration': max(0.4, min(avg, 5.0)), 'num_cuts': len(scenes)}
    except Exception:
        pass

    try:
        import cv2
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
            return {'avg_clip_duration': max(0.4, min(avg, 5.0)), 'num_cuts': len(clip_durs)}
    except Exception:
        pass

    return {'avg_clip_duration': 2.0, 'num_cuts': 8}


def build_crop_filter(w, h):
    if not w or not h:
        return 'scale=1080:1920'
    current = w / h
    target = 9 / 16
    if abs(current - target) < 0.04:
        return 'scale=1080:1920'
    if current > target:
        new_w = int(h * target)
        x_off = (w - new_w) // 2
        return f'crop={new_w}:{h}:{x_off}:0,scale=1080:1920'
    return 'scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1'


# ---------------------------------------------------------------------------
# Smart clip scoring — fully self-contained via CV2
# ---------------------------------------------------------------------------

def score_segments(video_path, num_segments, clip_dur, prefs):
    """
    Score candidate segments of the raw footage using CV2.

    Scoring combines three signals:
    - Motion:     frame-to-frame pixel difference (active cutting = high motion)
    - Sharpness:  Laplacian variance  (blurry clips score lower)
    - Brightness: penalises over/under-exposed frames

    Returns a list of (start_time, score) sorted by score descending.
    """
    try:
        import cv2
        import numpy as np

        _, _, duration = get_video_info(video_path)
        usable_start = duration * 0.05
        usable_end = duration * 0.92

        # Candidate start times: dense grid across usable range
        step = max(clip_dur * 0.5, 0.5)
        candidates = []
        t = usable_start
        while t + clip_dur <= usable_end:
            candidates.append(t)
            t += step

        if not candidates:
            return []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames: one every 0.5s across the usable range
        sample_interval = max(int(fps * 0.5), 1)
        frame_data = {}  # frame_idx -> {sharpness, brightness, motion}
        prev_gray = None

        start_frame = int(usable_start * fps)
        end_frame = int(usable_end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for idx in range(start_frame, min(end_frame, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            if (idx - start_frame) % sample_interval != 0:
                continue

            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # Sharpness: Laplacian variance
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Brightness: mean pixel, penalise extremes
            brightness = float(gray.mean())
            bscore = 1.0 - abs(brightness - 110) / 110.0  # peak at ~110/255
            bscore = max(0.0, bscore)

            # Motion: mean abs diff from previous frame
            motion = 0.0
            if prev_gray is not None:
                motion = float(cv2.absdiff(gray, prev_gray).mean())
            prev_gray = gray

            frame_data[idx] = {
                'sharpness': sharpness,
                'brightness': bscore,
                'motion': motion,
                'ts': idx / fps,
            }

        cap.release()

        if not frame_data:
            return []

        # Normalise sharpness and motion to [0, 1]
        sharp_vals = [v['sharpness'] for v in frame_data.values()]
        motion_vals = [v['motion'] for v in frame_data.values()]
        max_sharp = max(sharp_vals) or 1
        max_motion = max(motion_vals) or 1

        for v in frame_data.values():
            v['sharpness'] = v['sharpness'] / max_sharp
            v['motion'] = v['motion'] / max_motion

        # Preference multipliers
        motion_w = 0.5
        sharp_w  = 0.3
        bright_w = 0.2

        prefer = prefs.get('prefer', 'action')  # 'action' or 'calm'
        if prefer == 'calm':
            motion_w, sharp_w = 0.2, 0.6

        def segment_score(start):
            end = start + clip_dur
            relevant = [v for v in frame_data.values() if start <= v['ts'] < end]
            if not relevant:
                return 0.0
            avg_motion = sum(v['motion']    for v in relevant) / len(relevant)
            avg_sharp  = sum(v['sharpness'] for v in relevant) / len(relevant)
            avg_bright = sum(v['brightness'] for v in relevant) / len(relevant)
            return motion_w * avg_motion + sharp_w * avg_sharp + bright_w * avg_bright

        scored = [(start, segment_score(start)) for start in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    except Exception:
        return []


def select_best_clips_cv2(scored, num_clips, clip_dur, raw_dur):
    """Greedy non-overlapping selection from highest-scored segments."""
    selected = []
    for start, score in scored:
        if start + clip_dur > raw_dur:
            continue
        if any(abs(start - s) < clip_dur * 0.85 for s in selected):
            continue
        selected.append(start)
        if len(selected) >= num_clips:
            break
    selected.sort()
    return selected


# ---------------------------------------------------------------------------
# Keyword-based instruction parser (no LLM required)
# ---------------------------------------------------------------------------

FAST_WORDS  = {'fast', 'quick', 'rapid', 'snappy', 'faster', 'speed', 'quicker'}
SLOW_WORDS  = {'slow', 'slower', 'relaxed', 'chill', 'smooth', 'longer'}
CALM_WORDS  = {'calm', 'chill', 'waiting', 'idle', 'sitting', 'talking'}
ACTION_WORDS = {'action', 'cut', 'cutting', 'fade', 'lineup', 'edge', 'clip', 'clipper',
                'razor', 'trim', 'buzz', 'style', 'styling', 'active'}
CAP_ON_WORDS  = {'caption', 'captions', 'text', 'label', 'title', 'titles', 'overlay'}
CAP_OFF_WORDS = {'no caption', 'no captions', 'no text', 'without caption', 'no overlay'}

def parse_instructions_local(instructions):
    if not instructions or not instructions.strip():
        return {}

    text = instructions.lower()
    result = {}

    # Style
    if any(w in text for w in FAST_WORDS):
        result['style'] = 'fast'
    elif any(w in text for w in SLOW_WORDS):
        result['style'] = 'slow'

    # Clip preference
    if any(w in text for w in CALM_WORDS) and 'skip' in text:
        result['prefer'] = 'action'  # "skip waiting" → prefer motion
    elif any(w in text for w in ACTION_WORDS):
        result['prefer'] = 'action'
    elif any(w in text for w in CALM_WORDS):
        result['prefer'] = 'calm'

    # Captions
    if any(phrase in text for phrase in CAP_OFF_WORDS):
        result['add_captions'] = False
    elif any(w in text for w in CAP_ON_WORDS):
        result['add_captions'] = True

    return result


# ---------------------------------------------------------------------------
# Beat detection (librosa — no API required)
# ---------------------------------------------------------------------------

def detect_beats(video_path):
    try:
        import librosa
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wav_path = tmp.name
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', wav_path
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(wav_path):
            return []
        y, sr = librosa.load(wav_path, sr=22050)
        os.unlink(wav_path)
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
        return librosa.frames_to_time(beats, sr=sr).tolist()
    except Exception:
        return []


def snap_to_nearest_beat(ts, beats, max_drift=0.35):
    if not beats:
        return ts
    nearest = min(beats, key=lambda b: abs(b - ts))
    return nearest if abs(nearest - ts) <= max_drift else ts


# ---------------------------------------------------------------------------
# Caption overlay (FFmpeg drawtext — no API required)
# ---------------------------------------------------------------------------

# Simple label pool based on clip index and detected motion level
HYPE_LABELS = [
    'The Cut', 'Fresh Drop', 'Edge Up', 'The Fade', 'Clean Lines',
    'Taper Time', 'The Blend', 'Sharp', 'The Lineup', 'Precision',
    'The Finish', 'On Point', 'The Detail', 'The Style', 'Clean',
]

def make_caption(clip_index, score=None):
    """Pick a caption label from the pool, cycling by index."""
    return HYPE_LABELS[clip_index % len(HYPE_LABELS)]


def get_font_arg():
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/opentype/noto/NotoSans-Bold.ttf',
    ]
    for p in candidates:
        if os.path.exists(p):
            return f':fontfile={p}'
    return ''


def burn_caption(input_path, caption, output_path):
    if not caption:
        shutil.copy2(input_path, output_path)
        return

    safe = caption.replace("'", "’").replace(':', ' ').replace('\\', '').replace('%', '%%')
    font_arg = get_font_arg()

    vf = (
        f"drawtext=text='{safe}'{font_arg}:"
        "fontsize=52:fontcolor=white:"
        "x=(w-text_w)/2:y=h-110:"
        "shadowcolor=black@0.85:shadowx=2:shadowy=2:"
        "box=1:boxcolor=black@0.45:boxborderw=14"
    )
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', vf,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-an', output_path
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=90)
    if r.returncode != 0 or not os.path.exists(output_path):
        shutil.copy2(input_path, output_path)


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_video(session_id, ref_path, raw_path, output_path, instructions=''):
    try:
        jobs[session_id].update(status='analyzing', message='Analysing reference video pacing...')

        ref = analyze_reference(ref_path)
        avg_clip = ref['avg_clip_duration']
        num_ref_clips = ref['num_cuts']

        # Step 1 — parse keyword instructions
        prefs = parse_instructions_local(instructions)
        style = prefs.get('style', 'medium')
        if style == 'fast':
            avg_clip = max(0.4, avg_clip * 0.7)
        elif style == 'slow':
            avg_clip = min(5.0, avg_clip * 1.4)

        jobs[session_id]['message'] = (
            f'Reference: {num_ref_clips} cuts detected, avg {avg_clip:.1f}s per clip'
        )

        raw_w, raw_h, raw_dur = get_video_info(raw_path)
        if not raw_dur or raw_dur < 5:
            raise ValueError('Raw video is too short (minimum 5 seconds required)')

        max_total = 19.5
        num_clips = min(num_ref_clips, int(max_total / avg_clip))
        num_clips = max(num_clips, 4)
        clip_dur = avg_clip
        if num_clips * clip_dur > max_total:
            clip_dur = max_total / num_clips

        # Step 2 — CV2 smart clip selection
        clip_starts = []
        cv2_used = False

        jobs[session_id]['message'] = 'Scoring footage for motion, sharpness & brightness...'
        scored = score_segments(raw_path, num_clips, clip_dur, prefs)
        if scored:
            clip_starts = select_best_clips_cv2(scored, num_clips, clip_dur, raw_dur)
            cv2_used = bool(clip_starts)

        # Fallback — even distribution
        if not clip_starts:
            usable_start = raw_dur * 0.05
            usable_end = min(raw_dur * 0.92, raw_dur - clip_dur)
            usable_range = usable_end - usable_start
            if usable_range < clip_dur:
                raise ValueError('Raw video is too short to extract enough clips')
            while num_clips * (usable_range / num_clips) < clip_dur and num_clips > 4:
                num_clips -= 1
            spacing = usable_range / num_clips
            clip_starts = [
                usable_start + i * spacing
                for i in range(num_clips)
                if usable_start + i * spacing + clip_dur <= raw_dur
            ]

        # Step 3 — beat detection & snap-to-beat
        jobs[session_id]['message'] = 'Detecting music beats for sync cuts...'
        beats = detect_beats(raw_path)
        beat_synced = False
        if beats:
            jobs[session_id]['message'] = f'Found {len(beats)} beats — snapping cuts to rhythm...'
            clip_starts = [snap_to_nearest_beat(s, beats) for s in clip_starts]
            beat_synced = True

        # Step 4 — decide on captions
        add_captions = prefs.get('add_captions', True)

        # Step 5 — cut clips
        jobs[session_id].update(status='processing', message=f'Cutting {len(clip_starts)} clips...')

        vf = build_crop_filter(raw_w, raw_h)
        vf_args = ['-vf', vf] if vf else []
        session_dir = get_session_dir(session_id)
        clip_files = []

        for i, start in enumerate(clip_starts):
            raw_clip  = os.path.join(session_dir, f'raw_{i:03d}.mp4')
            final_clip = os.path.join(session_dir, f'clip_{i:03d}.mp4')

            cmd = [
                'ffmpeg', '-y',
                '-ss', f'{start:.3f}',
                '-i', raw_path,
                '-t', f'{clip_dur:.3f}',
                *vf_args,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
                '-r', '30', '-an',
                raw_clip
            ]
            subprocess.run(cmd, capture_output=True, timeout=180)

            if not os.path.exists(raw_clip) or os.path.getsize(raw_clip) == 0:
                continue

            if add_captions:
                caption = make_caption(i)
                burn_caption(raw_clip, caption, final_clip)
                try:
                    os.remove(raw_clip)
                except OSError:
                    pass
                clip_files.append(final_clip if os.path.exists(final_clip) else raw_clip)
            else:
                shutil.move(raw_clip, final_clip)
                clip_files.append(final_clip)

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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
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
            avg_clip=round(clip_dur, 2),
            smart_selection=cv2_used,
            beat_synced=beat_synced,
            captions_added=add_captions,
        )

    except Exception as e:
        jobs[session_id].update(status='error', message=str(e))


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

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

    instructions = ''
    if request.is_json:
        instructions = request.json.get('instructions', '')
    else:
        instructions = request.form.get('instructions', '')

    jobs[session_id] = {'status': 'queued', 'message': 'Starting...'}

    t = threading.Thread(
        target=process_video,
        args=(session_id, ref_path, raw_path, output_path, instructions),
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
