import os
import json
import uuid
import threading
import subprocess
import base64
import re
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

# Initialise Claude client if API key is present
_claude = None
try:
    import anthropic as _anthropic
    _api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if _api_key:
        _claude = _anthropic.Anthropic(api_key=_api_key)
except Exception:
    pass


def has_ai():
    return _claude is not None


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
    """Detect scene cuts in reference video using PySceneDetect, fall back to CV2."""
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
# AI — frame extraction
# ---------------------------------------------------------------------------

def extract_frames_for_ai(video_path, num_frames=16):
    """Return list of {timestamp, b64} dicts for evenly-spaced frames."""
    _, _, duration = get_video_info(video_path)
    if not duration:
        return []

    start_t = duration * 0.05
    end_t = duration * 0.92
    span = end_t - start_t
    if span <= 0:
        return []

    frames = []
    for i in range(num_frames):
        ts = start_t + (i / max(num_frames - 1, 1)) * span
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cmd = [
                'ffmpeg', '-y', '-ss', f'{ts:.3f}', '-i', video_path,
                '-vframes', '1', '-vf', 'scale=320:-1', '-q:v', '4', tmp_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=20)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                with open(tmp_path, 'rb') as f:
                    b64 = base64.standard_b64encode(f.read()).decode()
                frames.append({'timestamp': ts, 'b64': b64})
        except Exception:
            pass
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return frames


# ---------------------------------------------------------------------------
# AI — natural language instruction parsing
# ---------------------------------------------------------------------------

def parse_instructions(instructions):
    """Convert plain-text editing instructions into structured params via Claude."""
    if not instructions or not instructions.strip() or not has_ai():
        return {}
    try:
        resp = _claude.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=300,
            messages=[{
                'role': 'user',
                'content': (
                    f'Parse these video editing instructions into JSON.\n'
                    f'Instructions: "{instructions}"\n\n'
                    'Return ONLY valid JSON with these exact fields:\n'
                    '{\n'
                    '  "focus_on": ["moments or actions to prioritise"],\n'
                    '  "avoid": ["moments to skip"],\n'
                    '  "style": "fast" | "medium" | "slow",\n'
                    '  "add_captions": true | false,\n'
                    '  "caption_style": "hype" | "minimal" | "descriptive"\n'
                    '}'
                )
            }]
        )
        m = re.search(r'\{[\s\S]*\}', resp.content[0].text)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# AI — frame scoring via Claude vision
# ---------------------------------------------------------------------------

def score_frames(frames, instructions='', parsed=None):
    """Score each frame 1-10 for visual interest using Claude vision."""
    if not frames or not has_ai():
        return []

    focus_ctx = ''
    if parsed:
        if parsed.get('focus_on'):
            focus_ctx += f'\nPrioritise: {", ".join(parsed["focus_on"])}'
        if parsed.get('avoid'):
            focus_ctx += f'\nAvoid: {", ".join(parsed["avoid"])}'
    elif instructions:
        focus_ctx = f'\nUser request: {instructions}'

    content = [{
        'type': 'text',
        'text': (
            f'Analyse {len(frames)} frames from a barber/haircut video. '
            'Rate each 1-10 for social media highlight reel value.\n'
            '10 = dynamic cutting/styling action, great angle\n'
            '7-9 = clear barber work visible, decent framing\n'
            '4-6 = setup or transition moments\n'
            '1-3 = idle, blurry, bad angle, nothing happening\n'
            f'{focus_ctx}\n\n'
            'Return ONLY a JSON array — one object per frame:\n'
            '[{"frame":0,"score":8,"description":"close-up fade being cut"},...]\n\n'
            'Frames follow:'
        )
    }]

    for i, fr in enumerate(frames):
        content.append({'type': 'text', 'text': f'Frame {i} (t={fr["timestamp"]:.1f}s):'})
        content.append({
            'type': 'image',
            'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': fr['b64']}
        })

    try:
        resp = _claude.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=1200,
            messages=[{'role': 'user', 'content': content}]
        )
        m = re.search(r'\[[\s\S]*?\]', resp.content[0].text)
        if m:
            scores = json.loads(m.group())
            for s in scores:
                idx = s.get('frame', 0)
                if 0 <= idx < len(frames):
                    s['timestamp'] = frames[idx]['timestamp']
            return scores
    except Exception:
        pass
    return []


def select_best_clips(scores, num_clips, clip_dur, raw_dur):
    """Greedily pick non-overlapping clips from the highest-scored moments."""
    if not scores:
        return []
    ranked = sorted(scores, key=lambda x: x.get('score', 0), reverse=True)
    selected = []  # (start_time, description)
    for item in ranked:
        ts = item.get('timestamp', 0)
        start = max(0.0, ts - clip_dur * 0.3)
        if start + clip_dur > raw_dur:
            start = max(0.0, raw_dur - clip_dur)
        if any(abs(start - s) < clip_dur * 0.85 for s, _ in selected):
            continue
        selected.append((start, item.get('description', '')))
        if len(selected) >= num_clips:
            break
    selected.sort(key=lambda x: x[0])
    return selected


# ---------------------------------------------------------------------------
# AI — beat detection
# ---------------------------------------------------------------------------

def detect_beats(video_path):
    """Extract audio and find beat timestamps via librosa."""
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
# AI — caption generation & overlay
# ---------------------------------------------------------------------------

def generate_captions_batch(descriptions, style='hype'):
    """Convert clip descriptions to short punchy captions in one Claude call."""
    if not descriptions or not has_ai():
        return [''] * len(descriptions)

    style_map = {
        'hype': 'punchy social media energy, barbershop slang (e.g. "Fresh Fade Drop", "Edge Up Time", "Taper Season")',
        'minimal': 'minimal, 1-2 words each, title case',
        'descriptive': 'clear and brief, describe the action in 3-4 words',
    }
    style_desc = style_map.get(style, style_map['hype'])
    desc_list = '\n'.join(f'{i}. {d}' for i, d in enumerate(descriptions))

    try:
        resp = _claude.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=256,
            messages=[{
                'role': 'user',
                'content': (
                    f'Convert these barber video descriptions into {style_desc} captions.\n'
                    'Rules: 2-4 words each, title case, no punctuation.\n\n'
                    f'{desc_list}\n\n'
                    'Return ONLY a JSON string array:\n["caption 1","caption 2",...]'
                )
            }]
        )
        m = re.search(r'\[[\s\S]*?\]', resp.content[0].text)
        if m:
            captions = json.loads(m.group())
            if len(captions) == len(descriptions):
                return captions
    except Exception:
        pass
    return [''] * len(descriptions)


def get_font_arg():
    """Return ':fontfile=...' string for the first available bold font, or empty string."""
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
    """Burn a text caption near the bottom of the clip via FFmpeg drawtext."""
    if not caption:
        shutil.copy2(input_path, output_path)
        return

    # Sanitise text for FFmpeg drawtext
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

        # Step 1 — parse natural language instructions
        parsed = {}
        if instructions.strip() and has_ai():
            jobs[session_id]['message'] = 'Claude is reading your editing instructions...'
            parsed = parse_instructions(instructions)
            style = parsed.get('style', 'medium')
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

        # Step 2 — AI frame analysis & smart clip selection
        clip_selections = []  # list of (start_time, description)
        ai_used = False

        if has_ai():
            jobs[session_id]['message'] = 'Extracting frames for Claude to analyse...'
            frames = extract_frames_for_ai(raw_path, num_frames=min(16, num_clips * 3))

            if frames:
                jobs[session_id]['message'] = 'Claude is selecting the best moments in your footage...'
                scores = score_frames(frames, instructions, parsed)
                if scores:
                    clip_selections = select_best_clips(scores, num_clips, clip_dur, raw_dur)
                    ai_used = bool(clip_selections)

        # Fallback — even distribution
        if not clip_selections:
            usable_start = raw_dur * 0.05
            usable_end = min(raw_dur * 0.92, raw_dur - clip_dur)
            usable_range = usable_end - usable_start
            if usable_range < clip_dur:
                raise ValueError('Raw video is too short to extract enough clips')
            while num_clips * (usable_range / num_clips) < clip_dur and num_clips > 4:
                num_clips -= 1
            spacing = usable_range / num_clips
            clip_selections = [
                (usable_start + i * spacing, '')
                for i in range(num_clips)
                if usable_start + i * spacing + clip_dur <= raw_dur
            ]

        # Step 3 — beat detection & snap-to-beat
        jobs[session_id]['message'] = 'Detecting music beats for sync cuts...'
        beats = detect_beats(raw_path)
        beat_synced = False
        if beats:
            jobs[session_id]['message'] = f'Found {len(beats)} beats — snapping cuts to rhythm...'
            clip_selections = [
                (snap_to_nearest_beat(start, beats), desc)
                for start, desc in clip_selections
            ]
            beat_synced = True

        # Step 4 — generate captions
        add_captions = parsed.get('add_captions', has_ai())
        caption_style = parsed.get('caption_style', 'hype')
        captions = []

        if add_captions and has_ai():
            descriptions = [desc for _, desc in clip_selections]
            if any(descriptions):
                jobs[session_id]['message'] = 'Generating captions with Claude...'
                captions = generate_captions_batch(descriptions, caption_style)

        # Step 5 — cut clips
        jobs[session_id].update(status='processing', message=f'Cutting {len(clip_selections)} clips...')

        vf = build_crop_filter(raw_w, raw_h)
        vf_args = ['-vf', vf] if vf else []
        session_dir = get_session_dir(session_id)
        clip_files = []

        for i, (start, desc) in enumerate(clip_selections):
            raw_clip = os.path.join(session_dir, f'raw_{i:03d}.mp4')
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

            caption = captions[i] if i < len(captions) else ''
            if caption:
                burn_caption(raw_clip, caption, final_clip)
                try:
                    os.remove(raw_clip)
                except OSError:
                    pass
                if os.path.exists(final_clip) and os.path.getsize(final_clip) > 0:
                    clip_files.append(final_clip)
                else:
                    clip_files.append(raw_clip)
            else:
                shutil.move(raw_clip, final_clip)
                clip_files.append(final_clip)

            jobs[session_id]['message'] = f'Cut {i + 1}/{len(clip_selections)} clips...'

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
            ai_clip_selection=ai_used,
            beat_synced=beat_synced,
            captions_added=bool(captions and any(captions)),
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

    return jsonify({'session_id': session_id, 'ai_available': has_ai()})


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
    mode = 'AI-powered' if has_ai() else 'basic (set ANTHROPIC_API_KEY to enable AI)'
    print(f'Barber Video Editor — {mode}')
    print('Running at http://localhost:5555')
    app.run(host='0.0.0.0', port=5555, debug=False, threaded=True)
