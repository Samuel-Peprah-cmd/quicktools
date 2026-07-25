"""Video utilities: transcription, audio extraction, metadata, frame capture, 
and speaker diarization — powered by PyAV, faster-whisper, and Pyannote.

Requires the optional 'av' package: pip install av
(This is already installed automatically as a dependency of faster-whisper.)

Supports common video containers: MP4, MOV, MKV, AVI, WEBM, and more — since PyAV
uses FFmpeg's decoding engine internally, the same one that powers audiotools.
"""
import os
import shutil
import tempfile


def get_video_info(path: str) -> dict:
    """Return basic metadata about a video file: duration (seconds), width, height, and frame rate."""
    import av
    container = av.open(path)
    stream = container.streams.video[0]
    duration = float(container.duration / av.time_base) if container.duration else None
    info = {
        "duration_seconds": duration,
        "width": stream.width,
        "height": stream.height,
        "fps": float(stream.average_rate) if stream.average_rate else None,
    }
    container.close()
    return info


def get_video_duration(path: str) -> float:
    """Return the duration of a video file in seconds."""
    return get_video_info(path)["duration_seconds"]


def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    """Extract the audio track from a video file and save it as a standalone audio file
    (format is inferred from output_audio_path's extension, e.g. .mp3, .wav, .m4a)."""
    import av

    input_container = av.open(video_path)
    audio_stream = input_container.streams.audio[0]

    output_container = av.open(output_audio_path, mode="w")
    output_stream = output_container.add_stream("aac" if output_audio_path.endswith((".m4a", ".mp4")) else "mp3")

    for frame in input_container.decode(audio_stream):
        for packet in output_stream.encode(frame):
            output_container.mux(packet)

    for packet in output_stream.encode(None):
        output_container.mux(packet)

    output_container.close()
    input_container.close()


def extract_video_frame(video_path: str, timestamp_seconds: float, output_image_path: str) -> None:
    """Extract a single frame from a video at the given timestamp (seconds) and save it as an image."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]

    target_pts = int(timestamp_seconds / stream.time_base)
    container.seek(target_pts, stream=stream)

    for frame in container.decode(stream):
        if frame.time >= timestamp_seconds:
            frame.to_image().save(output_image_path)
            break

    container.close()


def transcribe_video(path: str, model_size: str = "base", language: str | None = None) -> str:
    """Transcribe the spoken audio in a video file to plain text. Works directly on
    video containers (MP4, MOV, MKV, etc.) — the audio track is extracted automatically."""
    from quicktools.audiotools import transcribe_audio
    return transcribe_audio(path, model_size=model_size, language=language)


def transcribe_video_with_timestamps(path: str, model_size: str = "base") -> list[dict]:
    """Transcribe a video's audio into timestamped segments, each with 'start', 'end', and 'text'."""
    from quicktools.audiotools import transcribe_audio_with_timestamps
    return transcribe_audio_with_timestamps(path, model_size=model_size)


def transcribe_video_word_level(path: str, model_size: str = "base") -> list[dict]:
    """Transcribe a video's audio into word-by-word timestamps, each with 'word', 'start', and 'end'.
    Useful for generating captions synced precisely to speech."""
    from quicktools.audiotools import transcribe_audio_word_level
    return transcribe_audio_word_level(path, model_size=model_size)


# --- NEW DIARIZATION AND STYLING FEATURES ---

from quicktools import audiotools

def transcribe_video_with_speakers(path_or_url: str, hf_token: str, model_size: str = "base", device: str = "auto") -> list[dict]:
    """
    Extracts audio from a local video file or web URL (YouTube, TikTok, IG, X), 
    transcribes it, and maps the text to specific speakers.
    """
    print(f"Processing video source: {path_or_url}")
    return audiotools.transcribe_with_speakers(path_or_url, hf_token, model_size, device)


def save_video_script_to_docx(transcript_data: list[dict], output_path: str, title_text: str = "Video Script & Transcript") -> None:
    """
    Formats speaker-mapped video transcript data as a professional script in Word (.docx).
    """
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
    except ImportError:
        raise ImportError("Saving to Word requires python-docx. Run: pip install python-docx")

    doc = Document()
    title = doc.add_heading(title_text, level=1)
    title.alignment = 1  # Center align

    last_speaker = None

    for entry in transcript_data:
        p = doc.add_paragraph()

        # Format timestamp as [MM:SS]
        start_m, start_s = divmod(int(entry["start"]), 60)
        time_str = f"[{start_m:02d}:{start_s:02d}]"

        # Speaker header line if speaker changed
        if entry["speaker"] != last_speaker:
            speaker_run = p.add_run(f"{time_str} {entry['speaker']}:\n")
            speaker_run.bold = True
            speaker_run.font.color.rgb = RGBColor(112, 48, 160)  # Distinct Purple highlight for video speakers
            last_speaker = entry["speaker"]

        p.add_run(entry["text"])
        p.paragraph_format.space_after = Pt(8)

    doc.save(output_path)


import os
import shutil
import random

def download_video(url: str, output_dir: str = ".", filename: str | None = None, resolution: str = "best", cookiefile: str | None = None) -> str:
    """
    Downloads a video file from supported web URLs (YouTube, TikTok, Instagram, X/Twitter, etc.).
    
    :param url: The web video URL to download.
    :param output_dir: Destination folder (defaults to current directory).
    :param filename: Optional custom filename (without extension). Defaults to video title.
    :param resolution: Quality target ('best' or 'worst').
    :param cookiefile: Path to a cookies.txt file to bypass login restrictions.
    :return: Absolute file path of the downloaded video.
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "Downloading videos requires 'yt-dlp'. Install it with: pip install yt-dlp"
        )

    os.makedirs(output_dir, exist_ok=True)
    out_template = f"{filename}.%(ext)s" if filename else "%(title)s.%(ext)s"
    target_path = os.path.join(output_dir, out_template)

    # --- AUTO-DETECT SERVER COOKIES ---
    if not cookiefile and os.path.exists('/app/cookies.txt'):
        cookiefile = '/app/cookies.txt'
    # ----------------------------------

    # --- THE SMART FFMPEG FALLBACK ---
    has_ffmpeg = shutil.which("ffmpeg") is not None
    
    if resolution == "best":
        if has_ffmpeg:
            # Grab max quality separate streams and merge them
            fmt = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        else:
            # Fall back to the best PRE-MERGED format (usually 720p max) to prevent a crash
            print("⚠️ [quicktools] FFmpeg not found. Falling back to pre-merged 720p format.")
            print("💡 Tip: Install FFmpeg ('winget install ffmpeg') to enable 1080p+ downloads.")
            fmt = 'best[ext=mp4]/best'
    else:
        fmt = 'worst'
    # ---------------------------------

    ydl_opts = {
        'format': fmt,
        'outtmpl': target_path,
        'quiet': False,
        'no_warnings': True,
        'merge_output_format': 'mp4' if has_ffmpeg else None,
        'extractor_args': {'youtube': ['player_client=android']}, 
    }

    # --- CRITICAL APPLE iOS FIXES ---
    if has_ffmpeg:
        # Force FFmpeg to move the moov atom to the top of the file (Fast Start)
        ydl_opts['postprocessor_args'] = ['-movflags', '+faststart']
    # --------------------------------

    # --- ATTACH COOKIE IF AVAILABLE ---
    if cookiefile and os.path.exists(cookiefile):
        print(f"🔑 Using cookie authentication file: {cookiefile}")
        ydl_opts['cookiefile'] = cookiefile
    # ----------------------------------

    # --- HELPER FUNCTION FOR EXECUTION AND VALIDATION ---
    def attempt_download(opts: dict) -> str:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)

            # Check if output was merged into an mp4 container
            base, _ = os.path.splitext(filepath)
            if os.path.exists(f"{base}.mp4"):
                filepath = f"{base}.mp4"

            # CRITICAL: verify the file actually exists AND has real content.
            # A silent merge failure can leave a 0-byte or missing file that would
            # otherwise look "successful" to the caller.
            if not os.path.exists(filepath):
                raise RuntimeError(f"Download appeared to succeed but no file was found at {filepath}")
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise RuntimeError(f"Downloaded file is empty (0 bytes): {filepath}")

            print(f"✅ Video saved to: {filepath} ({file_size / 1024 / 1024:.1f} MB)")
            return os.path.abspath(filepath)

    # === ATTEMPT 1: IPv6 Workaround ===
    print(f"📥 [Attempt 1] Downloading video from {url} via IPv6...")
    ydl_opts_ipv6 = ydl_opts.copy()
    ydl_opts_ipv6['source_address'] = '0::0'

    try:
        return attempt_download(ydl_opts_ipv6)
    except Exception as e_ipv6:
        print(f"⚠️ IPv6 Attempt Failed/Blocked: {str(e_ipv6)}")

        # === ATTEMPT 2: Rotating Webshare Proxy Fallback ===
        proxy_env = os.getenv("RESIDENTIAL_PROXY")
        if proxy_env:
            # Parse the comma-separated string of 10 proxies you added to .env
            proxy_list = [p.strip() for p in proxy_env.split(",") if p.strip()]
            chosen_proxy = random.choice(proxy_list)
            
            # Print masked proxy IP for terminal debugging (hides your username/password)
            masked_ip = chosen_proxy.split('@')[-1] if '@' in chosen_proxy else chosen_proxy
            print(f"🛡️ [Attempt 2] Routing download through random Webshare proxy ({masked_ip})...")
            
            ydl_opts_proxy = ydl_opts.copy()
            ydl_opts_proxy['proxy'] = chosen_proxy

            try:
                return attempt_download(ydl_opts_proxy)
            except Exception as e_proxy:
                print(f"❌ Proxy Attempt Failed: {str(e_proxy)}")
                raise RuntimeError(f"Both IPv6 and Proxy downloads failed. Last error: {str(e_proxy)}")
        else:
            print("❌ No RESIDENTIAL_PROXY found in environment variables.")
            raise RuntimeError(f"IPv6 download failed ({str(e_ipv6)}), and no Webshare Proxy was configured.")


import subprocess
def convert_video_to_animated(input_path: str, output_path: str, target_format: str = "gif", fps: int = 15, width: int = 512) -> None:
    """Converts a video to an animated GIF or WebP sticker using FFmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required for video-to-animation conversion. Please install it.")
    
    # WhatsApp/Telegram stickers prefer a 512px boundary. We scale proportionally.
    scale_filter = f"fps={fps},scale={width}:-1:flags=lanczos"
    
    if target_format.lower() == "gif":
        # High-quality GIF generation using a 2-pass color palette
        vf_cmd = f"{scale_filter},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
        cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", vf_cmd, "-loop", "0", output_path]
    elif target_format.lower() == "webp":
        # Animated WebP optimized for social media stickers
        cmd = [
            "ffmpeg", "-y", "-i", input_path, 
            "-vcodec", "libwebp", 
            "-vf", scale_filter,
            "-lossless", "0", 
            "-compression_level", "4", 
            "-q:v", "50", 
            "-loop", "0", 
            "-preset", "picture", 
            "-an", "-vsync", "0", 
            output_path
        ]
    else:
        raise ValueError("Target format must be 'gif' or 'webp'")

    # Execute FFmpeg silently
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")