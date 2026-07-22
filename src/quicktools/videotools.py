"""Video utilities: transcription, audio extraction, metadata, frame capture, 
and speaker diarization — powered by PyAV, faster-whisper, and Pyannote.

Requires the optional 'av' package: pip install av
(This is already installed automatically as a dependency of faster-whisper.)

Supports common video containers: MP4, MOV, MKV, AVI, WEBM, and more — since PyAV
uses FFmpeg's decoding engine internally, the same one that powers audiotools.
"""
import os
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