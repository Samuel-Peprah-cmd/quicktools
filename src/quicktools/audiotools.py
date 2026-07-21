"""Audio transcription utilities, powered by faster-whisper (an optimized implementation
of OpenAI's Whisper speech recognition model) and Pyannote for Speaker Diarization.

Requires the optional 'faster-whisper' package: pip install faster-whisper
Diarization requires 'pyannote.audio': pip install pyannote.audio
The first call for a given model_size downloads that model (requires internet, one-time).
No separate ffmpeg installation is required — audio decoding is bundled in.
"""

def _load_model(model_size: str = "base"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "Audio transcription requires faster-whisper. Install it with: pip install faster-whisper"
        )
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(path: str, model_size: str = "base", language: str | None = None) -> str:
    """Transcribe an audio file to plain text. model_size options (accuracy vs speed):
    'tiny', 'base', 'small', 'medium', 'large-v3'. language is an optional ISO code
    (e.g. 'en') to skip auto-detection and speed things up."""
    model = _load_model(model_size)
    segments, _ = model.transcribe(path, language=language)
    return " ".join(segment.text.strip() for segment in segments)


def transcribe_audio_with_timestamps(path: str, model_size: str = "base") -> list[dict]:
    """Transcribe audio into a list of segments, each with 'start', 'end' (seconds), and 'text'."""
    model = _load_model(model_size)
    segments, _ = model.transcribe(path)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def transcribe_audio_word_level(path: str, model_size: str = "base") -> list[dict]:
    """Transcribe audio into a word-by-word list, each with 'word', 'start', and 'end' (seconds).
    This gives true word-for-word timing, useful for captions or karaoke-style highlighting."""
    model = _load_model(model_size)
    segments, _ = model.transcribe(path, word_timestamps=True)
    words = []
    for segment in segments:
        for word in segment.words:
            words.append({"word": word.word.strip(), "start": word.start, "end": word.end})
    return words


def detect_audio_language(path: str, model_size: str = "base") -> str:
    """Detect the spoken language of an audio file, returning its ISO language code (e.g. 'en')."""
    model = _load_model(model_size)
    _, info = model.transcribe(path)
    return info.language


def save_transcript_as_srt(segments: list[dict], output_path: str) -> None:
    """Save a list of transcript segments (from transcribe_audio_with_timestamps) as an .srt
    subtitle file, ready to use with any video player or editor."""
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --- NEW DIARIZATION AND STYLING FEATURES ---

def transcribe_with_speakers(path: str, hf_token: str, model_size: str = "base") -> list[dict]:
    """
    Transcribes audio and aligns the text with Speaker IDs using Pyannote.
    Returns a list of dicts: [{"start": 0.0, "end": 5.0, "speaker": "Speaker 00", "text": "Hello"}]
    Requires a Hugging Face token with access to pyannote/speaker-diarization-3.1.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise ImportError(
            "Speaker diarization requires pyannote.audio. Install it with: pip install pyannote.audio"
        )

    print("1/3: Analyzing voice biometrics (Diarization)...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diarization = pipeline(path)

    print("2/3: Transcribing audio text...")
    model = _load_model(model_size)
    segments, _ = model.transcribe(path)

    print("3/3: Aligning text to speakers...")
    transcript_data = []
    
    for segment in segments:
        # Find the midpoint of the spoken phrase
        segment_midpoint = (segment.start + segment.end) / 2
        
        # Check which speaker was talking at that exact millisecond
        current_speaker = "Unknown"
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= segment_midpoint <= turn.end:
                current_speaker = speaker
                break
                
        transcript_data.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": current_speaker.replace("SPEAKER_", "Speaker "),
            "text": segment.text.strip()
        })
        
    return transcript_data


def save_transcript_to_docx(transcript_data: list[dict], output_path: str) -> None:
    """
    Takes speaker-mapped transcript data and generates a beautifully styled Word document.
    Requires the 'python-docx' package.
    """
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
    except ImportError:
        raise ImportError("Saving to Word requires python-docx. Run: pip install python-docx")
        
    doc = Document()
    title = doc.add_heading("Audio Meeting Transcript", level=1)
    title.alignment = 1 # Center align

    # Track the last speaker so we don't print the name over and over for continuous talking
    last_speaker = None

    for entry in transcript_data:
        p = doc.add_paragraph()
        
        # Convert raw seconds into [MM:SS] format
        start_m, start_s = divmod(int(entry['start']), 60)
        time_str = f"[{start_m:02d}:{start_s:02d}]"
        
        # Only print the Speaker ID if the speaker changed
        if entry['speaker'] != last_speaker:
            speaker_run = p.add_run(f"{time_str} {entry['speaker']}:\n")
            speaker_run.bold = True
            # Give the speaker tag a nice professional blue color
            speaker_run.font.color.rgb = RGBColor(0, 102, 204) 
            last_speaker = entry['speaker']
        
        # Add the actual spoken text
        p.add_run(entry['text'])
        p.paragraph_format.space_after = Pt(8)
        
    doc.save(output_path)