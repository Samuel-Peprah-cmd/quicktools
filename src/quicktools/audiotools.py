"""Audio transcription utilities, powered by faster-whisper (an optimized implementation
of OpenAI's Whisper speech recognition model) and Pyannote for Speaker Diarization.

Requires the optional 'faster-whisper' package: pip install faster-whisper
Diarization requires 'pyannote.audio': pip install pyannote.audio
The first call for a given model_size downloads that model (requires internet, one-time).
No separate ffmpeg installation is required — audio decoding is bundled in.
"""
import os
import warnings
import torch
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")

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

def _load_waveform_via_av(path: str, target_sr: int = 16000):
    """Decode any audio/video file into a mono waveform tensor using PyAV — bypassing
    pyannote's own (fragile, Windows-unfriendly) torchcodec-based audio loader entirely."""
    import av
    import torch
    import numpy as np

    container = av.open(path)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
    audio_stream = container.streams.audio[0]

    chunks = []
    for frame in container.decode(audio_stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray())
    container.close()

    if not chunks:
        raise RuntimeError(f"No audio could be decoded from '{path}'")

    audio = np.concatenate(chunks, axis=1).astype(np.float32) / 32768.0  # int16 -> [-1, 1]
    waveform = torch.from_numpy(audio)  # shape: (1, num_samples) since layout="mono"
    return waveform, target_sr


def _get_diarization_turns(diarization):
    """Normalize pyannote's diarization output across library versions into a list of
    (turn, speaker_label) pairs. Newer versions expose `.speaker_diarization` directly;
    older versions require `.itertracks(yield_label=True)`."""
    if hasattr(diarization, "speaker_diarization"):
        return list(diarization.speaker_diarization)
    return [(turn, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]


def _get_diarization_turns(output):
    """
    Helper to extract diarization turns, gracefully handling 
    the API changes introduced in pyannote.audio 4.x.
    """
    if hasattr(output, "speaker_diarization"):
        diarization = output.speaker_diarization
    else:
        diarization = output

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((turn, speaker))
    
    return turns


def transcribe_with_speakers(path: str, hf_token: str, model_size: str = "base",
                             device: str = "auto") -> list[dict]:
    """Transcribe audio and label each segment with which speaker said it (speaker diarization).
    Returns a list of dicts: [{"start": 0.0, "end": 5.0, "speaker": "Speaker 00", "text": "Hello"}]

    device: "auto" (default), "cuda", or "cpu". "auto" will detect an NVIDIA GPU if available.
    Diarization is dramatically faster on GPU (often 10-20x). On CPU, expect roughly 1-3x the audio's 
    own length in processing time; a 10-minute recording may take 10-30 minutes.

    Requires:
    - pip install pyannote.audio
    - A free Hugging Face account + access token (huggingface.co/settings/tokens)
    - Accepting the model terms at huggingface.co/pyannote/speaker-diarization-3.1
      AND huggingface.co/pyannote/speaker-diarization-community-1
    """
    if os.name == 'nt':
        print("💡 Tip: For faster native audio processing, you can install FFmpeg.")
        print("   Run this command in PowerShell: winget install ffmpeg\n")
        
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise ImportError(
            "Speaker diarization requires pyannote.audio. Install it with: pip install pyannote.audio"
        )

    # --- THE SPEED FIX: Auto-detect hardware ---
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # This warning fires purely because pyannote.audio's internal loader checks for
    # torchcodec at import time — it's irrelevant here since quicktools decodes audio
    # itself via PyAV and never touches pyannote's own loader. Safe to suppress.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        print("[1/4] Decoding audio...")
        waveform, sample_rate = _load_waveform_via_av(path)

        print("[2/4] Loading diarization model (first run downloads it, ~1-2 min)...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        
        # Apply the speed boost if GPU is available
        if device == "cuda":
            pipeline.to(torch.device("cuda"))
            print("⚡ NVIDIA GPU detected and enabled! Processing will be exponentially faster.")
        else:
            print("🐢 No GPU detected (or forced CPU). Running on CPU (this takes a very long time for large files).")

        print("[3/4] Running speaker diarization (the slow part — progress below)...")
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            with ProgressHook() as hook:
                diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
        except ImportError:
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    print("[4/4] Transcribing speech to text...")
    # NOTE: Assuming _load_model() handles your internal Whisper loading
    model = _load_model(model_size)
    segments, _ = model.transcribe(path)

    print("Aligning speakers to transcript...")
    # Use the helper to bypass the 4.x Pyannote crash
    turns = _get_diarization_turns(diarization)

    transcript_data = []
    for segment in segments:
        midpoint = (segment.start + segment.end) / 2
        current_speaker = "Unknown"
        for turn, speaker in turns:
            if turn.start <= midpoint <= turn.end:
                current_speaker = speaker
                break
        transcript_data.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": current_speaker.replace("SPEAKER_", "Speaker "),
            "text": segment.text.strip(),
        })
        
    print("Done!")
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