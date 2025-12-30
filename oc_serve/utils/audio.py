"""Helper functions for audio processing."""
import io
from typing import List, Tuple
import pydub

def split_audio_by_time(audio_bytes: bytes,
                        chunk_length_ms: int = 25000,
                        ) -> Tuple[List[bytes], float]:
    """Splits audio bytes into chunks of specified length in milliseconds.
    Args:
        audio_bytes (bytes): The input audio in bytes.
        chunk_length_ms (int): The length of each chunk in milliseconds.
    
    Returns:
        List[bytes]: A list of audio chunks in bytes.
        float: Total duration of the audio in seconds.
    """
    audio = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes))
    duration = audio.duration_seconds
    chunks = []

    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        buf = io.BytesIO()
        chunk.export(buf, format="wav")
        buf.seek(0)
        chunks.append(buf.read())

    return chunks, duration
