# tts.py
"""
Handles TTS (text-to-speech) generation logic.
"""
import tempfile
import asyncio
from edge_tts import Communicate

def tts_generate(text, voice="en-US-AriaNeural"):
    if not text or not text.strip():
        return None
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = temp_wav.name
    temp_wav.close()
    async def synthesize():
        communicate = Communicate(text, voice)
        await communicate.save(output_path)
    asyncio.run(synthesize())
    return output_path 