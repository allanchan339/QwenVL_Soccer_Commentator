"""Text-to-Speech service using ModelScope."""

import os
import asyncio
from typing import Optional
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import edge_tts

from ..config import TEMP_AUDIO_DIR, DEFAULT_AUDIO_FORMAT
from ..utils.video_utils import ensure_directory_exists


class TTSService:
    """Service for generating speech from text using ModelScope TTS."""
    
    def __init__(self, output_dir: str = TEMP_AUDIO_DIR):
        """Initialize the TTS service.
        
        Args:
            output_dir: Directory to save generated audio files
        """
        self.output_dir = output_dir
        ensure_directory_exists(self.output_dir)
        
        # Initialize the TTS pipeline
        self.model_id = 'speech_tts/speech_sambert-hifigan_tts_jiajia_Cantonese_16k'
        try:
            self.tts_pipeline = pipeline(task=Tasks.text_to_speech, model=self.model_id)
        except Exception as e:
            print(f"Warning: Failed to initialize TTS pipeline: {str(e)}")
            self.tts_pipeline = None
    
    def generate_audio(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        """Generate TTS audio from text using ModelScope.
        
        Args:
            text: Text to convert to speech
            filename: Optional filename for the audio file. If None, generates one based on text hash.
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not text or text.startswith("Error"):
            return None
        
        if not self.tts_pipeline:
            print("TTS pipeline not available")
            return None
        
        try:
            # Generate filename if not provided
            if filename is None:
                audio_filename = f"commentary_{hash(text) % 1000000}.wav"
            else:
                audio_filename = filename
            
            audio_path = os.path.join(self.output_dir, audio_filename)
            
            # Generate audio using ModelScope TTS
            output = self.tts_pipeline(input=text)
            wav_data = output[OutputKeys.OUTPUT_WAV]
            
            # Save the audio file
            with open(audio_path, 'wb') as f:
                f.write(wav_data)
            
            print(f"Generated TTS audio: {audio_path}")
            return audio_path
                
        except Exception as e:
            print(f"Error generating TTS audio: {str(e)}")
            return None 