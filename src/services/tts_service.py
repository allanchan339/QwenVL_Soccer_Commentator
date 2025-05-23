"""Text-to-Speech service using Minimax API."""

import os
from typing import Optional

from .minimax_tts import generate_minimax_tts
from ..config import TEMP_AUDIO_DIR, DEFAULT_AUDIO_FORMAT
from ..utils.video_utils import ensure_directory_exists


class TTSService:
    """Service for generating speech from text using Minimax TTS."""
    
    def __init__(self, output_dir: str = TEMP_AUDIO_DIR):
        """Initialize the TTS service.
        
        Args:
            output_dir: Directory to save generated audio files
        """
        self.output_dir = output_dir
        ensure_directory_exists(self.output_dir)
    
    def generate_audio(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        """Generate TTS audio from text.
        
        Args:
            text: Text to convert to speech
            filename: Optional filename for the audio file. If None, generates one based on text hash.
            
        Returns:
            Path to generated audio file if successful, None otherwise
        """
        if not text or text.startswith("Error"):
            return None
        
        try:
            # Generate filename if not provided
            if filename is None:
                audio_filename = f"commentary_{hash(text) % 1000000}.{DEFAULT_AUDIO_FORMAT}"
            else:
                audio_filename = filename
            
            audio_path = os.path.join(self.output_dir, audio_filename)
            
            # Call Minimax TTS
            generated_audio_path = generate_minimax_tts(text, audio_path)
            
            if generated_audio_path:
                return generated_audio_path
            else:
                print("Minimax TTS failed, no audio generated.")
                return None
                
        except Exception as e:
            print(f"Error generating TTS audio: {str(e)}")
            return None 