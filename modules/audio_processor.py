"""
Audio processing module for soccer video analysis.
Currently disabled - no audio processing functionality.
"""

from typing import Optional


class AudioProcessor:
    """Placeholder audio processor - currently disabled."""
    
    def __init__(self):
        """Initialize the audio processor (disabled)."""
        print("ℹ️  Audio processing is currently disabled")
    
    def generate_audio(self, text: str, voice_id: Optional[str] = None, speed: Optional[float] = None) -> Optional[str]:
        """Generate TTS audio from text (disabled).
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use (optional)
            speed: Speech speed (optional)
            
        Returns:
            None - audio generation disabled
        """
        print("ℹ️  Audio generation is disabled")
        return None
    
    def get_available_voices(self) -> list:
        """Get list of available voices (disabled).
        
        Returns:
            Empty list - no voices available
        """
        return []
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate if audio file exists (disabled).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            False - audio validation disabled
        """
        return False
    
    def cleanup_temp_audio(self, keep_latest: int = 5):
        """Clean up old temporary audio files (disabled).
        
        Args:
            keep_latest: Number of latest files to keep
        """
        print("ℹ️  Audio cleanup is disabled")
    
    def get_audio_info(self, audio_path: str) -> dict:
        """Get basic audio file information (disabled).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Error message indicating audio is disabled
        """
        return {"error": "Audio processing is disabled"} 