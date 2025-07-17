"""Text-to-Speech service using edge-tts."""

import os
import asyncio
from typing import Optional, Dict, List
import edge_tts

from ..config import TEMP_AUDIO_DIR
from ..utils.video_utils import ensure_directory_exists


class TTSService:
    """Service for generating speech from text using edge-tts."""
    
    def __init__(self, 
                 output_dir: str = TEMP_AUDIO_DIR,
                 edge_voice: str = "zh-CN-YunxiNeural"):
        """Initialize the TTS service.
        
        Args:
            output_dir: Directory to save generated audio files
            edge_voice: Voice to use for edge-tts (default: Chinese male voice)
        """
        self.output_dir = output_dir
        self.edge_voice = edge_voice
        ensure_directory_exists(self.output_dir)
    
    async def get_edge_voices(self) -> List[Dict[str, str]]:
        """Get available voices for edge-tts.
        
        Returns:
            List of voice dictionaries with Name, ShortName, Gender, and Locale
        """
        try:
            voices = await edge_tts.list_voices()
            return [
                {
                    "name": voice["Name"],
                    "short_name": voice["ShortName"], 
                    "gender": voice["Gender"],
                    "locale": voice["Locale"]
                }
                for voice in voices
            ]
        except Exception as e:
            print(f"Error getting edge-tts voices: {str(e)}")
            return []
    
    async def _generate_edge_tts(self, text: str, output_path: str, voice: Optional[str] = None) -> bool:
        """Generate TTS audio using edge-tts.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
            voice: Voice to use (defaults to self.edge_voice)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            current_voice = voice or self.edge_voice
            communicate = edge_tts.Communicate(text, current_voice)
            await communicate.save(output_path)
            print(f"Generated edge-tts audio: {output_path}")
            return True
        except Exception as e:
            print(f"Error generating edge-tts audio: {str(e)}")
            return False
    
    def _run_async(self, coro):
        """Run async coroutine in a way that works in both sync and async environments."""
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # If we're in an event loop, use create_task and run_until_complete via a new loop in a thread
                import threading
                result = []
                exc = []
                def run():
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result.append(new_loop.run_until_complete(coro))
                    except Exception as e:
                        exc.append(e)
                    finally:
                        new_loop.close()
                t = threading.Thread(target=run)
                t.start()
                t.join()
                if exc:
                    raise exc[0]
                return result[0]
            else:
                return asyncio.run(coro)
        except Exception as e:
            print(f"Error running async coroutine: {e}")
            return None
    
    def generate_audio(self, 
                      text: str, 
                      filename: Optional[str] = None,
                      voice: Optional[str] = None) -> Optional[str]:
        """Generate TTS audio from text using edge-tts.
        
        Args:
            text: Text to convert to speech
            filename: Optional filename for the audio file
            voice: Voice to use (defaults to self.edge_voice)
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not text or text.startswith("Error"):
            return None
        
        try:
            # Generate filename if not provided
            if filename is None:
                audio_filename = f"commentary_{abs(hash(text)) % 1000000}.wav"
            else:
                audio_filename = filename
            
            audio_path = os.path.join(self.output_dir, audio_filename)
            
            # Generate audio using edge-tts
            success = self._run_async(self._generate_edge_tts(text, audio_path, voice))
            
            return audio_path if success else None
                
        except Exception as e:
            print(f"Error generating TTS audio: {str(e)}")
            return None
    
    def set_edge_voice(self, voice: str):
        """Set the voice for edge-tts.
        
        Args:
            voice: Voice identifier (e.g., 'zh-CN-YunxiNeural')
        """
        self.edge_voice = voice
        print(f"Edge-TTS voice set to: {voice}") 