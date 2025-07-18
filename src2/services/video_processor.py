"""Video processing service for combining video with audio commentary."""

import os
import subprocess
from typing import Tuple, Optional

from ..config import PROCESSED_VIDEOS_DIR
from ..utils.video_utils import ensure_directory_exists


class VideoProcessor:
    """Service for processing and combining video with audio."""
    
    def __init__(self, output_dir: str = PROCESSED_VIDEOS_DIR):
        """Initialize the video processor.
        
        Args:
            output_dir: Directory to save processed videos
        """
        self.output_dir = output_dir
        ensure_directory_exists(self.output_dir)
    
    def combine_video_and_audio(
        self, 
        video_path: str, 
        audio_path: Optional[str], 
        commentary_text: str
    ) -> Tuple[Optional[str], str]:
        """Combine video with audio commentary using ffmpeg.
        
        Args:
            video_path: Path to the original video file
            audio_path: Path to the audio file (can be None)
            commentary_text: The commentary text for reference
            
        Returns:
            Tuple of (output_video_path, commentary_text)
        """
        if not video_path:
            return None, "Please upload and process a video first."
        
        try:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_commented.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            if audio_path and os.path.exists(audio_path):
                # Use ffmpeg to combine video and audio
                ffmpeg_command = [
                    "ffmpeg", "-y",  # -y to overwrite output file
                    "-i", video_path,  # Input video
                    "-i", audio_path,  # Input audio
                    "-c:v", "copy",  # Copy video codec
                    "-c:a", "aac",   # Use AAC audio codec
                    "-map", "0:v:0", # Map video from first input
                    "-map", "1:a:0", # Map audio from second input
                    "-shortest",     # End when shortest stream ends
                    output_path
                ]
                
                try:
                    subprocess.run(ffmpeg_command, check=True, capture_output=True)
                    return output_path, commentary_text
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("ffmpeg not available or failed, returning original video")
                    return video_path, commentary_text
            else:
                # If no audio file, just return original video with commentary text
                return video_path, commentary_text
                
        except Exception as e:
            print(f"Error combining video and audio: {str(e)}")
            return video_path, commentary_text 