"""
Video processing module for soccer video analysis.
Handles video analysis using AI models and video/audio combination.
"""

import os
import subprocess
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODELSCOPE_SDK_TOKEN = os.getenv("MODELSCOPE_SDK_TOKEN")
QWEN_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
PROCESSED_VIDEOS_DIR = "processed_videos"

# Ensure directory exists
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)


class VideoProcessor:
    """Handles video analysis and processing operations."""
    
    def __init__(self):
        """Initialize the video processor."""
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the OpenAI client for video analysis."""
        if MODELSCOPE_SDK_TOKEN:
            self.client = OpenAI(
                api_key=MODELSCOPE_SDK_TOKEN,
                base_url="https://api-inference.modelscope.cn/v1"
            )
    
    def analyze_video(self, video_path: str) -> str:
        """Analyze video and generate soccer commentary.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Generated commentary text or error message
        """
        if not video_path:
            return "No video provided"
        
        if not self.client:
            return "Error: ModelScope SDK token not configured"
        
        try:
            response = self.client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Analyze this soccer video and provide commentary in English. Focus on key moments, player actions, and game flow. Be descriptive and engaging."
                        },
                        {
                            "type": "video", 
                            "video": video_path
                        }
                    ]
                }],
                max_tokens=500,
                temperature=0.7
            )
            
            commentary = response.choices[0].message.content
            return commentary if commentary else "No commentary generated"
            
        except Exception as e:
            return f"Error analyzing video: {str(e)}"
    
    def combine_video_audio(self, video_path: str, audio_path: Optional[str]) -> str:
        """Combine video with audio using ffmpeg.
        
        Args:
            video_path: Path to the original video
            audio_path: Path to the audio file (optional)
            
        Returns:
            Path to the combined video or original video if no audio
        """
        if not audio_path or not os.path.exists(audio_path):
            return video_path
        
        try:
            # Generate output filename
            video_name = os.path.basename(video_path)
            name, ext = os.path.splitext(video_name)
            output_path = os.path.join(PROCESSED_VIDEOS_DIR, f"commented_{name}{ext}")
            
            # FFmpeg command to combine video and audio
            cmd = [
                "ffmpeg", 
                "-i", video_path,      # Input video
                "-i", audio_path,      # Input audio
                "-c:v", "copy",        # Copy video stream
                "-c:a", "aac",         # Encode audio as AAC
                "-shortest",           # Stop at shortest stream
                "-y",                  # Overwrite output file
                output_path
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return output_path
            else:
                print(f"FFmpeg error: {result.stderr}")
                return video_path
                
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg.")
            return video_path
        except Exception as e:
            print(f"Video combination error: {e}")
            return video_path
    
    def get_video_info(self, video_path: str) -> dict:
        """Get basic video information.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        
        try:
            # Get file size
            file_size = os.path.getsize(video_path)
            file_size_mb = round(file_size / (1024 * 1024), 2)
            
            return {
                "path": video_path,
                "filename": os.path.basename(video_path),
                "size_mb": file_size_mb,
                "exists": True
            }
        except Exception as e:
            return {"error": f"Could not get video info: {str(e)}"} 