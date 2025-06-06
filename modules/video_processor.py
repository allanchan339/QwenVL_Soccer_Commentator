"""
Video processing module for soccer video analysis.
Handles video analysis using AI models with Chinese commentary.
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
    
    def get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Video duration in seconds, or 30.0 as default
        """
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            print(f"Could not get video duration: {e}")
        
        # Default duration if unable to determine
        return 30.0
    
    def analyze_video(self, video_path: str) -> str:
        """Analyze video and generate Chinese soccer commentary.
        
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
            # Get video duration and calculate target words
            video_duration = self.get_video_duration(video_path)
            target_words = int(video_duration * 2.5)  # Reduced from 4 to 2.5 words per second
            max_tokens_limit = min(max(target_words * 2, 100), 600)  # Dynamic max_tokens based on target
            
            print(f"Video duration: {video_duration}s, Target words: {target_words}, Max tokens: {max_tokens_limit}")
            
            # Prepare the specific message format
            video_messages = [
                {
                    "role": "system", 
                    "content": "You are a professional commentator for soccer. You are responsible for providing real-time commentary on the game. IMPORTANT: Keep your commentary concise and within the specified word limit."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{video_path}",
                            "max_pixels": 720 * 1280,
                            "min_pixels": 256 * 256
                        },
                        {
                            "type": "text",
                            "text": f"Please describe this game, FOCUS ON the action of players and THE BALL, explicitly for goals, assists, fouls, offsides, yellow/red cards, substitutions, and corner kicks. The video is {round(video_duration, 0)} seconds long and therefore the commentary MUST be EXACTLY around {target_words} words long - no more, no less. You should also have an engaging tone. SKIP all non commentary content. BE CONCISE."
                        },
                        {
                            "type": "text", 
                            "text": "用中文回答，字数要严格控制在指定范围内"
                        },
                    ]
                },
            ]
            
            response = self.client.chat.completions.create(
                model=QWEN_MODEL,
                messages=video_messages,
                max_tokens=max_tokens_limit*2,  # Dynamic based on target words
                temperature=0.7
            )
            
            commentary = response.choices[0].message.content
            return commentary if commentary else "No commentary generated"
            
        except Exception as e:
            return f"Error analyzing video: {str(e)}"
    
    def combine_video_audio(self, video_path: str, audio_path: Optional[str]) -> str:
        """Combine video with audio using ffmpeg (currently not used).
        
        Args:
            video_path: Path to the original video
            audio_path: Path to the audio file (optional)
            
        Returns:
            Path to the original video (no processing needed)
        """
        # Since audio processing is disabled, just return original video
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
            
            # Get duration
            duration = self.get_video_duration(video_path)
            
            return {
                "path": video_path,
                "filename": os.path.basename(video_path),
                "size_mb": file_size_mb,
                "duration_seconds": duration,
                "exists": True
            }
        except Exception as e:
            return {"error": f"Could not get video info: {str(e)}"} 