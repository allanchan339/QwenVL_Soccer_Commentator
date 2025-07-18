"""Video analysis service using ModelScope Qwen2.5-VL model."""

from openai import OpenAI
from typing import Optional

from ..config import MODELSCOPE_SDK_TOKEN, QWEN_MODEL, MODELSCOPE_BASE_URL, TARGET_WORDS_PER_SECOND
from ..utils.video_utils import get_video_length


class VideoAnalyzer:
    """Service for analyzing soccer videos using Qwen2.5-VL model."""
    
    def __init__(self):
        """Initialize the video analyzer with OpenAI client."""
        self.client = None
        if MODELSCOPE_SDK_TOKEN:
            self.client = OpenAI(
                api_key=MODELSCOPE_SDK_TOKEN,
                base_url=MODELSCOPE_BASE_URL
            )
    
    def analyze_video(self, video_path: str) -> str:
        """Analyze video using ModelScope Qwen2.5-VL model.
        
        Args:
            video_path: Path to the video file to analyze
            
        Returns:
            Generated commentary text or error message
        """
        if not video_path:
            return "No video provided for analysis"
        
        if not self.client:
            return "Error: MODELSCOPE_SDK_TOKEN not configured. Please check your environment variables."
        
        try:
            # Get video length for commentary timing
            video_duration = get_video_length(video_path)
            target_words = int(video_duration * TARGET_WORDS_PER_SECOND - 1)
            
            # Prepare messages for the model
            video_messages = [
                {
                    "role": "system", 
                    "content": "You are a professional commentator for soccer. You are responsible for providing real-time commentary on the game."
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
                            "text": f"Please describe this game, FOCUS ON the action of players and THE BALL, explicitly for goals, assists, fouls, offsides, yellow/red cards, substitutions, and corner kicks. The video is {round(video_duration, 0)} seconds long and therefore the commentary should be around {target_words} words long. You should also have an engaging tone. SKIP all non commentary content"
                        },
                        {
                            "type": "text", 
                            "text": "用中文回答"
                        },
                    ]
                },
            ]
            
            # Call the model
            response = self.client.chat.completions.create(
                model=QWEN_MODEL,
                messages=video_messages,
                stream=False
            )
            
            commentary = response.choices[0].message.content
            return commentary
            
        except Exception as e:
            return f"Error analyzing video: {str(e)}" 