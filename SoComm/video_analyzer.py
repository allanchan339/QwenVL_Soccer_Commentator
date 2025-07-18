"""
VideoAnalyzer: Service for analyzing soccer videos using Qwen2.5-VL model.
"""
import os
from openai import OpenAI
from .utils import get_video_length
import sys

MODELSCOPE_SDK_TOKEN = os.getenv("MODELSCOPE_SDK_TOKEN")
QWEN_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
TARGET_WORDS_PER_SECOND = 4

class VideoAnalyzer:
    """Service for analyzing soccer videos using Qwen2.5-VL model."""
    def __init__(self):
        self.client = None
        if MODELSCOPE_SDK_TOKEN:
            self.client = OpenAI(api_key=MODELSCOPE_SDK_TOKEN, base_url=MODELSCOPE_BASE_URL)
    
    def analyze_video(self, video_path: str) -> str:
        if not video_path: return "No video provided for analysis"
        if not self.client: return "Error: MODELSCOPE_SDK_TOKEN not configured. Please check your .env file."
        try:
            video_duration = get_video_length(video_path)
            target_words = int(video_duration * TARGET_WORDS_PER_SECOND)
            system_prompt = "You are a professional commentator for soccer. You are responsible for providing real-time commentary on the game."
            user_prompt = f"Please describe this game, FOCUS ON the action of players and THE BALL, explicitly for goals, assists, fouls, offsides, yellow/red cards, substitutions, and corner kicks. The video is {round(video_duration, 0)} seconds long and therefore the commentary should be around {target_words} words long. You should also have an engaging tone. SKIP all non commentary content, 用廣東話回答, 不要使用英文, MAKE SURE YOU ARE SPOTTING CORRECT ACTIONS BEFORE ANSWERING"
            video_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "video", "video": f"file://{video_path}"},
                    {"type": "text", "text": user_prompt},
                ]},
            ]
            response = self.client.chat.completions.create(model=QWEN_MODEL, messages=video_messages, stream=False)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing video: {str(e)}" 