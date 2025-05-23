"""Configuration module for the soccer video analysis application."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
MODELSCOPE_SDK_TOKEN = os.getenv("MODELSCOPE_SDK_TOKEN")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")

# Model Configuration
QWEN_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"

# Directory Configuration
GALLERY_DIR = "video_gallery"
TEMP_AUDIO_DIR = "temp_audio"
PROCESSED_VIDEOS_DIR = "processed_videos"

# Video Processing Configuration
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
TARGET_WORDS_PER_SECOND = 4

# Audio Configuration
DEFAULT_AUDIO_FORMAT = 'mp3'

# UI Configuration
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 7860
DEFAULT_SHARE = True 