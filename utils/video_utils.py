"""
Video utility functions for soccer video analysis.
"""

import os
from typing import List, Tuple

# Configuration
GALLERY_DIR = "video_gallery"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

# Ensure directory exists
os.makedirs(GALLERY_DIR, exist_ok=True)


def load_gallery_videos() -> List[Tuple[str, str]]:
    """Load videos from gallery directory for Gradio Gallery component.
    
    Returns:
        List of (video_path, filename) tuples
    """
    if not os.path.exists(GALLERY_DIR):
        return []
    
    videos = []
    try:
        for filename in os.listdir(GALLERY_DIR):
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                video_path = os.path.join(GALLERY_DIR, filename)
                if os.path.exists(video_path):
                    videos.append((video_path, filename))
    except Exception as e:
        print(f"Error loading gallery videos: {e}")
    
    return sorted(videos, key=lambda x: x[1])  # Sort by filename


def get_video_info(video_path: str) -> dict:
    """Get basic video file information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    if not video_path or not os.path.exists(video_path):
        return {"error": "Video file not found"}
    
    try:
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