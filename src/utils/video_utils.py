"""Video utility functions for the soccer video analysis application."""

import os
import cv2
from typing import List, Tuple

from ..config import GALLERY_DIR, VIDEO_EXTENSIONS


def get_video_length(video_path: str) -> float:
    """Get video length in seconds using OpenCV.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Video duration in seconds, 0 if error
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if video_fps == 0:
            return 0
        else:
            return total_frames / video_fps
    except Exception as e:
        print(f"Error getting video length: {e}")
        return 0


def load_gallery_videos() -> List[Tuple[str, str]]:
    """Load videos from the gallery directory.
    
    Returns:
        List of tuples containing (file_path, filename)
    """
    if not os.path.exists(GALLERY_DIR):
        os.makedirs(GALLERY_DIR)
        return []
    
    video_files = []
    for f in os.listdir(GALLERY_DIR):
        if f.lower().endswith(VIDEO_EXTENSIONS):
            file_path = os.path.join(GALLERY_DIR, f)
            video_files.append((file_path, f))
    return video_files


def ensure_directory_exists(directory: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory) 