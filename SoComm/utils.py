# utils.py
"""
Shared utility functions for the app.
"""
import subprocess
import os

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

def check_video(video: str) -> str:
    import imageio
    import os
    import re
    if not isinstance(video, str):
        return video
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    output_file_name = "outputxxx_" + file_name
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results/output', exist_ok=True)
    os.makedirs('./results/input', exist_ok=True)
    output_video = os.path.join('./results/input', output_file_name)
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']
    frames = [im for im in reader]
    target_fps = 25
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L+1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target+1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])
    import imageio
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video

def ensure_directory_exists(directory_path: str):
    """
    Ensures a directory exists. If it doesn't, it creates it.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def get_video_length(video_path: str) -> float:
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return 0 if video_fps == 0 else total_frames / video_fps
    except Exception as e:
        print(f"Error getting video length: {e}")
        return 0

def load_gallery_videos() -> list:
    import os
    GALLERY_DIR = "video_gallery"
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
    ensure_directory_exists(GALLERY_DIR)
    video_files = []
    for f in os.listdir(GALLERY_DIR):
        if f.lower().endswith(VIDEO_EXTENSIONS):
            file_path = os.path.join(GALLERY_DIR, f)
            video_files.append((file_path, f))
    return video_files

def tts_to_audio(text, voice):
    from .tts import tts_generate
    audio_path = tts_generate(text, voice)
    return audio_path 