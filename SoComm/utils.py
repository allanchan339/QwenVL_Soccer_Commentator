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