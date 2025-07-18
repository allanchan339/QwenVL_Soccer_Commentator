#!/usr/bin/env python3
"""
Clean Gradio demo for soccer video analysis.

This file is now self-contained, with all logic previously in src/ moved here, and TTS functionality removed.
"""

# =================== CONFIGURATION ===================
import os
import cv2
import subprocess
from typing import Optional, Tuple, List
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables
load_dotenv()

# API Configuration
MODELSCOPE_SDK_TOKEN = os.getenv("MODELSCOPE_SDK_TOKEN")

# Model Configuration
QWEN_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"

# Directory Configuration
GALLERY_DIR = "video_gallery"
PROCESSED_VIDEOS_DIR = "processed_videos"

# Video Processing Configuration
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
TARGET_WORDS_PER_SECOND = 4

# UI Configuration
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 7869
DEFAULT_SHARE = True

# =================== UTILS ===================
def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_video_length(video_path: str) -> float:
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
    if not os.path.exists(GALLERY_DIR):
        os.makedirs(GALLERY_DIR)
        return []
    video_files = []
    for f in os.listdir(GALLERY_DIR):
        if f.lower().endswith(VIDEO_EXTENSIONS):
            file_path = os.path.join(GALLERY_DIR, f)
            video_files.append((file_path, f))
    return video_files

# =================== SERVICES ===================
# --- Video Analysis ---
class VideoAnalyzer:
    """Service for analyzing soccer videos using Qwen2.5-VL model."""
    def __init__(self):
        self.client = None
        if MODELSCOPE_SDK_TOKEN:
            self.client = OpenAI(
                api_key=MODELSCOPE_SDK_TOKEN,
                base_url=MODELSCOPE_BASE_URL
            )
    def analyze_video(self, video_path: str) -> str:
        if not video_path:
            return "No video provided for analysis"
        if not self.client:
            return "Error: MODELSCOPE_SDK_TOKEN not configured. Please check your environment variables."
        try:
            video_duration = get_video_length(video_path)
            target_words = int(video_duration * TARGET_WORDS_PER_SECOND - 1)
            video_messages = [
                {"role": "system", "content": "You are a professional commentator for soccer. You are responsible for providing real-time commentary on the game."},
                {"role": "user", "content": [
                    {"type": "video", "video": f"file://{video_path}", "max_pixels": 720 * 1280, "min_pixels": 256 * 256},
                    {"type": "text", "text": f"Please describe this game, FOCUS ON the action of players and THE BALL, explicitly for goals, assists, fouls, offsides, yellow/red cards, substitutions, and corner kicks. The video is {round(video_duration, 0)} seconds long and therefore the commentary should be around {target_words} words long. You should also have an engaging tone. SKIP all non commentary content, 用廣東話回答, 不要使用英文, MAKE SURE YOU ARE SPOTTING CORRECT ACTIONS BEFORE ANSWERING"},
                    {"type": "text", "text": ""},
                ]},
            ]
            response = self.client.chat.completions.create(
                model=QWEN_MODEL,
                messages=video_messages,
                stream=False
            )
            commentary = response.choices[0].message.content
            return commentary
        except Exception as e:
            return f"Error analyzing video: {str(e)}"

# --- Video Processor ---
class VideoProcessor:
    def __init__(self, output_dir: str = PROCESSED_VIDEOS_DIR):
        self.output_dir = output_dir
        ensure_directory_exists(self.output_dir)
    def combine_video_and_audio(self, video_path: str, commentary_text: str) -> Tuple[Optional[str], str]:
        if not video_path:
            return None, "Please upload and process a video first."
        try:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_commented.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            # For now, just copy the video as processed (no audio overlay)
            try:
                import shutil
                shutil.copy(video_path, output_path)
                return output_path, commentary_text
            except Exception as e:
                print(f"Error copying video: {str(e)}")
                return video_path, commentary_text
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return video_path, commentary_text

# --- Main Pipeline ---
class SoccerAnalysisPipeline:
    def __init__(self):
        self.video_analyzer = VideoAnalyzer()
        self.video_processor = VideoProcessor()
    def process_video(self, video_path: str) -> Tuple[Optional[str], str]:
        if not video_path:
            return None, "No video provided for processing"
        try:
            commentary = self.video_analyzer.analyze_video(video_path)
            if commentary.startswith("Error"):
                return video_path, commentary
            final_video_path, final_commentary = self.video_processor.combine_video_and_audio(
                video_path, commentary
            )
            return final_video_path, final_commentary
        except Exception as e:
            error_msg = f"Error in video processing pipeline: {str(e)}"
            return video_path, error_msg

# =================== UI ===================
class SoccerVideoInterface:
    def __init__(self):
        self.pipeline = SoccerAnalysisPipeline()
    def select_gallery_video(self, evt: gr.SelectData) -> Optional[str]:
        selected_data = evt.value
        if isinstance(selected_data, dict) and 'video' in selected_data and isinstance(selected_data['video'], dict) and 'path' in selected_data['video']:
            return selected_data['video']['path']
        elif isinstance(selected_data, tuple) and len(selected_data) > 0:
            return selected_data[0]
        elif isinstance(selected_data, str):
            return selected_data
        else:
            print(f"Warning: Unexpected data type or structure from gallery selection: {type(selected_data)}. Value: {selected_data}")
            return None
    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Soccer Video Analysis")
            # Top: Processed Video block spanning both columns
            processed_video_output = gr.Video(
                label="Processed Video",
                height=360
            )
            # Below: two columns for input/gallery and commentary
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        interactive=True,
                        height=400
                    )
                    upload_button = gr.Button("Process Video", variant="primary")
                    # gr.Markdown("### Video Gallery")
                    with gr.Column():
                        gallery = gr.Gallery(
                            show_label=True,
                            elem_id="Video gallery",
                            allow_preview=False,
                            value=load_gallery_videos(),
                            columns=4,
                            rows=2,
                            object_fit="contain"
                        )
                    gallery.select(
                        fn=self.select_gallery_video,
                        outputs=[video_input]
                    )
                with gr.Column(scale=1):
                    output_text = gr.Textbox(
                        label="Generated Commentary",
                        placeholder="Commentary will appear here...",
                        lines=5
                    )
            upload_button.click(
                fn=self._process_video_ui,
                inputs=[video_input],
                outputs=[processed_video_output, output_text]
            )
        return demo
    def _process_video_ui(self, video_path):
        # Wrapper for Gradio: returns (video, commentary)
        return self.pipeline.process_video(video_path)

# =================== MAIN ===================
def main():
    interface = SoccerVideoInterface()
    demo = interface.create_interface()
    demo.launch(
        server_name=DEFAULT_SERVER_NAME,
        server_port=DEFAULT_SERVER_PORT,
        share=DEFAULT_SHARE
    )

if __name__ == "__main__":
    main() 