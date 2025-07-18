#!/usr/bin/env python3
"""
Merged Gradio demo for soccer video analysis and inpainting/TTS.
This single file combines the functionality of app_all.py and gradio_demo.py.
"""

# =================== IMPORTS & CONFIGURATION ===================
import os
import cv2
import subprocess
from typing import Optional, Tuple, List
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import sys
import asyncio

# --- Imports from SoComm for Inpainting/TTS ---
from SoComm.models import load_all_models
from SoComm.inpainting import inference, debug_inpainting
from SoComm.tts import tts_generate
from SoComm.utils import fast_check_ffmpeg, check_video

# --- Load environment variables for Video Analysis ---
load_dotenv()

# =================== MODEL & APP CONFIGURATION ===================

# --- Arguments for Inpainting/TTS (using hardcoded defaults) ---
class AppArgs:
    ffmpeg_path = r"ffmpeg-master-latest-win64-gpl-shared\bin"
    use_float16 = True
args = AppArgs()

# --- Configuration for Video Analysis ---
MODELSCOPE_SDK_TOKEN = os.getenv("MODELSCOPE_SDK_TOKEN")
QWEN_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
GALLERY_DIR = "video_gallery"
PROCESSED_VIDEOS_DIR = "processed_videos"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
TARGET_WORDS_PER_SECOND = 4

# --- Gradio Server Config ---
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 7869
DEFAULT_SHARE = True

# =================== MODEL LOADING & SETUP ===================

# --- Load Inpainting/TTS models ---
print("Loading SoComm models for Inpainting and TTS...")
(device, vae, unet, pe, weight_dtype, audio_processor, whisper, timesteps) = load_all_models(args.use_float16)
print("SoComm models loaded successfully.")

# --- Check ffmpeg and add to PATH ---
if not fast_check_ffmpeg():
    print(f"Adding ffmpeg to PATH: {args.ffmpeg_path}")
    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find or configure ffmpeg. Video generation may fail.")

# --- Solve asynchronous IO issues on Windows ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# =================== UTILITY FUNCTIONS ===================

def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_video_length(video_path: str) -> float:
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return 0 if video_fps == 0 else total_frames / video_fps
    except Exception as e:
        print(f"Error getting video length: {e}")
        return 0

def load_gallery_videos() -> List[Tuple[str, str]]:
    ensure_directory_exists(GALLERY_DIR)
    video_files = []
    for f in os.listdir(GALLERY_DIR):
        if f.lower().endswith(VIDEO_EXTENSIONS):
            file_path = os.path.join(GALLERY_DIR, f)
            video_files.append((file_path, f))
    return video_files

def tts_to_audio(text, voice):
    """Wrapper for TTS functionality."""
    audio_path = tts_generate(text, voice)
    return audio_path

# =================== SERVICES (for Video Analysis) ===================

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

class VideoProcessor:
    """Service for post-processing video (placeholder)."""
    def __init__(self, output_dir: str = PROCESSED_VIDEOS_DIR):
        self.output_dir = output_dir
        ensure_directory_exists(self.output_dir)
    
    def combine_video_and_audio(self, video_path: str, commentary_text: str) -> Tuple[Optional[str], str]:
        if not video_path: return None, "Please upload a video first."
        try:
            import shutil
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_analyzed.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            shutil.copy(video_path, output_path)
            return output_path, commentary_text
        except Exception as e:
            return video_path, f"Error processing video: {str(e)}"

class SoccerAnalysisPipeline:
    """Pipeline for the video analysis part of the demo."""
    def __init__(self):
        self.video_analyzer = VideoAnalyzer()
        self.video_processor = VideoProcessor()

    def process_video(self, video_path: str) -> Tuple[Optional[str], str]:
        if not video_path: return None, "No video provided."
        commentary = self.video_analyzer.analyze_video(video_path)
        if commentary.startswith("Error"): return video_path, commentary
        final_video_path, final_commentary = self.video_processor.combine_video_and_audio(video_path, commentary)
        return final_video_path, final_commentary

# =================== MERGED GRADIO UI ===================

def merged_interface():
    """Defines the integrated Gradio UI for the soccer demo."""
    css = """#output_vid {max-width: 1024px; max-height: 576px}"""
    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("# Soccer Commentary & Talking Head Video Generation Demo")
        with gr.Row(equal_height=False):
            # --- LEFT COLUMN: All user actions/inputs ---
            with gr.Column(scale=1, elem_id="left-col"):
                gr.Markdown("### 1. Select or Upload Soccer Clip")
                video_input = gr.Video(label="Upload or Select Soccer Clip", interactive=True)
                
                gallery = gr.Gallery(
                    label="Video Gallery",
                    show_label=True,
                    elem_id="video_gallery",
                    value=load_gallery_videos(),
                    columns=4, rows=2, object_fit="contain", allow_preview=False
                )
                
                analyze_btn = gr.Button("Analyze Video", variant="primary")
                
                commentary_tts_box = gr.Textbox(label="Generated Commentary / Text for TTS", placeholder="Commentary will appear here, or type your own text for TTS...", lines=6, interactive=True)
                
                # --- TTS MODULE ---
                gr.Markdown("### 2. Text-to-Speech (TTS)")
                tts_voice = gr.Dropdown(
                    label="TTS Voice",
                    choices=[
                        "en-US-AriaNeural",
                        "en-US-GuyNeural",
                        "zh-CN-XiaoxiaoNeural",
                        "zh-CN-YunxiNeural",
                        "zh-HK-HiuGaaiNeural",
                        "zh-HK-HiuMaanNeural",
                        "zh-HK-WanLungNeural"
                    ],
                    value="zh-HK-WanLungNeural"
                )
                tts_btn = gr.Button("Synthesize Audio")
                
                # --- AUDIO BLOCK: TTS Output & Inpainting Driving Audio ---
                gr.Markdown("### 3. Talking Head Video Generation")
                driving_audio = gr.Audio(label="Driving Audio for Inpainting (TTS output or upload your own)", type="filepath", interactive=True)
                
                # --- INPAINTING INPUTS ---
                inpainting_video_input = gr.Video(label="Reference Video for Inpainting (Upload Commentator Face Video)", sources=['upload'])
                
                with gr.Row():
                    debug_btn = gr.Button("1. Test Inpainting")
                    generate_btn = gr.Button("2. Generate Full Video", variant="primary")
                
                # Inpainting parameter controls (hidden by default, can be shown if needed)
                bbox_shift = gr.Number(label="BBox Shift (px)", value=0, visible=False)
                extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1, visible=False)
                parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw", visible=False)
                left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=90, step=5, visible=False)
                right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5, visible=False)

            # --- RIGHT COLUMN: All outputs/results ---
            with gr.Column(scale=1, elem_id="right-col"):
                gr.Markdown("### Test Inpainting Result (First Frame)")
                debug_output_image = gr.Image(label="Test Inpainting Result (First Frame)")
                gr.Markdown("### Parameter Information")
                debug_output_info = gr.Textbox(label="Parameter Information", lines=4)
                gr.Markdown("### Generated Video for Inpainting")
                inpainting_output_video = gr.Video(label="Generated Video")

        # =================== EVENT HANDLERS ===================
        pipeline = SoccerAnalysisPipeline()
        def process_analysis_video(video_path):
            # Only return the commentary, not the video path
            _, commentary = pipeline.process_video(video_path)
            return commentary
        def select_gallery_video(evt: gr.SelectData) -> Optional[str]:
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
        # Copy text from commentary to TTS
        def copy_text(val):
            return val
        # Gallery selection sets the video input
        gallery.select(fn=select_gallery_video, outputs=[video_input])
        # Analyze button triggers video analysis and commentary (only updates the merged box)
        analyze_btn.click(
            fn=process_analysis_video,
            inputs=[video_input],
            outputs=[commentary_tts_box]
        )
        # TTS: output to driving_audio
        tts_btn.click(
            fn=tts_to_audio,
            inputs=[commentary_tts_box, tts_voice],
            outputs=[driving_audio]
        )
        # Inpainting video input change (ffmpeg check)
        inpainting_video_input.change(fn=check_video, inputs=[inpainting_video_input], outputs=[inpainting_video_input])
        # Test Inpainting
        debug_btn.click(
            fn=lambda video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek: debug_inpainting(
                video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek,
                device=device, vae=vae, unet=unet, pe=pe, 
                weight_dtype=weight_dtype, timesteps=timesteps
            ),
            inputs=[inpainting_video_input, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[debug_output_image, debug_output_info]
        )
        # Generate Full Video
        generate_btn.click(
            fn=lambda audio, video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek: inference(
                audio, video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek,
                device=device, vae=vae, unet=unet, pe=pe, 
                weight_dtype=weight_dtype, audio_processor=audio_processor, 
                whisper=whisper, timesteps=timesteps
            ),
            inputs=[driving_audio, inpainting_video_input, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[inpainting_output_video, debug_output_info]
        )
        gr.HTML("""<style>#left-col, #right-col { padding: 1.5rem; } #video_gallery { min-height: 200px; }</style>""")
    return demo

# =================== MAIN EXECUTION ===================
def main():
    """Initializes and launches the merged Gradio application."""
    ensure_directory_exists(GALLERY_DIR)
    ensure_directory_exists(PROCESSED_VIDEOS_DIR)
    
    demo = merged_interface()
    demo.queue().launch(
        server_name=DEFAULT_SERVER_NAME,
        server_port=DEFAULT_SERVER_PORT,
        share=DEFAULT_SHARE,
        debug=True
    )

if __name__ == "__main__":
    main() 