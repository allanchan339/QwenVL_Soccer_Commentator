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
import argparse

# --- Imports from SoComm for Inpainting/TTS ---
from SoComm.models import load_all_models
from SoComm.realtime_inpainting import create_realtime_avatar, realtime_inference
from musetalk.utils.blending import get_image_blending
from SoComm.tts import tts_generate
from SoComm.utils import fast_check_ffmpeg, check_video, ensure_directory_exists, get_video_length, load_gallery_videos, tts_to_audio
from SoComm.video_analyzer import VideoAnalyzer

# --- Load environment variables for Video Analysis ---
load_dotenv()

# =================== MODEL & APP CONFIGURATION ===================

def parse_args():
    parser = argparse.ArgumentParser(description="Soccer Video Analysis and Inpainting/TTS Demo")
    # General paths and server
    parser.add_argument('--ffmpeg_path', type=str, default=r"ffmpeg-master-latest-win64-gpl-shared\bin", help='Path to ffmpeg executable')
    parser.add_argument('--use_float16', action='store_true', default=True, help='Use float16 for models')
    parser.add_argument('--gallery_dir', type=str, default="video_gallery", help='Directory for video gallery')
    parser.add_argument('--processed_videos_dir', type=str, default="processed_videos", help='Directory for processed videos')
    parser.add_argument('--video_extensions', type=str, nargs='+', default=['.mp4', '.avi', '.mov', '.mkv'], help='Allowed video extensions')
    parser.add_argument('--default_server_name', type=str, default="0.0.0.0", help='Gradio server name')
    parser.add_argument('--default_server_port', type=int, default=7869, help='Gradio server port')
    parser.add_argument('--default_share', action='store_true', default=True, help='Gradio share option')
    # Model loading
    parser.add_argument('--unet_model_path', type=str, default="./models/musetalkV15/unet.pth", help='Path to UNet model weights')
    parser.add_argument('--vae_type', type=str, default="sd-vae", help='Type of VAE model')
    parser.add_argument('--unet_config', type=str, default="./models/musetalkV15/musetalk.json", help='Path to UNet config')
    parser.add_argument('--whisper_dir', type=str, default="./models/whisper", help='Directory for Whisper model')
    # Video analysis
    parser.add_argument('--qwen_model', type=str, default="Qwen/Qwen2.5-VL-72B-Instruct", help='Qwen model name')
    parser.add_argument('--modelscope_base_url', type=str, default="https://api-inference.modelscope.cn/v1", help='Modelscope API base URL')
    parser.add_argument('--target_words_per_second', type=int, default=4, help='Target words per second for commentary')
    # Inpainting/Generation parameters
    parser.add_argument('--inpainting_result_dir', type=str, default='./results/output', help='Directory for inpainting results')
    parser.add_argument('--debug_result_dir', type=str, default='./results/debug', help='Directory for debug inpainting results')
    parser.add_argument('--inpainting_fps', type=int, default=25, help='FPS for inpainting output')
    parser.add_argument('--inpainting_batch_size', type=int, default=8, help='Batch size for inpainting')
    parser.add_argument('--inpainting_output_vid_name', type=str, default='', help='Output video name for inpainting')
    parser.add_argument('--inpainting_use_saved_coord', action='store_true', default=False, help='Use saved coordinates for inpainting')
    parser.add_argument('--inpainting_audio_padding_left', type=int, default=2, help='Audio left padding for inpainting')
    parser.add_argument('--inpainting_audio_padding_right', type=int, default=2, help='Audio right padding for inpainting')
    parser.add_argument('--inpainting_version', type=str, default='v15', help='Inpainting model version')
    # Video processing/check_video
    parser.add_argument('--results_dir', type=str, default='./results', help='General results directory')
    parser.add_argument('--output_dir', type=str, default='./results/output', help='General output directory')
    parser.add_argument('--input_dir', type=str, default='./results/input', help='General input directory')
    return parser.parse_args()

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

args = parse_args()

# --- Load Inpainting/TTS models ---
print("Loading SoComm models for TTS & THG...")
(device, vae, unet, pe, weight_dtype, audio_processor, whisper, timesteps) = load_all_models(
    use_float16=args.use_float16,
    unet_model_path=args.unet_model_path,
    vae_type=args.vae_type,
    unet_config=args.unet_config,
    whisper_dir=args.whisper_dir
)
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

# =================== SERVICES (for Video Analysis) ===================

# =================== MERGED GRADIO UI ===================

def merged_interface(args):
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
                    value=load_gallery_videos(args.gallery_dir, args.video_extensions),
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
                
                # --- AUDIO BLOCK: TTS Output & Driving Audio for Video Generation ---
                gr.Markdown("### 3. Talking Head Video Generation")
                driving_audio = gr.Audio(label="Driving Audio (TTS output or upload your own)", type="filepath", interactive=True)
                
                # --- STANDARD VIDEO GENERATION ---
                gr.Markdown("#### Standard Video Generation (One-time processing)")
                
                generate_btn = gr.Button("Generate Standard Video", variant="primary")
                
                                # --- REAL-TIME AVATAR SECTION ---
                gr.Markdown("### 4. Real-Time Avatar (MuseV)")
                avatar_id_input = gr.Textbox(label="Avatar ID", placeholder="Enter unique avatar name", value="my_avatar")
                avatar_video_input = gr.Video(label="Avatar Reference Video (Upload face video for avatar creation)", sources=['upload'])
                
                with gr.Row():
                    create_avatar_btn = gr.Button("1. Create Avatar", variant="primary")
                    realtime_generate_btn = gr.Button("2. Generate Talking Head Video", variant="primary")
                
                # Inpainting parameter controls (hidden by default, can be shown if needed)
                # Move these to the bottom of the left column in a dropdown
                # bbox_shift = gr.Number(label="BBox Shift (px)", value=0, visible=False)
                # extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1, visible=False)
                # parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw", visible=False)
                # left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=90, step=5, visible=False)
                # right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5, visible=False)
                # Place at the bottom of the left column
                with gr.Accordion("Advanced Inpainting Parameters", open=False):
                    bbox_shift = gr.Number(label="BBox Shift (px)", value=0)
                    extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1)
                    parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw")
                    left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=90, step=5)
                    right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5)

            # --- RIGHT COLUMN: All outputs/results ---
            with gr.Column(scale=1, elem_id="right-col"):
                # Real-time avatar output
                gr.Markdown("### Real-Time Avatar Output")
                realtime_output_video = gr.Video(label="Real-Time Generated Video")
                
                # Standard inpainting output
                gr.Markdown("### Standard Inpainting Output")
                inpainting_output_video = gr.Video(label="Generated Video")
                

                
                # Parameter Information as dropdown/accordion
                with gr.Accordion("Parameter Information", open=False):
                    debug_output_info = gr.Textbox(label="Parameter Information", lines=4)
                
                # Avatar Status
                avatar_status = gr.Textbox(label="Avatar Status", value="No avatar created", interactive=False)

        # =================== EVENT HANDLERS ===================
        video_analyzer = VideoAnalyzer(
            qwen_model=args.qwen_model,
            modelscope_base_url=args.modelscope_base_url,
            target_words_per_second=args.target_words_per_second
        )
        
        # Helper functions for real-time inpainting
        
        def generate_standard_video(audio, video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek,
                                  device, vae, unet, pe, weight_dtype, audio_processor, 
                                  whisper, timesteps, result_dir, fps, batch_size, 
                                  output_vid_name, version):
            """Generate video using real-time inference approach."""
            if not audio or not video:
                return None, "Please provide both audio and video"
            
            try:
                # Create a temporary avatar for this video
                temp_avatar_id = f"temp_{os.path.basename(video).split('.')[0]}"
                
                result = realtime_inference(
                    avatar_id=temp_avatar_id,
                    audio_path=audio,
                    video_path=video,
                    bbox_shift=bbox_s,
                    batch_size=batch_size,
                    preparation=True,  # Create new avatar for this video
                    device=device,
                    vae=vae,
                    unet=unet,
                    pe=pe,
                    weight_dtype=weight_dtype,
                    audio_processor=audio_processor,
                    whisper=whisper,
                    timesteps=timesteps,
                    version=version,
                    extra_margin=extra_m,
                    parsing_mode=parsing_m,
                    left_cheek_width=l_cheek,
                    right_cheek_width=r_cheek,
                    fps=fps,
                    skip_save_images=False,
                    output_vid_name=output_vid_name if output_vid_name else f"standard_{temp_avatar_id}"
                )
                
                return result, f"Video generated successfully using real-time approach!"
                
            except Exception as e:
                return None, f"Error generating video: {str(e)}"
        def process_analysis_video(video_path):
            if not video_path:
                return "No video provided."
            commentary = video_analyzer.analyze_video(video_path)
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
        # Avatar video input change (ffmpeg check)
        avatar_video_input.change(fn=check_video, inputs=[avatar_video_input], outputs=[avatar_video_input])
        # Generate Full Video (using real-time inference)
        generate_btn.click(
            fn=lambda audio, video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek: generate_standard_video(
                audio, video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek,
                device=device, vae=vae, unet=unet, pe=pe, 
                weight_dtype=weight_dtype, audio_processor=audio_processor, 
                whisper=whisper, timesteps=timesteps,
                result_dir=args.inpainting_result_dir,
                fps=args.inpainting_fps,
                batch_size=args.inpainting_batch_size,
                output_vid_name=args.inpainting_output_vid_name,
                version=args.inpainting_version
            ),
            inputs=[driving_audio, avatar_video_input, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[inpainting_output_video, debug_output_info]
        )
        
        # --- REAL-TIME AVATAR EVENT HANDLERS ---
        
        def create_avatar(avatar_id, video_path, bbox_s, extra_m, parsing_m, l_cheek, r_cheek):
            """Create a real-time avatar."""
            if not avatar_id or not avatar_id.strip():
                return "Error: Please enter an avatar ID"
            if not video_path:
                return "Error: Please upload a reference video"
            
            try:
                avatar = create_realtime_avatar(
                    avatar_id=avatar_id.strip(),
                    video_path=video_path,
                    bbox_shift=bbox_s,
                    batch_size=20,
                    preparation=True,
                    device=device,
                    vae=vae,
                    unet=unet,
                    pe=pe,
                    weight_dtype=weight_dtype,
                    audio_processor=audio_processor,
                    whisper=whisper,
                    timesteps=timesteps,
                    version=args.inpainting_version,
                    extra_margin=extra_m,
                    parsing_mode=parsing_m,
                    left_cheek_width=l_cheek,
                    right_cheek_width=r_cheek
                )
                return f"Avatar '{avatar_id}' created successfully! Ready for real-time generation."
            except Exception as e:
                return f"Error creating avatar: {str(e)}"
        
        def generate_realtime_video(avatar_id, audio_path, bbox_s, extra_m, parsing_m, l_cheek, r_cheek):
            """Generate real-time video using existing avatar."""
            if not avatar_id or not avatar_id.strip():
                return None, "Error: Please enter an avatar ID"
            if not audio_path:
                return None, "Error: Please provide audio for generation"
            
            try:
                result = realtime_inference(
                    avatar_id=avatar_id.strip(),
                    audio_path=audio_path,
                    video_path="",  # Not used for real-time inference
                    bbox_shift=bbox_s,
                    batch_size=20,
                    preparation=False,  # Use existing avatar
                    device=device,
                    vae=vae,
                    unet=unet,
                    pe=pe,
                    weight_dtype=weight_dtype,
                    audio_processor=audio_processor,
                    whisper=whisper,
                    timesteps=timesteps,
                    version=args.inpainting_version,
                    extra_margin=extra_m,
                    parsing_mode=parsing_m,
                    left_cheek_width=l_cheek,
                    right_cheek_width=r_cheek,
                    fps=args.inpainting_fps,
                    skip_save_images=False,
                    output_vid_name=f"realtime_{avatar_id}"
                )
                return result, f"Real-time video generated successfully!"
            except Exception as e:
                return None, f"Error generating real-time video: {str(e)}"
        
        # Create Avatar button
        create_avatar_btn.click(
            fn=create_avatar,
            inputs=[avatar_id_input, avatar_video_input, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[avatar_status]
        )
        
        # Generate Real-Time Video button
        realtime_generate_btn.click(
            fn=generate_realtime_video,
            inputs=[avatar_id_input, driving_audio, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[realtime_output_video, debug_output_info]
        )
        gr.HTML("""<style>#left-col, #right-col { padding: 1.5rem; } #video_gallery { min-height: 200px; }</style>""")
    return demo

# =================== MAIN EXECUTION ===================
def main():
    """Initializes and launches the merged Gradio application."""
    ensure_directory_exists(args.gallery_dir)
    ensure_directory_exists(args.processed_videos_dir)
    
    demo = merged_interface(args)
    demo.queue().launch(
        server_name=args.default_server_name,
        server_port=args.default_server_port,
        share=args.default_share,
        debug=True
    )

if __name__ == "__main__":
    main() 