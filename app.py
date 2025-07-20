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
import time

# --- Imports from SoComm for Inpainting/TTS ---
from SoComm.models import load_all_models
from SoComm.realtime_inpainting import create_realtime_avatar, realtime_inference
from musetalk.utils.blending import get_image_blending
from SoComm.tts import tts_generate
from SoComm.utils import fast_check_ffmpeg, check_video, ensure_directory_exists, get_video_length, load_gallery_videos, tts_to_audio
from SoComm.video_analyzer import VideoAnalyzer

# --- Load environment variables for Video Analysis ---
load_dotenv(override=True)

# Assert that MODELSCOPE_SDK_TOKEN is properly loaded
MODELSCOPE_SDK_TOKEN = os.getenv("MODELSCOPE_SDK_TOKEN")
assert MODELSCOPE_SDK_TOKEN, "MODELSCOPE_SDK_TOKEN not found in environment variables. Please check your .env file and ensure python-dotenv is installed."

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
    parser.add_argument('--inpainting_batch_size', type=int, default=20, help='Batch size for inpainting')
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

def load_avatar_gallery(avatars_dir="results/v15/avatars"):
    """Load avatar gallery from the avatars directory."""
    gallery_items = []
    
    if not os.path.exists(avatars_dir):
        return gallery_items
    
    for avatar_name in os.listdir(avatars_dir):
        avatar_path = os.path.join(avatars_dir, avatar_name)
        if os.path.isdir(avatar_path):
            # Check if mask_coords.pkl exists (indicates avatar is ready for inference)
            mask_coords_path = os.path.join(avatar_path, "mask_coords.pkl")
            if not os.path.exists(mask_coords_path):
                continue  # Skip this avatar if mask_coords.pkl doesn't exist
            
            # Look for the first image in full_imgs directory
            full_imgs_path = os.path.join(avatar_path, "full_imgs")
            if os.path.exists(full_imgs_path):
                # Try to find 00000001.png first, then any other image
                target_image = os.path.join(full_imgs_path, "00000001.png")
                if not os.path.exists(target_image):
                    # Find any image file
                    for img_file in os.listdir(full_imgs_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            target_image = os.path.join(full_imgs_path, img_file)
                            break
                
                if os.path.exists(target_image):
                    gallery_items.append((target_image, avatar_name))
    
    return gallery_items

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
                
                # Navigation tabs for soccer clip input/gallery
                with gr.Tabs():
                    with gr.TabItem("1. Upload Video"):
                        video_input = gr.Video(label="Upload Soccer Clip", interactive=True)
                    
                    with gr.TabItem("2. Video Gallery"):
                        gr.Markdown("**Select from existing soccer clips:**")
                        gallery = gr.Gallery(
                            label="Video Gallery",
                            show_label=True,
                            elem_id="video_gallery",
                            value=load_gallery_videos(args.gallery_dir, args.video_extensions),
                            columns=4, rows=2, object_fit="contain", allow_preview=False
                        )
                        refresh_gallery_btn = gr.Button("🔄 Refresh Video Gallery", size="sm")
                
                # System prompt textbox for video analysis
                default_system_prompt = "You are a professional commentator for soccer. You are responsible for providing real-time commentary on the game.  Describe this game scene, FOCUS ON the action of players and THE BALL, explicitly for goals, assists, fouls, offsides, yellow/red cards, substitutions, and corner kicks. You should also have an engaging tone. SKIP all non commentary content, 用廣東話回答, 不要使用英文, MAKE SURE YOU ARE SPOTTING CORRECT ACTIONS BEFORE ANSWERING"
                system_prompt_box = gr.Textbox(label="System Prompt for Soccer Commentary AI", value=default_system_prompt, lines=3)
                analyze_btn = gr.Button("Analyze Video", variant="primary")
                
                commentary_tts_box = gr.Textbox(label="Generated Commentary / Text for TTS", placeholder="Commentary will appear here, or type your own text for TTS...", lines=4, interactive=True)
                
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
                
                # Progress tracking component
                progress_tracker = gr.Progress()
                
                # --- AUDIO BLOCK: TTS Output & Driving Audio for Video Generation ---
                gr.Markdown("### 3. Talking Head Video Generation")
                driving_audio = gr.Audio(label="Driving Audio (TTS output or upload your own)", type="filepath", interactive=True)
                
                gr.Markdown("### 3.1. Avatar Creation (MuseV)")
                
                # Navigation tabs for avatar creation/gallery
                with gr.Tabs():
                    with gr.TabItem("1. Avatar Creation"):
                        avatar_id_input = gr.Textbox(label="Avatar ID", placeholder="Enter unique avatar name", value="my_avatar")
                        avatar_video_input = gr.Video(label="Avatar Reference Video (Upload face video for avatar creation)", sources=['upload'])
                        
                        
                        create_avatar_btn = gr.Button("Create Avatar (Which consumes time)", variant="primary")
                    
                    with gr.TabItem("2. Avatar Gallery"):
                        gr.Markdown("**Select from existing avatars:**")
                        avatar_gallery = gr.Gallery(
                            label="Avatar Gallery",
                            show_label=True,
                            elem_id="avatar_gallery",
                            value=load_avatar_gallery(),
                            columns=3, rows=2, object_fit="contain", allow_preview=False
                        )
                        refresh_avatar_gallery_btn = gr.Button("🔄 Refresh Avatar Gallery", size="sm")
                
                    realtime_generate_btn = gr.Button("Generate Talking Head", variant="primary")

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
                
                # Parameter Information as dropdown/accordion
                with gr.Accordion("Running Information", open=True):
                    debug_output_info = gr.Textbox(label="Console Output", lines=4)
                


        # =================== EVENT HANDLERS ===================
        video_analyzer = VideoAnalyzer(
            qwen_model=args.qwen_model,
            modelscope_base_url=args.modelscope_base_url,
            target_words_per_second=args.target_words_per_second,
            modelscope_token=MODELSCOPE_SDK_TOKEN
        )
        
        # Helper functions for real-time inpainting
        
        def process_analysis_video(video_path, system_prompt, progress=gr.Progress(track_tqdm=True)):
            if not video_path:
                gr.Info("❌ No video provided. Please upload or select a video first.")
                return "No video provided. Please upload or select a video first."
            
            try:
                gr.Info("🔄 Starting video analysis... This may take a few moments.")
                commentary = video_analyzer.analyze_video(video_path, system_prompt=system_prompt)
                gr.Info("✅ Video analysis completed successfully!")
                return commentary
            except Exception as e:
                gr.Info(f"❌ Error analyzing video: {str(e)}")
                return f"Error analyzing video: {str(e)}"
        def select_gallery_video(evt: gr.SelectData) -> Optional[str]:
            selected_data = evt.value
            if isinstance(selected_data, dict) and 'video' in selected_data and isinstance(selected_data['video'], dict) and 'path' in selected_data['video']:
                video_path = selected_data['video']['path']
                gr.Info(f"✅ Selected video: {os.path.basename(video_path)}")
                return video_path
            elif isinstance(selected_data, tuple) and len(selected_data) > 0:
                video_path = selected_data[0]
                gr.Info(f"✅ Selected video: {os.path.basename(video_path)}")
                return video_path
            elif isinstance(selected_data, str):
                gr.Info(f"✅ Selected video: {os.path.basename(selected_data)}")
                return selected_data
            else:
                print(f"Warning: Unexpected data type or structure from gallery selection: {type(selected_data)}. Value: {selected_data}")
                gr.Info("❌ Failed to select video. Please try again.")
                return None
        
        def select_avatar_from_gallery(evt: gr.SelectData) -> Tuple[str, str]:
            """Handle avatar gallery selection to set avatar ID and show preview."""
            selected_data = evt.value
            if isinstance(selected_data, dict) and 'caption' in selected_data:
                avatar_name = selected_data['caption']
                gr.Info(f"✅ Avatar '{avatar_name}' selected successfully!")
                return avatar_name, f"Selected avatar: {avatar_name}"
            elif isinstance(selected_data, tuple) and len(selected_data) >= 2:
                image_path, avatar_name = selected_data[0], selected_data[1]
                gr.Info(f"✅ Avatar '{avatar_name}' selected successfully!")
                return avatar_name, f"Selected avatar: {avatar_name}"
            elif isinstance(selected_data, str):
                # Extract avatar name from path
                avatar_name = os.path.basename(os.path.dirname(os.path.dirname(selected_data)))
                gr.Info(f"✅ Avatar '{avatar_name}' selected successfully!")
                return avatar_name, f"Selected avatar: {avatar_name}"
            else:
                print(f"Warning: Unexpected data type from avatar gallery selection: {type(selected_data)}. Value: {selected_data}")
                gr.Info("❌ Failed to select avatar. Please try again.")
                return "", "No avatar selected"
        

        # Copy text from commentary to TTS
        def copy_text(val):
            return val
        # Gallery selection sets the video input
        gallery.select(fn=select_gallery_video, outputs=[video_input])
        
        # Refresh video gallery
        refresh_gallery_btn.click(fn=load_gallery_videos, inputs=[gr.State(args.gallery_dir), gr.State(args.video_extensions)], outputs=[gallery])
        

        
        # Avatar gallery selection sets the avatar ID
        avatar_gallery.select(fn=select_avatar_from_gallery, outputs=[avatar_id_input, debug_output_info])
        
        # Refresh avatar gallery
        refresh_avatar_gallery_btn.click(fn=load_avatar_gallery, outputs=[avatar_gallery])
        # Analyze button triggers video analysis and commentary (only updates the merged box)
        analyze_btn.click(
            fn=process_analysis_video,
            inputs=[video_input, system_prompt_box],
            outputs=[commentary_tts_box]
        )
        # TTS: output to driving_audio
        def tts_with_feedback(text, voice):
            if not text or not text.strip():
                gr.Info("❌ No text provided for TTS. Please enter text or analyze a video first.")
                return None
            try:
                gr.Info("🔄 Generating speech... Please wait.")
                audio_path = tts_to_audio(text, voice)
                gr.Info("✅ Speech generated successfully!")
                return audio_path
            except Exception as e:
                gr.Info(f"❌ Error generating speech: {str(e)}")
                return None
        
        tts_btn.click(
            fn=tts_with_feedback,
            inputs=[commentary_tts_box, tts_voice],
            outputs=[driving_audio]
        )
        # Avatar video input change (ffmpeg check)
        avatar_video_input.change(fn=check_video, inputs=[avatar_video_input], outputs=[avatar_video_input])
        # Generate Full Video (using real-time inference)
        # generate_btn.click(
        #     fn=lambda audio, video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek: generate_standard_video(
        #         audio, video, bbox_s, extra_m, parsing_m, l_cheek, r_cheek,
        #         device=device, vae=vae, unet=unet, pe=pe, 
        #         weight_dtype=weight_dtype, audio_processor=audio_processor, 
        #         whisper=whisper, timesteps=timesteps,
        #         result_dir=args.inpainting_result_dir,
        #         fps=args.inpainting_fps,
        #         batch_size=args.inpainting_batch_size,
        #         output_vid_name=args.inpainting_output_vid_name,
        #         version=args.inpainting_version
        #     ),
        #     inputs=[driving_audio, avatar_video_input, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
        #     outputs=[inpainting_output_video, debug_output_info]
        # )
        
        # --- REAL-TIME AVATAR EVENT HANDLERS ---
        
        def create_avatar(avatar_id, video_path, bbox_s, extra_m, parsing_m, l_cheek, r_cheek, progress=gr.Progress(track_tqdm=True)):
            """Create a real-time avatar."""
            if not avatar_id or not avatar_id.strip():
                gr.Info("Error: Please enter an avatar ID")
                return "Error: Please enter an avatar ID"
            if not video_path:
                gr.Info("Error: Please upload a reference video")
                return "Error: Please upload a reference video"
            
            # Check if avatar already exists
            avatar_dir = os.path.join("results", "v15", "avatars", avatar_id.strip())
            if os.path.exists(avatar_dir):
                gr.Warning(f"Avatar ID '{avatar_id.strip()}' already exists! Please choose a different name.")
                return f"❌ Avatar ID '{avatar_id.strip()}' already exists! Please rename and try again."
            
            try:
                progress(0, desc="Starting avatar creation...")
                start_time = time.time()
                
                progress(0.1, desc="Loading models and preparing avatar...")
                avatar = create_realtime_avatar(
                    avatar_id=avatar_id.strip(),
                    video_path=video_path,
                    bbox_shift=bbox_s,
                    batch_size=args.inpainting_batch_size,
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
                
                progress(1.0, desc="Avatar creation completed!")
                total_time = time.time() - start_time
                
                # Create detailed parameter information
                param_info = f"""
                Avatar '{avatar_id}' created successfully! Ready for real-time generation.

                PARAMETERS USED:
                • Avatar ID: {avatar_id.strip()}
                • Video File: {os.path.basename(video_path)}
                • BBox Shift: {bbox_s}px
                • Extra Margin: {extra_m}px  
                • Parsing Mode: {parsing_m}
                • Left Cheek Width: {l_cheek}px
                • Right Cheek Width: {r_cheek}px
                • Model Version: {args.inpainting_version}
                • Batch Size: {args.inpainting_batch_size}

                PROCESSING STATS:
                • Avatar Creation Time: {total_time:.2f}s
                • Status: Ready for real-time generation
                """
                
                gr.Info(f"Avatar '{avatar_id}' created successfully! Ready for real-time generation.")
                return param_info
            except Exception as e:
                gr.Info(f"Error creating avatar: {str(e)}")
                return f"❌ Error creating avatar: {str(e)}"
        
        def generate_realtime_video(avatar_id, audio_path, bbox_s, extra_m, parsing_m, l_cheek, r_cheek, progress=gr.Progress(track_tqdm=True)):
            """Generate real-time video using existing avatar."""
            if not avatar_id or not avatar_id.strip():
                return None, "Error: Please enter an avatar ID"
            if not audio_path:
                return None, "Error: Please provide audio for generation"
            
            try:
                progress(0, desc="Starting video generation...")
                
                progress(0.2, desc="Loading avatar and processing audio...")
                result = realtime_inference(
                    avatar_id=avatar_id.strip(),
                    audio_path=audio_path,
                    video_path="",  # Not used for real-time inference
                    bbox_shift=bbox_s,
                    batch_size=args.inpainting_batch_size,
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
                
                progress(1.0, desc="Video generation completed!")
                
                # Extract timing information from result
                processing_time = result.get('processing_time', 0)
                frame_count = result.get('frame_count', 0)
                fps_achieved = result.get('fps_achieved', 0)
                output_video = result.get('output_video', None)
                
                # Create detailed parameter information
                param_info = f"""Real-time video generated successfully!

PARAMETERS USED:
• Avatar ID: {avatar_id.strip()}
• BBox Shift: {bbox_s}px
• Extra Margin: {extra_m}px  
• Parsing Mode: {parsing_m}
• Left Cheek Width: {l_cheek}px
• Right Cheek Width: {r_cheek}px
• Model Version: {args.inpainting_version}
• Target FPS: {args.inpainting_fps}
• Batch Size: {args.inpainting_batch_size}

PROCESSING STATS:
• Total Processing Time: {processing_time:.2f}s
• Frame Count: {frame_count}
• Achieved FPS: {fps_achieved:.2f}
• Audio File: {os.path.basename(audio_path)}
• Output: {output_video if output_video else 'Generated successfully'}"""
                
                return output_video, param_info
            except Exception as e:
                return None, f"❌ Error generating video: {str(e)}"
        
        # Create Avatar button
        create_avatar_btn.click(
            fn=create_avatar,
            inputs=[avatar_id_input, avatar_video_input, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[debug_output_info]
        )
        
        # Generate Real-Time Video button
        realtime_generate_btn.click(
            fn=generate_realtime_video,
            inputs=[avatar_id_input, driving_audio, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[realtime_output_video, debug_output_info]
        )
        gr.HTML("""<style>#left-col, #right-col { padding: 1.5rem; } #video_gallery { min-height: 200px; } #avatar_gallery { min-height: 200px; margin-top: 1rem; }</style>""")
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