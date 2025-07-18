# app_all.py
import argparse
import sys
import gradio as gr
from SoComm.models import load_all_models
from SoComm.inpainting import inference, debug_inpainting
from SoComm.tts import tts_generate
from SoComm.utils import fast_check_ffmpeg, check_video

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_path", type=str, default=r"ffmpeg-master-latest-win64-gpl-shared\bin", help="Path to ffmpeg executable")
parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
parser.add_argument("--share", action="store_true", help="Create a public link")
parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
args = parser.parse_args()

# Load models and device
(device, vae, unet, pe, weight_dtype, audio_processor, whisper, timesteps) = load_all_models(args.use_float16)

# Check ffmpeg and add to PATH
if not fast_check_ffmpeg():
    print(f"Adding ffmpeg to PATH: {args.ffmpeg_path}")
    path_separator = ';' if sys.platform == 'win32' else ':'
    import os
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

# Solve asynchronous IO issues on Windows
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Gradio UI construction
css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height: 576px}"""
def tts_to_audio(text, voice):
    audio_path = tts_generate(text, voice)
    return audio_path

def create_gradio_ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """<div align='center'> <h1>Soccer Commentary Generation</h1> \
                        <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                        </br>\
                        Allan\
                        <br>\
                        CIS\
                    </h2> \
                    """
        )
        with gr.Row():
            with gr.Column():
                tts_text = gr.Textbox(label="Text for TTS", placeholder="Enter text to synthesize audio", lines=2)
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
                    value="en-US-AriaNeural"
                )
                tts_btn = gr.Button("TTS")
                audio = gr.Audio(label="Driving Audio", type="filepath")
                video = gr.Video(label="Reference Video",sources=['upload'])
                bbox_shift = gr.Number(label="BBox_shift value, px", value=0)
                extra_margin = gr.Slider(label="Extra Margin", minimum=0, maximum=40, value=10, step=1)
                parsing_mode = gr.Radio(label="Parsing Mode", choices=["jaw", "raw"], value="jaw")
                left_cheek_width = gr.Slider(label="Left Cheek Width", minimum=20, maximum=160, value=90, step=5)
                right_cheek_width = gr.Slider(label="Right Cheek Width", minimum=20, maximum=160, value=90, step=5)
                bbox_shift_scale = gr.Textbox(label="'left_cheek_width' and 'right_cheek_width' parameters determine the range of left and right cheeks editing when parsing model is 'jaw'. The 'extra_margin' parameter determines the movement range of the jaw. Users can freely adjust these three parameters to obtain better inpainting results.")
                with gr.Row():
                    debug_btn = gr.Button("1. Test Inpainting ")
                    btn = gr.Button("2. Generate")
            with gr.Column():
                debug_image = gr.Image(label="Test Inpainting Result (First Frame)")
                debug_info = gr.Textbox(label="Parameter Information", lines=5)
                out1 = gr.Video()
        video.change(
            fn=check_video, inputs=[video], outputs=[video]
        )
        btn.click(
            fn=lambda audio, video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width: inference(
                audio, video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width,
                device=device, vae=vae, unet=unet, pe=pe, weight_dtype=weight_dtype, audio_processor=audio_processor, whisper=whisper, timesteps=timesteps
            ),
            inputs=[audio, video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[out1, bbox_shift_scale]
        )
        debug_btn.click(
            fn=lambda video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width: debug_inpainting(
                video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width,
                device=device, vae=vae, unet=unet, pe=pe, weight_dtype=weight_dtype, timesteps=timesteps
            ),
            inputs=[video, bbox_shift, extra_margin, parsing_mode, left_cheek_width, right_cheek_width],
            outputs=[debug_image, debug_info]
        )
        tts_btn.click(
            fn=tts_to_audio,
            inputs=[tts_text, tts_voice],
            outputs=[audio]
        )
    return demo

demo = create_gradio_ui()
demo.queue().launch(
    share=True,
    debug=True,
    server_name=args.ip,
    # server_port=args.port
)
