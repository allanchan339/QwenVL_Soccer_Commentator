import gradio as gr
import os
import tempfile

def load_gallery_videos():
    gallery_dir = "video_gallery"
    if not os.path.exists(gallery_dir):
        os.makedirs(gallery_dir)
        return []
    
    video_files = []
    for f in os.listdir(gallery_dir):
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            file_path = os.path.join(gallery_dir, f)
            video_files.append((file_path, f))  # Return tuple of (file_path, filename)
    return video_files

def process_video(video_path):
    # This is a placeholder for your video processing logic
    # You would typically call your soccer analysis model here
    commentary = analyze_video_with_llm(video_path)
    # Simulate TTS generation (placeholder)
    simulated_audio_path = generate_tts_audio(commentary)
    # Simulate combining video and audio (placeholder)
    # In a real scenario, this would output a new video file path
    # For now, we'll just return the original video path and the commentary
    final_video_path, final_commentary = combine_video_and_audio_final(video_path, simulated_audio_path, commentary)
    return final_video_path, final_commentary

def analyze_video_with_llm(video_path):
    # Placeholder for LLM video analysis
    # In a real implementation, this would involve calling an LLM
    # and generating descriptive commentary.
    base_name = os.path.basename(video_path) if video_path else "No video"
    return f"Detailed LLM analysis and commentary for video: {base_name}. The game was intense, with many close calls..."

def generate_tts_audio(commentary_text):
    # Placeholder for TTS generation
    # This would convert commentary_text to an audio file and return its path.
    # For now, we'll just simulate it.
    print(f"Simulating TTS generation for: {commentary_text[:50]}...")
    return "path/to/simulated_audio.mp3"

def combine_video_and_audio_final(original_video_path, audio_path, commentary_text):
    # Placeholder for combining video with new audio and displaying text
    # In a real implementation, you'd use libraries like OpenCV/moviepy
    # to merge the video with 'audio_path' and maybe overlay 'commentary_text'.
    # For this placeholder, we'll just return the original video path and the commentary.
    print(f"Simulating video/audio combination for: {original_video_path} and {audio_path}")
    if not original_video_path: # Handle case where no video is processed yet
        return None, "Please upload and process a video first."
    return original_video_path, commentary_text

def select_gallery_video(evt: gr.SelectData):
    # evt.value corresponds to the selected item from the gallery.
    # Based on the warning, evt.value can be a dictionary like:
    # {'video': {'path': 'filepath', ...}, 'caption': 'filename'}
    # It might also be a tuple (file_path, filename) or a direct string path in other cases.
    selected_data = evt.value

    if isinstance(selected_data, dict) and 'video' in selected_data and isinstance(selected_data['video'], dict) and 'path' in selected_data['video']:
        # Handle the dictionary structure: {'video': {'path': 'filepath', ...}, ...}
        return selected_data['video']['path']
    elif isinstance(selected_data, tuple) and len(selected_data) > 0:
        # Handle tuple: (file_path, filename)
        return selected_data[0]
    elif isinstance(selected_data, str):
        # Handle string: file_path
        return selected_data
    else:
        # Fallback for unexpected data type.
        print(f"Warning: Unexpected data type or structure from gallery selection: {type(selected_data)}. Value: {selected_data}")
        return None

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Soccer Video Analysis")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Left side - Video input and controls
                video_input = gr.Video(
                    # label="Upload Soccer Video",
                    interactive=True,
                    height=400
                )
                
                upload_button = gr.Button("Process Video", variant="primary")
                
                # Add gallery below the video input
                gr.Markdown("### Video Gallery")
                with gr.Column():
                    gallery = gr.Gallery(
                        # label="Available Videos",
                        show_label=True,
                        elem_id="gallery",
                        # columns=4,
                        # height=300,
                        allow_preview=False,
                        value=load_gallery_videos()
                    )
                    refresh_button = gr.Button("🔄 Refresh Gallery")
                
                def refresh_gallery():
                    return load_gallery_videos()
                
                refresh_button.click(
                    fn=refresh_gallery,
                    outputs=[gallery]
                )
                
                # Add click event for gallery items
                gallery.select(
                    fn=select_gallery_video,
                    outputs=[video_input]
                )
            
            with gr.Column(scale=1):
                # Right side - Results
                processed_video_output = gr.Video(
                    label="Processed Video with Commentary",
                    height=300, # Adjust height as needed
                    # interactive=False # Typically, output is not interactive
                )
                output_text = gr.Textbox(
                    label="Generated Commentary",
                    placeholder="Commentary will appear here...",
                    lines=5 # Adjust lines as needed
                )
                
        # Event handler
        upload_button.click(
            fn=process_video,
            inputs=[video_input],
            outputs=[processed_video_output, output_text]
        )
        
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
