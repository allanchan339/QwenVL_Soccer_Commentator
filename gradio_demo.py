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
    return f"Processed video: {os.path.basename(video_path)}"

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
                output_text = gr.Textbox(
                    label="Analysis Results",
                    placeholder="Results will appear here...",
                    lines=10
                )
                
        # Event handler
        upload_button.click(
            fn=process_video,
            inputs=[video_input],
            outputs=[output_text]
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
