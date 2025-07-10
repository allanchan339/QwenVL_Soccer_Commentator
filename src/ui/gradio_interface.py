"""Gradio user interface for the soccer video analysis application."""

import gradio as gr
from typing import Optional

from ..utils.video_utils import load_gallery_videos
from ..services.soccer_pipeline import SoccerAnalysisPipeline


class SoccerVideoInterface:
    """Gradio interface for soccer video analysis."""
    
    def __init__(self):
        """Initialize the interface with the processing pipeline."""
        self.pipeline = SoccerAnalysisPipeline()
    
    def select_gallery_video(self, evt: gr.SelectData) -> Optional[str]:
        """Handle gallery video selection.
        
        Args:
            evt: Gradio SelectData event
            
        Returns:
            Selected video path or None
        """
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
    
    def refresh_gallery(self):
        """Refresh the video gallery."""
        return load_gallery_videos()
    
    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface.
        
        Returns:
            Configured Gradio Blocks interface
        """
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Soccer Video Analysis")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Left side - Video input and controls
                    video_input = gr.Video(
                        interactive=True,
                        height=400
                    )
                    
                    upload_button = gr.Button("Process Video", variant="primary")
                    
                    # Add gallery below the video input
                    gr.Markdown("### Video Gallery")
                    with gr.Column():
                        gallery = gr.Gallery(
                            show_label=True,
                            elem_id="gallery",
                            allow_preview=False,
                            value=load_gallery_videos()
                        )
                        refresh_button = gr.Button("🔄 Refresh Gallery")
                    
                    refresh_button.click(
                        fn=self.refresh_gallery,
                        outputs=[gallery]
                    )
                    
                    # Add click event for gallery items
                    gallery.select(
                        fn=self.select_gallery_video,
                        outputs=[video_input]
                    )
                
                with gr.Column(scale=1):
                    # Right side - Results
                    processed_video_output = gr.Video(
                        label="Processed Video with Commentary",
                        height=300
                    )
                    audio_output = gr.Audio(
                        label="Generated Commentary Audio",
                        type="filepath"
                    )
                    output_text = gr.Textbox(
                        label="Generated Commentary",
                        placeholder="Commentary will appear here...",
                        lines=5
                    )
                    
            # Event handler
            upload_button.click(
                fn=self.pipeline.process_video,
                inputs=[video_input],
                outputs=[processed_video_output, audio_output, output_text]
            )
            
        return demo 