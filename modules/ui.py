"""
User Interface module for soccer video analysis.
Handles Gradio interface creation and user interactions.
"""

import os
import gradio as gr
from typing import Optional, Tuple, List
from modules.video_processor import VideoProcessor
from modules.audio_processor import AudioProcessor

# Configuration
GALLERY_DIR = "video_gallery"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

# Ensure directory exists
os.makedirs(GALLERY_DIR, exist_ok=True)


class SoccerVideoUI:
    """Gradio user interface for soccer video analysis."""
    
    def __init__(self):
        """Initialize the UI with processors."""
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
    
    def load_gallery_videos(self) -> List[Tuple[str, str]]:
        """Load videos from gallery directory.
        
        Returns:
            List of (video_path, filename) tuples
        """
        if not os.path.exists(GALLERY_DIR):
            return []
        
        videos = []
        try:
            for filename in os.listdir(GALLERY_DIR):
                if filename.lower().endswith(VIDEO_EXTENSIONS):
                    video_path = os.path.join(GALLERY_DIR, filename)
                    if os.path.exists(video_path):
                        videos.append((video_path, filename))
        except Exception as e:
            print(f"Error loading gallery videos: {e}")
        
        return sorted(videos, key=lambda x: x[1])  # Sort by filename
    
    def select_gallery_video(self, evt: gr.SelectData) -> Optional[str]:
        """Handle gallery video selection.
        
        Args:
            evt: Gradio SelectData event
            
        Returns:
            Selected video path or None
        """
        try:
            selected_data = evt.value
            
            if isinstance(selected_data, tuple) and len(selected_data) > 0:
                return selected_data[0]  # Return file path
            elif isinstance(selected_data, str):
                return selected_data
            elif isinstance(selected_data, dict):
                # Handle different gallery data structures
                if 'video' in selected_data and isinstance(selected_data['video'], dict):
                    return selected_data['video'].get('path')
                
        except Exception as e:
            print(f"Gallery selection error: {e}")
        
        return None
    
    def process_video_pipeline(self, video_path: str, progress=gr.Progress()) -> Tuple[Optional[str], str]:
        """Complete video processing pipeline with progress tracking.
        
        Args:
            video_path: Path to input video
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (processed_video_path, commentary_text)
        """
        if not video_path:
            return None, "❌ Please upload or select a video file"
        
        try:
            # Step 1: Video Analysis
            progress(0.1, desc="🔍 Analyzing video...")
            commentary = self.video_processor.analyze_video(video_path)
            
            if commentary.startswith("Error"):
                return video_path, f"❌ {commentary}"
            
            progress(0.4, desc="✅ Analysis complete")
            
            # Step 2: Audio Generation (Disabled)
            progress(0.5, desc="ℹ️  Audio generation disabled - skipping...")
            audio_path = None  # No audio processing
            
            progress(0.7, desc="✅ Analysis complete (audio disabled)")
            
            # Step 3: Return original video (no audio combination needed)
            progress(1.0, desc="✅ Processing complete!")
            final_video = video_path  # Return original video since no audio
            
            return final_video, f"✅ **Commentary Generated:**\n\n{commentary}"
            
        except Exception as e:
            error_msg = f"❌ Processing error: {str(e)}"
            return video_path, error_msg
    
    def refresh_gallery(self) -> List[Tuple[str, str]]:
        """Refresh the video gallery.
        
        Returns:
            Updated list of gallery videos
        """
        return self.load_gallery_videos()
    
    def get_video_info_display(self, video_path: str) -> str:
        """Get formatted video information for display.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Formatted video information string
        """
        if not video_path:
            return "No video selected"
        
        info = self.video_processor.get_video_info(video_path)
        
        if "error" in info:
            return f"⚠️ {info['error']}"
        
        return f"📹 **{info['filename']}** ({info['size_mb']} MB)"
    
    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface.
        
        Returns:
            Configured Gradio Blocks interface
        """
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .gallery-item {
            border-radius: 8px;
        }
        """
        
        with gr.Blocks(
            title="Soccer Video Analysis", 
            theme=gr.themes.Soft(),
            css=css
        ) as demo:
            
            # Header
            gr.Markdown(
                "# ⚽ Soccer Video Analysis\n"
                "Upload a soccer video to get AI-generated text commentary"
            )
            
            with gr.Row():
                # Left Column - Input
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="📹 Upload Video",
                        height=350,
                        interactive=True
                    )
                    
                    # Video info display
                    video_info = gr.Markdown(
                        value="No video selected",
                        elem_classes=["video-info"]
                    )
                    
                    # Process button
                    process_btn = gr.Button(
                        "🎬 Generate Commentary", 
                        variant="primary", 
                        size="lg",
                        scale=1
                    )
                    
                    # Gallery section
                    with gr.Group():
                        gr.Markdown("### 📁 Video Gallery")
                        gallery = gr.Gallery(
                            value=self.load_gallery_videos(),
                            show_label=False,
                            allow_preview=False,
                            columns=2,
                            rows=2,
                            height=200
                        )
                        
                        with gr.Row():
                            refresh_btn = gr.Button("🔄 Refresh", size="sm")
                            gallery_info = gr.Markdown(
                                f"📊 {len(self.load_gallery_videos())} videos in gallery",
                                elem_classes=["gallery-info"]
                            )
                
                # Right Column - Output
                with gr.Column(scale=1):
                    output_video = gr.Video(
                        label="📹 Original Video (unchanged)",
                        height=300
                    )
                    
                    commentary_output = gr.Markdown(
                        value="Commentary will appear here after processing...",
                        label="Generated Commentary",
                        elem_classes=["commentary-output"],
                        max_height=300
                    )
                    
                    # Processing status
                    with gr.Group():
                        gr.Markdown("### ℹ️ Processing Status")
                        gr.Markdown(
                            "- ✅ Video analysis enabled\n"
                            "- ❌ Audio generation disabled\n"
                            "- ℹ️ Videos will be returned with text commentary only"
                        )
            
            # Event Handlers
            
            # Update video info when video changes
            video_input.change(
                fn=self.get_video_info_display,
                inputs=[video_input],
                outputs=[video_info]
            )
            
            # Main processing
            process_btn.click(
                fn=self.process_video_pipeline,
                inputs=[video_input],
                outputs=[output_video, commentary_output],
                show_progress=True
            )
            
            # Gallery interactions
            gallery.select(
                fn=self.select_gallery_video,
                outputs=[video_input]
            )
            
            refresh_btn.click(
                fn=self.refresh_gallery,
                outputs=[gallery, gallery_info]
            )
            

            
            # Tips section
            with gr.Row():
                gr.Markdown(
                    "### 💡 Tips\n"
                    "- Upload soccer videos for best results\n"
                    "- Videos should be clear and show game action\n"
                    "- Processing takes 15-30 seconds for video analysis\n"
                    "- Commentary will appear as text only (audio disabled)"
                )
            
        return demo 