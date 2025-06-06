"""
Soccer analysis pipeline service.
Coordinates video processing and commentary generation.
"""

import os
import sys
from typing import Tuple, Optional
import gradio as gr

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor


class SoccerAnalysisPipeline:
    """Main pipeline for soccer video analysis and processing."""
    
    def __init__(self):
        """Initialize the pipeline with processors."""
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
    
    def process_video(self, video_path: str, progress=gr.Progress()) -> Tuple[Optional[str], str]:
        """Process video through the complete analysis pipeline.
        
        Args:
            video_path: Path to input video file
            progress: Gradio progress tracker (optional)
            
        Returns:
            Tuple of (processed_video_path, commentary_text)
        """
        if not video_path:
            return None, "❌ Please upload or select a video file"
        
        if not os.path.exists(video_path):
            return None, "❌ Video file not found"
        
        try:
            # Step 1: Video Analysis
            if progress:
                progress(0.1, desc="🔍 Analyzing video content...")
            
            commentary = self.video_processor.analyze_video(video_path)
            
            if commentary.startswith("Error"):
                return video_path, f"❌ {commentary}"
            
            if progress:
                progress(0.6, desc="✅ Analysis complete")
            
            # Step 2: Audio Generation (Currently disabled)
            if progress:
                progress(0.7, desc="ℹ️  Audio generation disabled - skipping...")
            
            # Since audio processing is disabled, return original video
            if progress:
                progress(1.0, desc="✅ Processing complete!")
            
            # Format the commentary output
            formatted_commentary = f"✅ **Commentary Generated:**\n\n{commentary}"
            
            return video_path, formatted_commentary
            
        except Exception as e:
            error_msg = f"❌ Processing error: {str(e)}"
            return video_path, error_msg
    
    def get_video_info(self, video_path: str) -> str:
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
        
        duration_text = ""
        if "duration_seconds" in info:
            duration_text = f" - {info['duration_seconds']:.1f}s"
        
        return f"📹 **{info['filename']}** ({info['size_mb']} MB{duration_text})" 