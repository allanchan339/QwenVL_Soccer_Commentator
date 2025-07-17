"""Main pipeline service for soccer video analysis workflow."""

from typing import Tuple, Optional

from .video_analysis import VideoAnalyzer
from .tts_service import TTSService
from .video_processor import VideoProcessor
from ..config import DEFAULT_EDGE_VOICE


class SoccerAnalysisPipeline:
    """Main pipeline for processing soccer videos with commentary generation."""
    
    def __init__(self, edge_voice: str = DEFAULT_EDGE_VOICE):
        """Initialize the pipeline with all required services.
        
        Args:
            edge_voice: Voice to use for edge-tts
        """
        self.video_analyzer = VideoAnalyzer()
        self.tts_service = TTSService(edge_voice=edge_voice)
        self.video_processor = VideoProcessor()
    
    def set_tts_voice(self, voice: str):
        """Set the TTS voice for edge-tts.
        
        Args:
            voice: Voice identifier for edge-tts
        """
        self.tts_service.set_edge_voice(voice)
    
    def process_video(self, 
                     video_path: str, 
                     voice: Optional[str] = None) -> Tuple[Optional[str], Optional[str], str]:
        """Process a soccer video through the complete analysis pipeline.
        
        Args:
            video_path: Path to the video file to process
            voice: Optional voice override for edge-tts
            
        Returns:
            Tuple of (processed_video_path, audio_path, commentary_text)
        """
        if not video_path:
            return None, None, "No video provided for processing"
        
        try:
            # Step 1: Analyze video to generate commentary
            commentary = self.video_analyzer.analyze_video(video_path)
            
            if commentary.startswith("Error"):
                return video_path, None, commentary
            
            # Step 2: Generate TTS audio from commentary
            audio_path = self.tts_service.generate_audio(commentary, voice=voice)
            
            # Step 3: Combine video with audio
            final_video_path, final_commentary = self.video_processor.combine_video_and_audio(
                video_path, audio_path, commentary
            )
            
            return final_video_path, audio_path, final_commentary
            
        except Exception as e:
            error_msg = f"Error in video processing pipeline: {str(e)}"
            return video_path, None, error_msg 