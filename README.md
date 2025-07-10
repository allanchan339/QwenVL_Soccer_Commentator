# Soccer Video Analysis - Refactored Architecture

This project has been refactored to follow a clean, modular architecture with proper separation of concerns.

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── config.py                    # Centralized configuration
│   ├── services/                    # Business logic services
│   │   ├── __init__.py
│   │   ├── minimax_tts.py          # Minimax TTS API wrapper
│   │   ├── soccer_pipeline.py      # Main processing pipeline
│   │   ├── tts_service.py          # ModelScope TTS service
│   │   ├── video_analysis.py       # Video analysis with Qwen model
│   │   └── video_processor.py      # Video processing and combining
│   ├── ui/                          # User interface components
│   │   ├── __init__.py
│   │   └── gradio_interface.py     # Gradio UI implementation
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       └── video_utils.py          # Video-related helper functions
├── gradio_demo.py                  # Clean main entry point
├── flask_app.py                    # Flask application (if needed)
├── modelscope_app.py               # ModelScope app (if needed)
├── requirements.txt
└── .env                            # Environment variables
```

## Key Features of the Refactored Architecture

### 1. **Separation of Concerns**
- **Services**: Handle business logic (video analysis, TTS, video processing)
- **UI**: Manages user interface components
- **Utils**: Provides reusable utility functions
- **Config**: Centralizes all configuration settings

### 2. **Modular Design**
- Each service is self-contained and can be tested independently
- Clean interfaces between modules
- Easy to extend or replace individual components

### 3. **AI-Powered Features**
- **Video Analysis**: Uses Qwen2.5-VL model for soccer video analysis
- **Text-to-Speech**: Uses ModelScope Cantonese TTS for commentary audio
- **Video Processing**: Combines video with generated audio commentary

### 4. **User Interface**
- **Video Upload/Selection**: Upload videos or select from gallery
- **Audio Playback**: Play generated commentary audio separately
- **Video Gallery**: Pre-loaded sample soccer videos
- **Real-time Processing**: Process videos with AI-generated commentary

## How to Use

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python gradio_demo.py

# Test TTS functionality
python test_tts.py
```

## Benefits of This Refactoring

1. **Maintainability**: Code is organized logically and easy to navigate
2. **Testability**: Each module can be tested independently
3. **Reusability**: Services can be reused in different contexts
4. **Scalability**: Easy to add new features or replace components
5. **Clean Dependencies**: Clear import structure with minimal coupling

## Example Usage of Individual Services

```python
from src.services.soccer_pipeline import SoccerAnalysisPipeline
from src.services.video_analysis import VideoAnalyzer
from src.services.tts_service import TTSService

# Use the complete pipeline
pipeline = SoccerAnalysisPipeline()
result_video, audio_path, commentary = pipeline.process_video("path/to/video.mp4")

# Or use individual services
analyzer = VideoAnalyzer()
commentary = analyzer.analyze_video("path/to/video.mp4")

tts = TTSService()
audio_path = tts.generate_audio(commentary)
```

## TTS Integration

The application now includes full TTS functionality using ModelScope:
- **Model**: `speech_tts/speech_sambert-hifigan_tts_jiajia_Cantonese_16k`
- **Language**: Cantonese Chinese
- **Output**: WAV audio files
- **UI Integration**: Audio player in Gradio interface 