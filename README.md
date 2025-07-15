# Soccer Video Analysis - Refactored Architecture

This project has been refactored to follow a clean, modular architecture with proper separation of concerns.

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── config.py                    # Centralized configuration
│   ├── services/                    # Business logic services
│   │   ├── __init__.py
│   │   ├── soccer_pipeline.py      # Main processing pipeline
│   │   ├── tts_service.py          # Edge-TTS service
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
- **Text-to-Speech**: Uses Microsoft Edge-TTS for commentary audio
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

## TTS Integration (Edge-TTS)

The application uses Microsoft's Edge-TTS for high-quality speech synthesis:

### Features:
- **Provider**: Microsoft Edge TTS
- **Languages**: Supports 100+ languages including Chinese
- **Voices**: Multiple Chinese voices (male/female)
- **Benefits**: Free, fast, high-quality, no API key needed
- **Default Voice**: `zh-CN-YunxiNeural` (Chinese male)

### Quick Setup:
```bash
# Install edge-tts (included in requirements.txt)
pip install edge-tts>=6.1.0

# Test the TTS integration
python example_edge_tts.py

# Run comprehensive TTS tests
python test_tts.py
```

### Usage Examples:

#### Basic Usage:
```python
from src.services.tts_service import TTSService

# Initialize TTS service
tts = TTSService()
audio_path = tts.generate_audio("这是一个测试文本")
```

#### Voice Customization:
```python
# Change default voice
tts.set_edge_voice("zh-CN-XiaoxiaoNeural")  # Female voice
audio_path = tts.generate_audio("女性语音测试")

# Or specify voice per generation
audio_path = tts.generate_audio(
    "临时使用不同语音", 
    voice="zh-CN-YunyeNeural"
)
```

#### Pipeline Integration:
```python
from src.services.soccer_pipeline import SoccerAnalysisPipeline

# Initialize pipeline
pipeline = SoccerAnalysisPipeline()

# Process video with specific voice
result_video, audio, commentary = pipeline.process_video(
    "video.mp4", 
    voice="zh-CN-XiaoxiaoNeural"
)
```

### Available Chinese Voices:
- `zh-CN-YunxiNeural` - Male (default)
- `zh-CN-XiaoxiaoNeural` - Female  
- `zh-CN-YunyeNeural` - Male
- `zh-CN-XiaoyiNeural` - Female
- And many more... (use `get_edge_voices()` to list all)

### Configuration:
The default voice can be configured in `src/config.py`:
```python
DEFAULT_EDGE_VOICE = "zh-CN-YunxiNeural"
``` 