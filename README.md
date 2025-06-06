# Soccer Video Analysis - 3-File Structure ⚽

A clean, modular soccer video analysis application with AI-generated commentary, organized into 3 focused files.

## 📁 Project Structure

```
├── simple_app.py           # Main entry point (20 lines)
├── video_processor.py      # Video analysis & processing (130 lines)
├── audio_processor.py      # Text-to-speech generation (140 lines)
├── ui.py                   # Gradio user interface (250 lines)
├── simple_requirements.txt # Dependencies
├── setup_simple.py        # Setup script
└── .env                   # API configuration
```

## 🎯 Module Responsibilities

### 1. `video_processor.py` - Video Processing
- **VideoProcessor class**: Handles video analysis and processing
- **Key Functions**:
  - `analyze_video()` - AI video analysis with Qwen model
  - `combine_video_audio()` - Merge video with generated audio using ffmpeg
  - `get_video_info()` - Extract video metadata

### 2. `audio_processor.py` - Audio Generation  
- **AudioProcessor class**: Manages text-to-speech operations
- **Key Functions**:
  - `generate_audio()` - Convert text to speech using Minimax API
  - `get_available_voices()` - List available voice options
  - `cleanup_temp_audio()` - Clean up temporary audio files

### 3. `ui.py` - User Interface
- **SoccerVideoUI class**: Handles all user interactions
- **Key Functions**:
  - `create_interface()` - Build Gradio web interface
  - `process_video_pipeline()` - Coordinate complete processing workflow
  - `load_gallery_videos()` - Manage video gallery

### 4. `simple_app.py` - Main Entry Point
- **Simple coordinator**: Initializes directories and launches UI
- **Minimal code**: Just 20 lines to start the application

## 🚀 Quick Start

```bash
# Setup (one command)
python setup_simple.py

# Configure API keys in .env
MODELSCOPE_SDK_TOKEN=your_token_here
MINIMAX_GROUP_ID=your_group_id_here  
MINIMAX_API_KEY=your_api_key_here

# Run the application
python simple_app.py
```

## ✨ Benefits of This Structure

### **Separation of Concerns**
- 🎥 Video processing isolated in one module
- 🎙️ Audio generation in dedicated module  
- 🖥️ UI logic cleanly separated
- 🚀 Main app just coordinates

### **Easy to Understand**
- Each file has a single, clear purpose
- Functions are logically grouped
- Dependencies are minimal between modules

### **Simple to Extend**
- Add new video processing features → Edit `video_processor.py`
- Add new audio options → Edit `audio_processor.py`
- Improve UI → Edit `ui.py`
- Change startup behavior → Edit `simple_app.py`

### **Easy to Test**
- Test video processing independently
- Test audio generation separately
- Test UI components in isolation

## 🔧 Features

- **🎬 Video Analysis**: Qwen AI model analyzes soccer videos
- **🎙️ Text-to-Speech**: Minimax API converts commentary to audio
- **🎥 Video Processing**: FFmpeg combines video with generated audio
- **📱 Modern UI**: Clean Gradio interface with progress tracking
- **📁 Gallery**: Browse and select from uploaded videos
- **⚙️ Voice Options**: Multiple voice choices and speed control

## 📊 Comparison with Previous Versions

| Structure | Files | Lines | Complexity | Maintainability |
|-----------|-------|-------|------------|----------------|
| Original Complex | 15+ files | 500+ lines | High | Complex |
| Single File | 1 file | 200 lines | Low | Simple |
| **3-File Structure** | **4 files** | **540 lines** | **Medium** | **Optimal** |

## 🛠 Dependencies

```
gradio>=4.0.0     # Web interface
openai>=1.0.0     # AI model integration  
python-dotenv>=1.0.0  # Environment variables
requests>=2.28.0  # HTTP requests
```

## 🎪 Usage Examples

### Using Individual Modules

```python
# Video processing only
from video_processor import VideoProcessor
processor = VideoProcessor()
commentary = processor.analyze_video("video.mp4")

# Audio generation only  
from audio_processor import AudioProcessor
audio = AudioProcessor()
audio_path = audio.generate_audio("Great goal!", voice_id="male-qiaoshu")

# UI only
from ui import SoccerVideoUI
ui = SoccerVideoUI()
demo = ui.create_interface()
```

### Full Pipeline
```python
# Complete application
python simple_app.py
# Opens http://localhost:7860 automatically
```

## 🎯 Perfect Balance

This 3-file structure provides the **perfect balance** between:
- ✅ **Simplicity** (not too many files)
- ✅ **Organization** (clear separation of concerns)  
- ✅ **Maintainability** (easy to modify and extend)
- ✅ **Readability** (each module has a clear purpose)

**Best for**: Teams that want clean code organization without over-engineering! 