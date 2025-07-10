#!/usr/bin/env python3
"""
Test script for TTS functionality.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.tts_service import TTSService

def test_tts():
    """Test the TTS service with a sample text."""
    print("Testing TTS service...")
    
    # Initialize TTS service
    tts = TTSService()
    
    # Test text
    test_text = "这是一个测试文本，用于验证TTS功能是否正常工作。"
    
    print(f"Generating audio for text: {test_text}")
    
    # Generate audio
    audio_path = tts.generate_audio(test_text, "test_output.wav")
    
    if audio_path:
        print(f"✅ TTS test successful! Audio saved to: {audio_path}")
        print(f"File exists: {os.path.exists(audio_path)}")
        print(f"File size: {os.path.getsize(audio_path)} bytes")
    else:
        print("❌ TTS test failed!")

if __name__ == "__main__":
    test_tts() 