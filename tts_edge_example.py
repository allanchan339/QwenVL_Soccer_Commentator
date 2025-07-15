#!/usr/bin/env python3
"""
Example script demonstrating edge-tts integration in the TTS service.
"""

import os
import sys
import asyncio

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.tts_service import TTSService

async def example_edge_tts_usage():
    """Demonstrate edge-tts usage."""
    print("=== Edge-TTS Integration Example ===\n")
    
    # Initialize TTS service
    tts = TTSService()
    
    # Example 1: Basic text-to-speech
    print("1. Basic TTS generation:")
    text = "欢迎使用集成了edge-tts的足球视频分析系统！"
    audio_path = tts.generate_audio(text, "example_basic.wav")
    
    if audio_path:
        print(f"✅ Generated: {audio_path}")
    else:
        print("❌ Failed to generate audio")
    
    # Example 2: Different voices
    print("\n2. Using different voices:")
    voices_to_try = [
        ("zh-CN-YunxiNeural", "Male voice"),
        ("zh-CN-XiaoxiaoNeural", "Female voice")
    ]
    
    for voice, description in voices_to_try:
        print(f"   Testing {description} ({voice}):")
        audio_path = tts.generate_audio(
            "这是一个语音测试。", 
            f"example_{voice.split('-')[-1].lower()}.wav",
            voice=voice
        )
        if audio_path:
            print(f"   ✅ Generated: {audio_path}")
        else:
            print(f"   ❌ Failed with voice {voice}")
    
    # Example 3: Soccer commentary examples
    print("\n3. Soccer commentary examples:")
    commentary_examples = [
        "进球了！这是一个精彩的射门！",
        "裁判吹响了犯规的哨声。",
        "球员传球给队友，这是一个很好的配合。"
    ]
    
    for i, commentary in enumerate(commentary_examples):
        print(f"   Commentary {i+1}: {commentary}")
        audio_path = tts.generate_audio(commentary, f"example_commentary_{i+1}.wav")
        if audio_path:
            print(f"   ✅ Generated: {audio_path}")
        else:
            print(f"   ❌ Failed to generate commentary {i+1}")
    
    # Example 4: List available voices
    print("\n4. Available Chinese voices:")
    voices = await tts.get_edge_voices()
    chinese_voices = [v for v in voices if 'zh-CN' in v['locale']]
    
    print(f"   Found {len(chinese_voices)} Chinese voices:")
    for voice in chinese_voices[:3]:  # Show first 3
        print(f"   - {voice['short_name']}: {voice['name']} ({voice['gender']})")
    
    print("\n5. Cantonese (Hong Kong) voices:")
    cantonese_voices = [
        ("zh-HK-HiuGaaiNeural", "Cantonese Female 1"),
        ("zh-HK-HiuMaanNeural", "Cantonese Female 2"),
        ("zh-HK-WanLungNeural", "Cantonese Male")
    ]
    cantonese_text = "你好，呢度係廣東話測試。"  # "Hello, this is a Cantonese test."
    for voice, description in cantonese_voices:
        print(f"   Testing {description} ({voice}):")
        audio_path = tts.generate_audio(
            cantonese_text,
            f"example_{voice.split('-')[-1].lower()}_cantonese.wav",
            voice=voice
        )
        if audio_path:
            print(f"   ✅ Generated: {audio_path}")
        else:
            print(f"   ❌ Failed with voice {voice}")
    
    print(f"\n=== Example Complete ===")
    print("Generated audio files are saved in the temp_audio/ directory")

if __name__ == "__main__":
    asyncio.run(example_edge_tts_usage())
