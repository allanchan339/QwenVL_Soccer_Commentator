# This Python file uses the following encoding: utf-8

import json
import subprocess
import time
from typing import Iterator
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials from environment variables
MINIMAX_GROUP_ID = os.getenv('MINIMAX_GROUP_ID')
MINIMAX_API_KEY = os.getenv('MINIMAX_API_KEY')

# Default file format
DEFAULT_FILE_FORMAT = 'mp3'


def build_tts_stream_headers() -> dict:
    if not MINIMAX_API_KEY:
        raise ValueError("MINIMAX_API_KEY not found in environment variables.")
    headers = {
        'accept': 'application/json, text/plain, */*',
        'content-type': 'application/json',
        'authorization': "Bearer " + MINIMAX_API_KEY,
    }
    return headers


def build_tts_stream_body(text: str, voice_id: str = "male-qn-qingse", speed: float = 1.0, vol: float = 1.0, pitch: int = 0, sample_rate: int = 32000, bitrate: int = 128000, audio_format: str = DEFAULT_FILE_FORMAT, channel: int = 1) -> str:
    body = {
        "model": "speech-02-turbo",
        "text": text,
        "stream": False, # Changed to non-stream mode
        "voice_setting": {
            "voice_id": voice_id,
            "speed": speed,
            "vol": vol,
            "pitch": pitch
        },
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate": bitrate,
            "format": audio_format,
            "channel": channel
        }
    }
    return json.dumps(body)


def call_tts_api_for_audio(text_to_speak: str) -> bytes | None:
    """
    Calls Minimax TTS API in non-stream mode and returns the complete audio data.
    Returns audio bytes if successful, None otherwise.
    """
    if not MINIMAX_GROUP_ID:
        raise ValueError("MINIMAX_GROUP_ID not found in environment variables.")
    
    tts_url = f"https://api.minimaxi.chat/v1/t2a_v2?GroupId={MINIMAX_GROUP_ID}"
    tts_headers = build_tts_stream_headers()
    tts_body = build_tts_stream_body(text_to_speak, audio_format=DEFAULT_FILE_FORMAT)

    try:
        response = requests.post(tts_url, headers=tts_headers, data=tts_body, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # In non-stream mode, we get a complete JSON response
        data = response.json()
        
        if "data" in data and "audio" in data["data"]:
            # The audio data is hex encoded
            audio_hex = data["data"]["audio"]
            if audio_hex:
                return bytes.fromhex(audio_hex)
            else:
                print("No audio data found in response")
                return None
        else:
            print("Unexpected response format from Minimax TTS API")
            print(f"Response: {data}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Minimax TTS API request failed: {e}")
        raise # Re-raise the exception to be handled by the caller
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        return None
    except Exception as e:
        print(f"Error processing API response: {e}")
        return None

def generate_minimax_tts(text_to_speak: str, output_path: str) -> str | None:
    """
    Generates TTS audio using Minimax API and saves it to output_path.
    Returns the output_path if successful, None otherwise.
    """
    if not MINIMAX_GROUP_ID or not MINIMAX_API_KEY:
        print("Minimax API credentials (MINIMAX_GROUP_ID, MINIMAX_API_KEY) not set in environment.")
        return None

    try:
        audio_data = call_tts_api_for_audio(text_to_speak)
        
        if not audio_data:
            print("No audio data received from Minimax TTS API")
            return None
        
        # Ensure the directory for the output path exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'wb') as file:
            file.write(audio_data)
        
        print(f"Minimax TTS audio saved to: {output_path}")
        return output_path
    except ValueError as ve: # Catch specific errors from missing credentials
        print(f"Configuration error for Minimax TTS: {ve}")
        return None
    except Exception as e:
        print(f"Error during Minimax TTS generation or saving: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    # Ensure MINIMAX_GROUP_ID and MINIMAX_API_KEY are set in your .env file or environment
    
    test_text = "你好，欢迎使用Minimax语音合成服务。这是一个测试句子。"
    timestamp = int(time.time())
    example_output_filename = f'example_tts_output_{timestamp}.{DEFAULT_FILE_FORMAT}'
    
    # Create a dummy .env if it doesn't exist for the example to run, or remind the user
    if not (MINIMAX_GROUP_ID and MINIMAX_API_KEY):
        print("Please set MINIMAX_GROUP_ID and MINIMAX_API_KEY in your .env file to run the example.")
    else:
        print(f"Attempting to generate TTS for: \"{test_text}\"")
        output_file = generate_minimax_tts(test_text, example_output_filename)
        if output_file:
            print(f"Test TTS audio successfully saved to: {output_file}")
            # You can try playing it with mpv or other audio player
            # Example: subprocess.run(["mpv", output_file])
        else:
            print("Failed to generate test TTS audio.") 