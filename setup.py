#!/usr/bin/env python3
"""
Setup script for the simplified soccer video analysis app.
"""

import os
import shutil
import sys


def create_directories():
    """Create required directories."""
    dirs = ["video_gallery", "temp_audio", "processed_videos"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}")


def copy_env_file():
    """Copy environment file if it exists."""
    if os.path.exists("env.example"):
        if not os.path.exists(".env"):
            shutil.copy("env.example", ".env")
            print("✓ Created .env file from env.example")
            print("  Please update .env with your API keys")
        else:
            print("✓ .env file already exists")


def copy_gallery_videos():
    """Copy existing videos to gallery."""
    if os.path.exists("video_gallery") and os.listdir("video_gallery"):
        print("✓ Video gallery already has content")
        return
    
    # Look for videos in common locations
    video_sources = [".", "videos", "samples"]
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    for source_dir in video_sources:
        if not os.path.exists(source_dir):
            continue
            
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(video_extensions):
                source_path = os.path.join(source_dir, filename)
                dest_path = os.path.join("video_gallery", filename)
                
                if not os.path.exists(dest_path):
                    shutil.copy2(source_path, dest_path)
                    print(f"✓ Copied video: {filename}")


def install_dependencies():
    """Install dependencies."""
    print("\nInstalling dependencies...")
    os.system(f"{sys.executable} -m pip install -r simple_requirements.txt")


def main():
    """Main setup function."""
    print("🚀 Setting up simplified soccer video analysis app...")
    
    create_directories()
    copy_env_file()
    copy_gallery_videos()
    
    print("\n📦 Installing dependencies...")
    install_dependencies()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Update your .env file with API keys")
    print("2. Run: python simple_app.py")
    print("3. Open http://localhost:7860 in your browser")


if __name__ == "__main__":
    main() 