#!/usr/bin/env python3
"""
Soccer Video Analysis App - Split into 3 Modules
Main entry point that coordinates video processing, audio generation, and UI.
"""

import os
from modules.ui import SoccerVideoUI


def main():
    """Main function to run the app."""
    print("🚀 Starting Soccer Video Analysis App...")
    print("📁 Ensuring directories exist...")
    
    # Ensure required directories exist
    directories = ["video_gallery", "temp_audio", "processed_videos"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("🎬 Initializing UI...")
    
    # Create and launch the UI
    ui = SoccerVideoUI()
    demo = ui.create_interface()
    
    print("🌐 Launching web interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )


if __name__ == "__main__":
    main() 