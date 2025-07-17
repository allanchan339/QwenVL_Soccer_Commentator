#!/usr/bin/env python3
"""
Clean Gradio demo for soccer video analysis.

This file demonstrates the refactored architecture with proper separation of concerns.
"""

from src.ui.gradio_interface import SoccerVideoInterface
from src.config import DEFAULT_SERVER_NAME, DEFAULT_SERVER_PORT, DEFAULT_SHARE


def main():
    """Main function to create and launch the Gradio interface."""
    # Create the interface
    interface = SoccerVideoInterface()
    demo = interface.create_interface()
    
    # Launch the demo
    demo.launch(
        server_name=DEFAULT_SERVER_NAME,
        server_port=DEFAULT_SERVER_PORT,
        share=DEFAULT_SHARE
    )


if __name__ == "__main__":
    main() 