import subprocess
import sys
import time
import os
import threading
import argparse

def start_server():
    """Start the FastAPI server."""
    print("Starting TTS Server...")
    subprocess.Popen([sys.executable, "server.py"])
    # Give the server some time to start up
    time.sleep(2)
    print("Server started!")

def start_client():
    """Start the Tkinter client application."""
    print("Starting TTS Client...")
    subprocess.Popen([sys.executable, "client.py"])
    print("Client started!")

def download_samples():
    """Download sample audio files."""
    print("Downloading sample files...")
    subprocess.call([sys.executable, "download_samples.py"])

def main():
    """Main function to parse arguments and start the system."""
    parser = argparse.ArgumentParser(description="CSM Text-to-Speech Client-Server System")
    parser.add_argument("--server-only", action="store_true", help="Start only the server")
    parser.add_argument("--client-only", action="store_true", help="Start only the client")
    parser.add_argument("--download-samples", action="store_true", help="Download sample audio files")
    args = parser.parse_args()

    if args.download_samples:
        download_samples()
        return

    if args.server_only:
        start_server()
    elif args.client_only:
        start_client()
    else:
        # Start both server and client
        start_server()
        start_client()

    print("\nCSM Text-to-Speech system is now running!")
    print("Press Ctrl+C to exit")
    
    try:
        # Keep the main process running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main() 