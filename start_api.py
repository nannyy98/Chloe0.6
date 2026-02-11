#!/usr/bin/env python3
"""
Script to start the Chloe AI API server
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def start_api():
    """Start the FastAPI server"""
    print("📡 Starting Chloe AI API Server...")
    print("Access the API at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    
    try:
        # Start the uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api.main_api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")

def main():
    """Main function"""
    print("🚀 Starting Chloe AI API Server Setup")
    
    # Change to the script's directory
    os.chdir(Path(__file__).parent)
    
    # Install dependencies
    install_dependencies()
    
    # Start the API
    start_api()

if __name__ == "__main__":
    main()