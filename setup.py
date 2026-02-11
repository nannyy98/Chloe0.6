#!/usr/bin/env python3
"""
Setup script for Chloe AI
Installs dependencies and initializes the project
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    
    try:
        # Install pip packages from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        "data",
        "models",
        "logs"
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        path.mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def initialize_git():
    """Initialize git repository if not already initialized"""
    if not Path(".git").exists():
        try:
            subprocess.check_call(["git", "init"])
            subprocess.check_call(["git", "add", "."])
            print("✅ Initialized git repository")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Git initialization failed: {e}")

def create_env_file():
    """Create a sample .env file"""
    env_content = """# Chloe AI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_MODEL=llama2  # or whatever model you want to use locally
DATA_PATH=./data
MODEL_PATH=./models
LOG_LEVEL=INFO
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("📄 Created .env file with configuration template")

def main():
    """Main setup function"""
    print("🚀 Setting up Chloe AI project...")
    
    # Change to the project directory
    os.chdir(Path(__file__).parent)
    
    # Run setup steps
    create_directories()
    initialize_git()
    install_dependencies()
    create_env_file()
    
    print("\n✅ Chloe AI setup completed!")
    print("\n📋 Next steps:")
    print("1. Set up your API keys in the .env file")
    print("2. Run 'python main.py' to start the application")
    print("3. Or run 'uvicorn api.main_api:app --reload' to start the API server")

if __name__ == "__main__":
    main()