#!/bin/bash

# Script to run Chloe AI API server

echo "🚀 Starting Chloe AI API Server..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Activated virtual environment"
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Installed dependencies"
fi

# Run the API server
echo "📡 Starting API server on http://localhost:8000"
uvicorn api.main_api:app --host 0.0.0.0 --port 8000 --reload

echo "✅ Chloe AI API Server stopped"