#!/bin/bash
# Unified Setup Script
# ====================
# Sets up the entire development environment

set -e

echo "🚀 RAG Chatbot Platform Setup"
echo "=============================="

# Check for dependencies
echo "🔍 Checking for dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required"
    exit 1
fi
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required"
    exit 1
fi
echo "✅ Dependencies found."

# Create and activate virtual environment
echo "📦 Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "✅ Virtual environment created."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed."

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d
echo "✅ Docker services started."

# Wait for ScyllaDB cluster to be fully healthy
./scripts/wait-for-scylla.sh

# Initialize databases
echo "🗄️ Initializing databases..."
python scripts/manage.py init-db --with-data
echo "✅ Databases initialized."

echo ""
echo "🎉 Setup complete! The environment is ready."
echo "   To start the API server, run: make run"
echo "   To run tests, run: make test"