#!/bin/bash
# postgres_setup.sh - Quick PostgreSQL setup for the chatbot application

echo "🚀 Setting up PostgreSQL for MultiDB Chatbot..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Stop and remove existing container if it exists
echo "🔄 Cleaning up existing PostgreSQL container..."
docker stop chatbot-postgres 2>/dev/null || true
docker rm chatbot-postgres 2>/dev/null || true

# Start new PostgreSQL container with your exact .env settings
echo "🐳 Starting PostgreSQL container..."
docker run --name chatbot-postgres \
  -e POSTGRES_DB=chatbot_app \
  -e POSTGRES_USER=chatbot_user \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  -d postgres:15

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker exec chatbot-postgres pg_isready -U chatbot_user -d chatbot_app > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ PostgreSQL failed to start within 30 seconds"
        docker logs chatbot-postgres
        exit 1
    fi
    sleep 1
done

# Test the connection
echo "🔍 Testing PostgreSQL connection..."
docker exec chatbot-postgres psql -U chatbot_user -d chatbot_app -c "SELECT version();" > /dev/null

if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL connection successful!"
    echo ""
    echo "📋 Database Details:"
    echo "   Host: localhost"
    echo "   Port: 5432"
    echo "   Database: chatbot_app"
    echo "   User: chatbot_user"
    echo "   Password: secure_password"
    echo ""
    echo "🚀 Ready to run: python test_run.py"
    echo "🚀 Or run app: python main.py"
    echo ""
    echo "🛑 To stop PostgreSQL later: docker stop chatbot-postgres"
    echo "🔄 To start again: docker start chatbot-postgres"
else
    echo "❌ PostgreSQL connection failed"
    docker logs chatbot-postgres
    exit 1
fi