#!/bin/bash

# Milvus Setup Script for Podman
# This script sets up Milvus vector database using Podman

set -e

echo "🚀 Setting up Milvus Vector Database with Podman..."

# Check if podman is installed
if ! command -v podman &> /dev/null; then
    echo "❌ Podman is not installed. Please install Podman first."
    echo "   macOS: brew install podman"
    echo "   Linux: Follow instructions at https://podman.io/getting-started/installation"
    exit 1
fi

# Check if podman-compose is available
if ! command -v podman-compose &> /dev/null; then
    echo "📦 Installing podman-compose..."
    pip3 install podman-compose
fi

# Create volumes directory
echo "📁 Creating volume directories..."
mkdir -p ./volumes/etcd
mkdir -p ./volumes/minio
mkdir -p ./volumes/milvus

# Set proper permissions
chmod -R 755 ./volumes

echo "🐳 Starting Milvus services with Podman..."

# Start services using podman-compose
podman-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for Milvus to be ready..."
sleep 30

# Check if Milvus is running
echo "🔍 Checking Milvus health..."
for i in {1..10}; do
    if curl -f http://localhost:9091/healthz > /dev/null 2>&1; then
        echo "✅ Milvus is healthy!"
        break
    else
        echo "   Attempt $i/10: Milvus not ready yet, waiting..."
        sleep 10
    fi
    
    if [ $i -eq 10 ]; then
        echo "❌ Milvus failed to start properly"
        echo "   Check logs with: podman-compose logs milvus"
        exit 1
    fi
done

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install pymilvus

echo "🎉 Milvus setup complete!"
echo ""
echo "📊 Service URLs:"
echo "   Milvus:      http://localhost:19530"
echo "   Milvus Web:  http://localhost:9091"
echo "   MinIO:       http://localhost:9000"
echo "   MinIO Admin: http://localhost:9001"
echo ""
echo "🔧 Useful commands:"
echo "   Check status:  podman-compose ps"
echo "   View logs:     podman-compose logs milvus"
echo "   Stop services: podman-compose down"
echo "   Restart:       podman-compose restart"
echo ""
echo "🚀 You can now run the face recognition API with Milvus backend!"
