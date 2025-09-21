#!/bin/bash

# Setup Environment Configuration Script
# This script helps create a .env file from the template

echo "🔧 Setting up environment configuration..."

# Check if config.env exists
if [ ! -f "config.env" ]; then
    echo "❌ config.env template not found!"
    exit 1
fi

# Create .env from template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from config.env template..."
    cp config.env .env
    echo "✅ Created .env file"
else
    echo "ℹ️  .env file already exists"
fi

echo ""
echo "🔑 Please update the following values in your .env file:"
echo ""
echo "Required Spaces Configuration:"
echo "  SPACES_ACCESS_ID=your_actual_access_id"
echo "  SPACES_SECRET_KEY=your_actual_secret_key"
echo "  SPACES_BUCKET_NAME=your_bucket_name"
echo ""
echo "Optional Configuration:"
echo "  EMBEDDING_SIZE=1024 (or 512 for older models)"
echo "  API_PORT=8080 (or your preferred port)"
echo "  CONFIDENCE_THRESHOLD=0.5 (face detection threshold)"
echo "  SIMILARITY_THRESHOLD=0.4 (face recognition threshold)"
echo ""
echo "📝 Edit .env file:"
echo "  nano .env"
echo "  # or"
echo "  code .env"
echo ""
echo "🚀 After editing, start the server:"
echo "  python api.py --use-milvus"
echo ""
echo "✅ Setup complete!"
