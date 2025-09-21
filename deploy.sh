#!/bin/bash

# Production Deployment Script for Face Recognition API
# This script helps deploy the API to production servers (Manual Deployment)

set -e

echo "🚀 Face Recognition API - Manual Production Deployment"
echo "===================================================="

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "📋 Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  Warning: .env file not found. Please create one with your configuration."
    echo "   You can copy from env.template as a template:"
    echo "   cp env.template .env"
    echo "   Then edit .env with your production values."
fi

# Configuration
DEPLOY_DIR="${DEPLOY_DIR:-/app/face-recognition}"
SERVICE_NAME="${SERVICE_NAME:-face-recognition-api}"
VENV_DIR="$DEPLOY_DIR/venv"

# Create deployment directory
echo "📁 Creating deployment directory..."
sudo mkdir -p $DEPLOY_DIR
sudo mkdir -p $DEPLOY_DIR/logs
sudo mkdir -p $DEPLOY_DIR/weights
sudo mkdir -p $DEPLOY_DIR/database
sudo mkdir -p $DEPLOY_DIR/assets

# Copy application files (excluding volumes, logs, cache)
echo "📋 Copying application files..."
rsync -av --exclude='volumes/' \
          --exclude='*.log' \
          --exclude='__pycache__/' \
          --exclude='.git/' \
          --exclude='venv/' \
          --exclude='.env' \
          ./ $DEPLOY_DIR/

# Set proper ownership
sudo chown -R $(whoami):$(whoami) $DEPLOY_DIR

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
cd $DEPLOY_DIR
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn

# Download model weights if not present
echo "⚖️  Checking model weights..."
if [ ! -f "$DEPLOY_DIR/weights/det_10g.onnx" ]; then
    echo "📥 Downloading model weights..."
    cd $DEPLOY_DIR
    bash download.sh
fi

# Setup production configuration
echo "⚙️  Setting up production configuration..."
if [ -f ".env" ]; then
    echo "📋 Copying .env file to deployment directory..."
    cp .env $DEPLOY_DIR/.env
else
    echo "⚠️  Warning: .env file not found in current directory."
    echo "   Please create a .env file with your configuration before deployment."
    echo "   You can copy from env.template as a template:"
    echo "   cp env.template .env"
    echo "   Then edit .env with your production values."
    exit 1
fi

# Create necessary directories for face database and assets
echo "📁 Creating necessary directories..."
mkdir -p $DEPLOY_DIR/database/face_database
mkdir -p $DEPLOY_DIR/assets/faces

# Create systemd service file
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Face Recognition API
After=network.target

[Service]
Type=exec
User=$(whoami)
Group=$(whoami)
WorkingDirectory=$DEPLOY_DIR
Environment=PATH=$VENV_DIR/bin
EnvironmentFile=$DEPLOY_DIR/.env
ExecStart=$VENV_DIR/bin/gunicorn --config gunicorn.conf.py wsgi:application
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create nginx configuration (optional)
echo "🌐 Creating nginx configuration..."
sudo tee /etc/nginx/sites-available/$SERVICE_NAME > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;  # Update with your domain
    
    client_max_body_size 20M;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_timeout 300s;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
        access_log off;
    }
}
EOF

# Enable nginx site (optional)
# sudo ln -sf /etc/nginx/sites-available/$SERVICE_NAME /etc/nginx/sites-enabled/
# sudo nginx -t && sudo systemctl reload nginx

# Reload systemd and start service
echo "🔄 Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo ""
echo "✅ Manual Deployment completed!"
echo ""
echo "📊 Service Status:"
sudo systemctl status $SERVICE_NAME --no-pager -l
echo ""
echo "🔧 Useful Commands:"
echo "  Start:   sudo systemctl start $SERVICE_NAME"
echo "  Stop:    sudo systemctl stop $SERVICE_NAME" 
echo "  Restart: sudo systemctl restart $SERVICE_NAME"
echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo "  Status:  sudo systemctl status $SERVICE_NAME"
echo ""
echo "🌐 API Endpoints:"
echo "  Health:  http://your-server:${API_PORT:-8080}/health"
echo "  Upload:  http://your-server:${API_PORT:-8080}/upload"
echo "  Recognize: http://your-server:${API_PORT:-8080}/recognize"
echo ""
echo "📁 Deployment Structure:"
echo "  App Directory: $DEPLOY_DIR"
echo "  Virtual Env:   $VENV_DIR"
echo "  Environment:   $DEPLOY_DIR/.env"
echo "  Logs:          $DEPLOY_DIR/logs/"
echo "  Weights:       $DEPLOY_DIR/weights/"
echo "  Database:      $DEPLOY_DIR/database/"
echo "  Assets:        $DEPLOY_DIR/assets/"
echo ""
echo "⚠️  Don't forget to:"
echo "  1. Verify .env file contains correct production credentials"
echo "  2. Configure firewall for port ${API_PORT:-8080}"
echo "  3. Setup SSL certificate for HTTPS"
echo "  4. Configure domain name in nginx"
echo "  5. Upload face images to $DEPLOY_DIR/assets/faces/ directory"
echo "  6. Initialize Milvus database if using vector storage"
