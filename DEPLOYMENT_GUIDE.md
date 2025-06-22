# Naarad AI Assistant - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the enhanced Naarad AI Assistant with voice features, WebSocket real-time streaming, personalization, and analytics integration.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   External      │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   Services      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                 │    │                 │    │                 │
│ • Chat UI       │    │ • WebSocket     │    │ • OpenAI API    │
│ • Voice UI      │    │ • Voice Agent   │    │ • Supabase      │
│ • Analytics     │    │ • Analytics     │    │ • Monitoring    │
│ • Personalization│   │ • Personalization│   │ • Load Balancer │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04+ / CentOS 8+ / macOS 12+
- **CPU**: 4+ cores (8+ recommended for production)
- **RAM**: 8GB+ (16GB+ recommended for production)
- **Storage**: 50GB+ available space
- **Network**: Stable internet connection

### Software Requirements

- Python 3.8+
- Node.js 16+
- PostgreSQL 12+ (or Supabase)
- Redis 6+ (for caching)
- Nginx (for load balancing)

### API Keys and Services

- OpenAI API key
- Supabase credentials
- SSL certificates (for production)

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd naarad
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your configuration
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 4. Database Setup

```bash
# If using Supabase
python scripts/supabase_setup.py

# If using local PostgreSQL
python scripts/setup_database.py
```

## Configuration

### Environment Variables

#### Backend (.env)

```bash
# Application
APP_NAME=Naarad AI Assistant
APP_VERSION=2.0.0
ENVIRONMENT=production
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql://user:password@localhost/naarad
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Voice Processing
VOICE_ENABLED=true
SPEECH_RECOGNITION_MODEL=whisper-1
TEXT_TO_SPEECH_MODEL=tts-1

# WebSocket
WEBSOCKET_MAX_CONNECTIONS=1000
WEBSOCKET_HEARTBEAT_INTERVAL=30

# Analytics
ANALYTICS_ENABLED=true
MONITORING_ENABLED=true

# Security
SECRET_KEY=your_secret_key
ALLOWED_ORIGINS=["https://yourdomain.com"]

# SSL (Production)
SSL_ENABLED=true
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

#### Frontend (.env)

```bash
REACT_APP_API_URL=https://your-api-domain.com
REACT_APP_WEBSOCKET_URL=wss://your-api-domain.com
REACT_APP_VOICE_ENABLED=true
REACT_APP_ANALYTICS_ENABLED=true
```

## Deployment Options

### 1. Development Deployment

```bash
# Backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm start
```

### 2. Production Deployment

#### Using Docker

```bash
# Build images
docker build -t naarad-backend ./backend
docker build -t naarad-frontend ./frontend

# Run with docker-compose
docker-compose up -d
```

#### Using Systemd (Linux)

```bash
# Create systemd service files
sudo cp deployment/naarad-backend.service /etc/systemd/system/
sudo cp deployment/naarad-frontend.service /etc/systemd/system/

# Enable and start services
sudo systemctl enable naarad-backend
sudo systemctl enable naarad-frontend
sudo systemctl start naarad-backend
sudo systemctl start naarad-frontend
```

#### Using PM2 (Node.js)

```bash
# Install PM2
npm install -g pm2

# Start backend
cd backend
pm2 start ecosystem.config.js

# Start frontend
cd frontend
pm2 start ecosystem.config.js
```

## WebSocket Production Deployment

### 1. Load Balancer Configuration (Nginx)

```nginx
upstream websocket_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # WebSocket endpoint
    location /api/v1/ws {
        proxy_pass http://websocket_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
    
    # Regular API endpoints
    location /api/ {
        proxy_pass http://websocket_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Frontend
    location / {
        root /var/www/naarad-frontend;
        try_files $uri $uri/ /index.html;
    }
}
```

### 2. WebSocket Deployment Script

```bash
# Run the deployment script
python scripts/deploy_websocket.py --config deployment_config.json
```

### 3. Health Checks

```bash
# Check WebSocket health
curl http://localhost:8000/api/v1/ws/health

# Check overall system health
curl http://localhost:8000/health
```

## Voice Features Testing

### 1. Voice Health Check

```bash
# Test voice service health
curl http://localhost:8000/api/v1/voice/health

# Get available voices
curl http://localhost:8000/api/v1/voice/voices
```

### 2. Voice Testing Script

```bash
# Run comprehensive voice tests
python tests/test_voice_features.py

# Test specific voice features
python -m pytest tests/test_voice_features.py::TestVoiceFeatures::test_voice_process_endpoint
```

### 3. Manual Voice Testing

```bash
# Test speech synthesis
curl -X POST http://localhost:8000/api/v1/voice/synthesize \
  -F "text=Hello, this is a test message" \
  -F "voice=alloy" \
  -F "format=mp3"

# Test voice processing
curl -X POST http://localhost:8000/api/v1/voice/process \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": "base64_encoded_audio_data",
    "user_id": "test_user",
    "voice_preference": "alloy"
  }'
```

## Personalization Validation

### 1. Run Validation Script

```bash
# Run comprehensive validation
python scripts/validate_personalization.py

# Run specific tests
python scripts/validate_personalization.py --test-only learning
python scripts/validate_personalization.py --test-only accuracy
```

### 2. Monitor Learning Progress

```bash
# Check user preferences
curl http://localhost:8000/api/v1/personalization/preferences/user_123

# Get learning insights
curl http://localhost:8000/api/v1/personalization/insights/user_123
```

### 3. Validate Algorithm Performance

```bash
# Test algorithm validation
curl -X POST http://localhost:8000/api/v1/personalization/validate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "test_data": [...],
    "validation_type": "accuracy"
  }'
```

## Analytics Integration

### 1. Run Integration Script

```bash
# Run comprehensive analytics integration
python scripts/integrate_analytics.py

# Collect monitoring data only
python scripts/integrate_analytics.py --time-range 24h
```

### 2. Monitor Analytics

```bash
# Check analytics health
curl http://localhost:8000/api/v1/analytics/health

# Get available metrics
curl http://localhost:8000/api/v1/analytics/metrics
```

### 3. Generate Reports

```bash
# Analyze monitoring data
curl -X POST http://localhost:8000/api/v1/analytics/monitoring \
  -H "Content-Type: application/json" \
  -d '{
    "time_range": "24h",
    "metrics": ["response_time", "success_rate", "user_engagement"],
    "user_id": "system_admin"
  }'
```

## Monitoring and Logging

### 1. Application Monitoring

```bash
# Check application logs
tail -f logs/naarad.log

# Monitor system resources
htop
iotop
```

### 2. WebSocket Monitoring

```bash
# Monitor WebSocket connections
netstat -an | grep :8000 | grep ESTABLISHED | wc -l

# Check WebSocket metrics
curl http://localhost:8000/api/v1/ws/health
```

### 3. Voice Service Monitoring

```bash
# Monitor voice processing
tail -f logs/voice.log

# Check voice service metrics
curl http://localhost:8000/api/v1/voice/health
```

## Security Considerations

### 1. SSL/TLS Configuration

```bash
# Generate SSL certificates (Let's Encrypt)
sudo certbot --nginx -d your-domain.com

# Configure SSL in Nginx
sudo nano /etc/nginx/sites-available/naarad
```

### 2. Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

### 3. Rate Limiting

```bash
# Configure rate limiting in Nginx
# Add to nginx.conf:
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req zone=api burst=20 nodelay;
```

## Performance Optimization

### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_messages_user_id ON messages(user_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);
CREATE INDEX idx_user_preferences_user_id ON user_preferences(user_id);
```

### 2. Caching Configuration

```bash
# Configure Redis caching
redis-cli config set maxmemory 256mb
redis-cli config set maxmemory-policy allkeys-lru
```

### 3. Load Balancing

```bash
# Configure multiple backend instances
# Edit nginx.conf to add more upstream servers
upstream websocket_backend {
    server 127.0.0.1:8000 weight=1;
    server 127.0.0.1:8001 weight=1;
    server 127.0.0.1:8002 weight=1;
}
```

## Troubleshooting

### Common Issues

1. **WebSocket Connection Issues**
   ```bash
   # Check WebSocket service
   curl http://localhost:8000/api/v1/ws/health
   
   # Check firewall
   sudo ufw status
   ```

2. **Voice Processing Issues**
   ```bash
   # Check voice service
   curl http://localhost:8000/api/v1/voice/health
   
   # Check audio dependencies
   python -c "import pyaudio; print('PyAudio OK')"
   ```

3. **Database Connection Issues**
   ```bash
   # Test database connection
   python -c "from config.database_config import engine; print('DB OK')"
   ```

### Log Analysis

```bash
# Search for errors
grep -i error logs/naarad.log

# Search for specific user issues
grep "user_123" logs/naarad.log

# Monitor real-time logs
tail -f logs/naarad.log | grep -E "(ERROR|WARNING)"
```

## Backup and Recovery

### 1. Database Backup

```bash
# Create database backup
pg_dump naarad > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore database
psql naarad < backup_20240115_143022.sql
```

### 2. Configuration Backup

```bash
# Backup configuration files
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env config/ logs/
```

### 3. Application Backup

```bash
# Backup application files
tar -czf app_backup_$(date +%Y%m%d).tar.gz backend/ frontend/
```

## Scaling Considerations

### 1. Horizontal Scaling

```bash
# Add more backend instances
# Update nginx configuration
# Use load balancer for WebSocket connections
```

### 2. Vertical Scaling

```bash
# Increase server resources
# Optimize database queries
# Implement caching strategies
```

### 3. Auto-scaling

```bash
# Configure auto-scaling based on metrics
# Monitor CPU, memory, and connection usage
# Set up alerts for scaling events
```

## Maintenance

### 1. Regular Updates

```bash
# Update dependencies
pip install -r requirements.txt --upgrade
npm update

# Update application
git pull origin main
```

### 2. Health Checks

```bash
# Run health checks
python scripts/health_check.py

# Monitor system resources
python scripts/monitor_system.py
```

### 3. Performance Monitoring

```bash
# Monitor performance metrics
python scripts/performance_monitor.py

# Generate performance reports
python scripts/generate_reports.py
```

## Support and Documentation

- **API Documentation**: http://localhost:8000/docs
- **System Status**: http://localhost:8000/health
- **Logs**: `logs/` directory
- **Configuration**: `config/` directory

For additional support, refer to the project documentation or contact the development team. 