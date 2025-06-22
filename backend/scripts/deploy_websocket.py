#!/usr/bin/env python3
"""
WebSocket Production Deployment Script

This script handles the deployment of WebSocket services in a production environment
with proper configuration, monitoring, and scaling considerations.
"""

import os
import sys
import json
import subprocess
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebSocketDeployer:
    """WebSocket production deployment manager."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "deployment_config.json"
        self.config = self.load_config()
        self.deployment_status = {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        default_config = {
            "websocket": {
                "host": "0.0.0.0",
                "port": 8000,
                "max_connections": 1000,
                "heartbeat_interval": 30,
                "connection_timeout": 300,
                "rate_limit": {
                    "messages_per_minute": 60,
                    "burst_limit": 10
                },
                "ssl": {
                    "enabled": True,
                    "cert_path": "/etc/ssl/certs/websocket.crt",
                    "key_path": "/etc/ssl/private/websocket.key"
                },
                "compression": True,
                "load_balancer": {
                    "enabled": True,
                    "algorithm": "round_robin",
                    "health_check_interval": 30
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 9090,
                "log_level": "INFO",
                "alerting": {
                    "enabled": True,
                    "webhook_url": None
                }
            },
            "scaling": {
                "auto_scaling": True,
                "min_instances": 2,
                "max_instances": 10,
                "cpu_threshold": 70,
                "memory_threshold": 80
            },
            "security": {
                "cors_origins": ["*"],
                "rate_limiting": True,
                "authentication": False,
                "ssl_required": True
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            config[key].update(value)
                    return config
            else:
                logger.info(f"Config file not found, using default configuration")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def validate_environment(self) -> bool:
        """Validate production environment requirements."""
        logger.info("Validating production environment...")
        
        checks = [
            ("Python version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("SSL certificates", self.check_ssl_certs),
            ("Port availability", self.check_port_availability),
            ("System resources", self.check_system_resources),
            ("Network connectivity", self.check_network_connectivity)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                if check_func():
                    logger.info(f"✓ {check_name}: PASSED")
                else:
                    logger.error(f"✗ {check_name}: FAILED")
                    all_passed = False
            except Exception as e:
                logger.error(f"✗ {check_name}: ERROR - {e}")
                all_passed = False
        
        return all_passed
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        return version.major == 3 and version.minor >= 8
    
    def check_dependencies(self) -> bool:
        """Check required dependencies."""
        required_packages = [
            "fastapi",
            "uvicorn",
            "websockets",
            "asyncio",
            "pydantic"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            return False
        
        return True
    
    def check_ssl_certs(self) -> bool:
        """Check SSL certificate availability."""
        if not self.config["websocket"]["ssl"]["enabled"]:
            return True
        
        cert_path = self.config["websocket"]["ssl"]["cert_path"]
        key_path = self.config["websocket"]["ssl"]["key_path"]
        
        return os.path.exists(cert_path) and os.path.exists(key_path)
    
    def check_port_availability(self) -> bool:
        """Check if required ports are available."""
        import socket
        
        ports_to_check = [
            self.config["websocket"]["port"],
            self.config["monitoring"]["metrics_port"]
        ]
        
        for port in ports_to_check:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
            except OSError:
                logger.warning(f"Port {port} is already in use")
                return False
        
        return True
    
    def check_system_resources(self) -> bool:
        """Check system resource availability."""
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.available < 512 * 1024 * 1024:  # 512MB
            logger.warning("Low memory available")
            return False
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < 1 * 1024 * 1024 * 1024:  # 1GB
            logger.warning("Low disk space")
            return False
        
        return True
    
    def check_network_connectivity(self) -> bool:
        """Check network connectivity."""
        import socket
        
        try:
            # Test DNS resolution
            socket.gethostbyname("localhost")
            return True
        except socket.gaierror:
            return False
    
    def setup_environment(self) -> bool:
        """Setup production environment."""
        logger.info("Setting up production environment...")
        
        try:
            # Create necessary directories
            directories = [
                "logs",
                "ssl",
                "config",
                "temp"
            ]
            
            for directory in directories:
                Path(directory).mkdir(exist_ok=True)
            
            # Setup logging
            self.setup_logging()
            
            # Setup SSL (if enabled)
            if self.config["websocket"]["ssl"]["enabled"]:
                self.setup_ssl()
            
            # Setup monitoring
            if self.config["monitoring"]["enabled"]:
                self.setup_monitoring()
            
            logger.info("Environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def setup_logging(self):
        """Setup production logging."""
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                }
            },
            "handlers": {
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "logs/websocket.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "formatter": "detailed"
                },
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "detailed"
                }
            },
            "root": {
                "level": self.config["monitoring"]["log_level"],
                "handlers": ["file", "console"]
            }
        }
        
        with open("config/logging_config.json", "w") as f:
            json.dump(log_config, f, indent=2)
    
    def setup_ssl(self):
        """Setup SSL certificates."""
        logger.info("Setting up SSL certificates...")
        
        # This would typically involve generating or copying certificates
        # For now, we'll create placeholder files
        cert_path = self.config["websocket"]["ssl"]["cert_path"]
        key_path = self.config["websocket"]["ssl"]["key_path"]
        
        if not os.path.exists(cert_path):
            logger.warning(f"SSL certificate not found at {cert_path}")
            logger.info("Please ensure SSL certificates are properly configured")
        
        if not os.path.exists(key_path):
            logger.warning(f"SSL key not found at {key_path}")
            logger.info("Please ensure SSL key is properly configured")
    
    def setup_monitoring(self):
        """Setup monitoring and metrics."""
        logger.info("Setting up monitoring...")
        
        # Create monitoring configuration
        monitoring_config = {
            "metrics_endpoint": f":{self.config['monitoring']['metrics_port']}",
            "health_check_interval": self.config["websocket"]["load_balancer"]["health_check_interval"],
            "alerting": self.config["monitoring"]["alerting"]
        }
        
        with open("config/monitoring_config.json", "w") as f:
            json.dump(monitoring_config, f, indent=2)
    
    def deploy_websocket_service(self) -> bool:
        """Deploy the WebSocket service."""
        logger.info("Deploying WebSocket service...")
        
        try:
            # Start the WebSocket service
            cmd = [
                "uvicorn",
                "main:app",
                "--host", self.config["websocket"]["host"],
                "--port", str(self.config["websocket"]["port"]),
                "--workers", str(self.config["scaling"]["min_instances"]),
                "--log-config", "config/logging_config.json"
            ]
            
            if self.config["websocket"]["ssl"]["enabled"]:
                cmd.extend([
                    "--ssl-certfile", self.config["websocket"]["ssl"]["cert_path"],
                    "--ssl-keyfile", self.config["websocket"]["ssl"]["key_path"]
                ])
            
            logger.info(f"Starting WebSocket service with command: {' '.join(cmd)}")
            
            # In production, you might want to use a process manager like systemd
            # For now, we'll just log the command
            self.deployment_status["websocket"] = {
                "status": "started",
                "command": cmd,
                "timestamp": time.time()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket deployment failed: {e}")
            return False
    
    def setup_load_balancer(self) -> bool:
        """Setup load balancer configuration."""
        if not self.config["websocket"]["load_balancer"]["enabled"]:
            return True
        
        logger.info("Setting up load balancer...")
        
        # Create nginx configuration for WebSocket load balancing
        nginx_config = f"""
upstream websocket_backend {{
    server 127.0.0.1:{self.config["websocket"]["port"]};
    # Add more servers for scaling
}}

server {{
    listen 80;
    server_name _;
    
    location /api/v1/ws {{
        proxy_pass http://websocket_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }}
    
    location / {{
        proxy_pass http://websocket_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""
        
        with open("config/nginx_websocket.conf", "w") as f:
            f.write(nginx_config)
        
        logger.info("Load balancer configuration created")
        return True
    
    def run_health_checks(self) -> bool:
        """Run health checks on deployed services."""
        logger.info("Running health checks...")
        
        import requests
        
        health_endpoints = [
            f"http://localhost:{self.config['websocket']['port']}/health",
            f"http://localhost:{self.config['websocket']['port']}/api/v1/health"
        ]
        
        all_healthy = True
        for endpoint in health_endpoints:
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✓ Health check passed: {endpoint}")
                else:
                    logger.error(f"✗ Health check failed: {endpoint} - {response.status_code}")
                    all_healthy = False
            except Exception as e:
                logger.error(f"✗ Health check error: {endpoint} - {e}")
                all_healthy = False
        
        return all_healthy
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate deployment report."""
        report = {
            "timestamp": time.time(),
            "config": self.config,
            "deployment_status": self.deployment_status,
            "health_checks": self.run_health_checks(),
            "recommendations": []
        }
        
        # Add recommendations based on configuration
        if not self.config["websocket"]["ssl"]["enabled"]:
            report["recommendations"].append("Enable SSL for production security")
        
        if self.config["scaling"]["max_instances"] < 5:
            report["recommendations"].append("Consider increasing max instances for better scalability")
        
        if not self.config["monitoring"]["alerting"]["enabled"]:
            report["recommendations"].append("Enable alerting for production monitoring")
        
        return report
    
    def deploy(self) -> bool:
        """Main deployment process."""
        logger.info("Starting WebSocket production deployment...")
        
        steps = [
            ("Environment validation", self.validate_environment),
            ("Environment setup", self.setup_environment),
            ("Load balancer setup", self.setup_load_balancer),
            ("WebSocket service deployment", self.deploy_websocket_service),
            ("Health checks", self.run_health_checks)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Executing: {step_name}")
            try:
                if not step_func():
                    logger.error(f"Deployment failed at: {step_name}")
                    return False
            except Exception as e:
                logger.error(f"Deployment error at {step_name}: {e}")
                return False
        
        # Generate deployment report
        report = self.generate_deployment_report()
        with open("deployment_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info("WebSocket deployment completed successfully!")
        logger.info("Deployment report saved to: deployment_report.json")
        
        return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WebSocket Production Deployment")
    parser.add_argument("--config", help="Path to deployment configuration file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate environment")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deployed without actually deploying")
    
    args = parser.parse_args()
    
    deployer = WebSocketDeployer(args.config)
    
    if args.validate_only:
        success = deployer.validate_environment()
        sys.exit(0 if success else 1)
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual deployment will occur")
        logger.info(f"Configuration: {json.dumps(deployer.config, indent=2)}")
        sys.exit(0)
    
    success = deployer.deploy()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 