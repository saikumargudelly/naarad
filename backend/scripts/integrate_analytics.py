#!/usr/bin/env python3
"""
Analytics Integration Script

This script integrates analytics with existing monitoring systems and
provides comprehensive data analysis and insights.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalyticsIntegrator:
    """Analytics integration with monitoring systems."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.integration_results = {}
        self.analytics_data = {}
    
    async def collect_monitoring_data(self, time_range: str = "24h") -> Dict[str, Any]:
        """Collect data from existing monitoring systems."""
        logger.info(f"Collecting monitoring data for time range: {time_range}")
        
        monitoring_data = {
            "system_metrics": await self.get_system_metrics(),
            "application_metrics": await self.get_application_metrics(),
            "user_metrics": await self.get_user_metrics(),
            "performance_metrics": await self.get_performance_metrics(),
            "error_metrics": await self.get_error_metrics()
        }
        
        return monitoring_data
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            # Simulate system metrics collection
            # In production, this would connect to actual monitoring systems
            system_metrics = {
                "cpu_usage": [45, 52, 48, 61, 55, 49, 58, 62, 67, 71],
                "memory_usage": [68, 72, 75, 78, 81, 79, 83, 85, 87, 89],
                "disk_usage": [45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
                "network_io": [120, 135, 142, 158, 165, 178, 192, 205, 218, 230],
                "timestamp": [
                    (datetime.now() - timedelta(hours=i)).isoformat()
                    for i in range(10, 0, -1)
                ]
            }
            
            return system_metrics
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    async def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-level metrics."""
        try:
            # Simulate application metrics collection
            app_metrics = {
                "request_count": [1250, 1380, 1420, 1560, 1680, 1750, 1820, 1950, 2080, 2200],
                "response_time": [1.2, 1.1, 1.3, 1.4, 1.2, 1.5, 1.3, 1.6, 1.4, 1.7],
                "error_rate": [0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.01, 0.05, 0.02, 0.03],
                "active_users": [45, 52, 58, 63, 67, 72, 78, 83, 89, 95],
                "timestamp": [
                    (datetime.now() - timedelta(hours=i)).isoformat()
                    for i in range(10, 0, -1)
                ]
            }
            
            return app_metrics
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return {}
    
    async def get_user_metrics(self) -> Dict[str, Any]:
        """Get user behavior metrics."""
        try:
            # Simulate user metrics collection
            user_metrics = {
                "session_duration": [1800, 2100, 1950, 2400, 2250, 2700, 2550, 3000, 2850, 3300],
                "messages_per_session": [8, 12, 10, 15, 13, 18, 16, 20, 18, 22],
                "feature_usage": {
                    "voice": [0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.45, 0.48, 0.52, 0.55],
                    "analytics": [0.15, 0.18, 0.22, 0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.45],
                    "personalization": [0.35, 0.38, 0.42, 0.45, 0.48, 0.52, 0.55, 0.58, 0.62, 0.65]
                },
                "user_satisfaction": [4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.8, 4.9],
                "timestamp": [
                    (datetime.now() - timedelta(hours=i)).isoformat()
                    for i in range(10, 0, -1)
                ]
            }
            
            return user_metrics
        except Exception as e:
            logger.error(f"Error collecting user metrics: {e}")
            return {}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            # Simulate performance metrics collection
            perf_metrics = {
                "websocket_connections": [25, 28, 32, 35, 38, 42, 45, 48, 52, 55],
                "message_throughput": [150, 165, 180, 195, 210, 225, 240, 255, 270, 285],
                "voice_processing_time": [2.1, 2.0, 2.2, 2.1, 2.3, 2.2, 2.4, 2.3, 2.5, 2.4],
                "analytics_processing_time": [1.8, 1.7, 1.9, 1.8, 2.0, 1.9, 2.1, 2.0, 2.2, 2.1],
                "timestamp": [
                    (datetime.now() - timedelta(hours=i)).isoformat()
                    for i in range(10, 0, -1)
                ]
            }
            
            return perf_metrics
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return {}
    
    async def get_error_metrics(self) -> Dict[str, Any]:
        """Get error and exception metrics."""
        try:
            # Simulate error metrics collection
            error_metrics = {
                "total_errors": [12, 15, 8, 18, 11, 14, 9, 16, 12, 13],
                "error_types": {
                    "network": [3, 4, 2, 5, 3, 4, 2, 5, 3, 4],
                    "authentication": [2, 3, 1, 4, 2, 3, 1, 4, 2, 3],
                    "processing": [4, 5, 3, 6, 4, 5, 3, 6, 4, 5],
                    "timeout": [3, 3, 2, 3, 2, 2, 3, 1, 3, 1]
                },
                "error_rate": [0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.01, 0.05, 0.02, 0.03],
                "timestamp": [
                    (datetime.now() - timedelta(hours=i)).isoformat()
                    for i in range(10, 0, -1)
                ]
            }
            
            return error_metrics
        except Exception as e:
            logger.error(f"Error collecting error metrics: {e}")
            return {}
    
    async def analyze_monitoring_data(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze collected monitoring data."""
        logger.info("Analyzing monitoring data...")
        
        analysis_results = {
            "system_analysis": await self.analyze_system_metrics(monitoring_data.get("system_metrics", {})),
            "application_analysis": await self.analyze_application_metrics(monitoring_data.get("application_metrics", {})),
            "user_analysis": await self.analyze_user_metrics(monitoring_data.get("user_metrics", {})),
            "performance_analysis": await self.analyze_performance_metrics(monitoring_data.get("performance_metrics", {})),
            "error_analysis": await self.analyze_error_metrics(monitoring_data.get("error_metrics", {})),
            "trends": await self.analyze_trends(monitoring_data),
            "correlations": await self.analyze_correlations(monitoring_data)
        }
        
        return analysis_results
    
    async def analyze_system_metrics(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system metrics."""
        if not system_metrics:
            return {}
        
        df = pd.DataFrame(system_metrics)
        
        analysis = {
            "cpu_usage": {
                "mean": df["cpu_usage"].mean(),
                "max": df["cpu_usage"].max(),
                "min": df["cpu_usage"].min(),
                "trend": "increasing" if df["cpu_usage"].iloc[-1] > df["cpu_usage"].iloc[0] else "decreasing"
            },
            "memory_usage": {
                "mean": df["memory_usage"].mean(),
                "max": df["memory_usage"].max(),
                "min": df["memory_usage"].min(),
                "trend": "increasing" if df["memory_usage"].iloc[-1] > df["memory_usage"].iloc[0] else "decreasing"
            },
            "alerts": []
        }
        
        # Generate alerts
        if df["cpu_usage"].max() > 70:
            analysis["alerts"].append("High CPU usage detected")
        
        if df["memory_usage"].max() > 85:
            analysis["alerts"].append("High memory usage detected")
        
        return analysis
    
    async def analyze_application_metrics(self, app_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze application metrics."""
        if not app_metrics:
            return {}
        
        df = pd.DataFrame(app_metrics)
        
        analysis = {
            "request_volume": {
                "total": df["request_count"].sum(),
                "average_per_hour": df["request_count"].mean(),
                "trend": "increasing" if df["request_count"].iloc[-1] > df["request_count"].iloc[0] else "decreasing"
            },
            "response_time": {
                "average": df["response_time"].mean(),
                "max": df["response_time"].max(),
                "min": df["response_time"].min(),
                "trend": "increasing" if df["response_time"].iloc[-1] > df["response_time"].iloc[0] else "decreasing"
            },
            "error_rate": {
                "average": df["error_rate"].mean(),
                "max": df["error_rate"].max(),
                "trend": "increasing" if df["error_rate"].iloc[-1] > df["error_rate"].iloc[0] else "decreasing"
            },
            "alerts": []
        }
        
        # Generate alerts
        if df["response_time"].max() > 2.0:
            analysis["alerts"].append("High response time detected")
        
        if df["error_rate"].max() > 0.05:
            analysis["alerts"].append("High error rate detected")
        
        return analysis
    
    async def analyze_user_metrics(self, user_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user behavior metrics."""
        if not user_metrics:
            return {}
        
        df = pd.DataFrame(user_metrics)
        
        analysis = {
            "engagement": {
                "avg_session_duration": df["session_duration"].mean(),
                "avg_messages_per_session": df["messages_per_session"].mean(),
                "trend": "increasing" if df["session_duration"].iloc[-1] > df["session_duration"].iloc[0] else "decreasing"
            },
            "feature_adoption": {
                "voice_usage": df["feature_usage"]["voice"].mean(),
                "analytics_usage": df["feature_usage"]["analytics"].mean(),
                "personalization_usage": df["feature_usage"]["personalization"].mean()
            },
            "satisfaction": {
                "average_rating": df["user_satisfaction"].mean(),
                "trend": "increasing" if df["user_satisfaction"].iloc[-1] > df["user_satisfaction"].iloc[0] else "decreasing"
            },
            "insights": []
        }
        
        # Generate insights
        if df["feature_usage"]["voice"].iloc[-1] > 0.5:
            analysis["insights"].append("Voice feature adoption is strong")
        
        if df["user_satisfaction"].mean() > 4.5:
            analysis["insights"].append("High user satisfaction levels")
        
        return analysis
    
    async def analyze_performance_metrics(self, perf_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics."""
        if not perf_metrics:
            return {}
        
        df = pd.DataFrame(perf_metrics)
        
        analysis = {
            "websocket_performance": {
                "avg_connections": df["websocket_connections"].mean(),
                "max_connections": df["websocket_connections"].max(),
                "trend": "increasing" if df["websocket_connections"].iloc[-1] > df["websocket_connections"].iloc[0] else "decreasing"
            },
            "throughput": {
                "avg_messages_per_hour": df["message_throughput"].mean(),
                "max_messages_per_hour": df["message_throughput"].max(),
                "trend": "increasing" if df["message_throughput"].iloc[-1] > df["message_throughput"].iloc[0] else "decreasing"
            },
            "processing_times": {
                "avg_voice_processing": df["voice_processing_time"].mean(),
                "avg_analytics_processing": df["analytics_processing_time"].mean()
            },
            "alerts": []
        }
        
        # Generate alerts
        if df["voice_processing_time"].max() > 3.0:
            analysis["alerts"].append("Slow voice processing detected")
        
        if df["analytics_processing_time"].max() > 2.5:
            analysis["alerts"].append("Slow analytics processing detected")
        
        return analysis
    
    async def analyze_error_metrics(self, error_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error metrics."""
        if not error_metrics:
            return {}
        
        df = pd.DataFrame(error_metrics)
        
        analysis = {
            "error_summary": {
                "total_errors": df["total_errors"].sum(),
                "avg_errors_per_hour": df["total_errors"].mean(),
                "error_rate_trend": "increasing" if df["error_rate"].iloc[-1] > df["error_rate"].iloc[0] else "decreasing"
            },
            "error_types": {
                "network_errors": df["error_types"]["network"].sum(),
                "auth_errors": df["error_types"]["authentication"].sum(),
                "processing_errors": df["error_types"]["processing"].sum(),
                "timeout_errors": df["error_types"]["timeout"].sum()
            },
            "critical_issues": []
        }
        
        # Identify critical issues
        if df["error_rate"].max() > 0.05:
            analysis["critical_issues"].append("High error rate detected")
        
        if df["error_types"]["authentication"].sum() > 10:
            analysis["critical_issues"].append("Authentication issues detected")
        
        return analysis
    
    async def analyze_trends(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends across all metrics."""
        trends = {
            "system_trends": {},
            "application_trends": {},
            "user_trends": {},
            "performance_trends": {}
        }
        
        # Analyze system trends
        if "system_metrics" in monitoring_data:
            sys_data = monitoring_data["system_metrics"]
            if sys_data:
                df = pd.DataFrame(sys_data)
                trends["system_trends"] = {
                    "cpu_trend": "increasing" if df["cpu_usage"].iloc[-1] > df["cpu_usage"].iloc[0] else "decreasing",
                    "memory_trend": "increasing" if df["memory_usage"].iloc[-1] > df["memory_usage"].iloc[0] else "decreasing"
                }
        
        # Analyze application trends
        if "application_metrics" in monitoring_data:
            app_data = monitoring_data["application_metrics"]
            if app_data:
                df = pd.DataFrame(app_data)
                trends["application_trends"] = {
                    "request_trend": "increasing" if df["request_count"].iloc[-1] > df["request_count"].iloc[0] else "decreasing",
                    "response_time_trend": "increasing" if df["response_time"].iloc[-1] > df["response_time"].iloc[0] else "decreasing"
                }
        
        return trends
    
    async def analyze_correlations(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        correlations = {}
        
        # Create combined dataframe for correlation analysis
        combined_data = {}
        
        if "system_metrics" in monitoring_data and "application_metrics" in monitoring_data:
            sys_data = monitoring_data["system_metrics"]
            app_data = monitoring_data["application_metrics"]
            
            if sys_data and app_data:
                # Align data by timestamp
                combined_data = {
                    "cpu_usage": sys_data["cpu_usage"],
                    "memory_usage": sys_data["memory_usage"],
                    "request_count": app_data["request_count"],
                    "response_time": app_data["response_time"],
                    "error_rate": app_data["error_rate"]
                }
                
                df = pd.DataFrame(combined_data)
                corr_matrix = df.corr()
                
                correlations = {
                    "cpu_response_time": corr_matrix.loc["cpu_usage", "response_time"],
                    "memory_error_rate": corr_matrix.loc["memory_usage", "error_rate"],
                    "requests_cpu": corr_matrix.loc["request_count", "cpu_usage"]
                }
        
        return correlations
    
    async def generate_visualizations(self, monitoring_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> List[str]:
        """Generate visualizations for monitoring data."""
        logger.info("Generating visualizations...")
        
        visualization_files = []
        
        try:
            # Set up matplotlib style
            plt.style.use('dark_background')
            sns.set_palette("husl")
            
            # Create output directory
            output_dir = Path("analytics_output")
            output_dir.mkdir(exist_ok=True)
            
            # 1. System Performance Dashboard
            if "system_metrics" in monitoring_data and monitoring_data["system_metrics"]:
                await self.create_system_dashboard(monitoring_data["system_metrics"], output_dir)
                visualization_files.append(str(output_dir / "system_dashboard.png"))
            
            # 2. Application Performance Dashboard
            if "application_metrics" in monitoring_data and monitoring_data["application_metrics"]:
                await self.create_application_dashboard(monitoring_data["application_metrics"], output_dir)
                visualization_files.append(str(output_dir / "application_dashboard.png"))
            
            # 3. User Behavior Dashboard
            if "user_metrics" in monitoring_data and monitoring_data["user_metrics"]:
                await self.create_user_dashboard(monitoring_data["user_metrics"], output_dir)
                visualization_files.append(str(output_dir / "user_dashboard.png"))
            
            # 4. Error Analysis Dashboard
            if "error_metrics" in monitoring_data and monitoring_data["error_metrics"]:
                await self.create_error_dashboard(monitoring_data["error_metrics"], output_dir)
                visualization_files.append(str(output_dir / "error_dashboard.png"))
            
            # 5. Performance Trends Dashboard
            if "performance_metrics" in monitoring_data and monitoring_data["performance_metrics"]:
                await self.create_performance_dashboard(monitoring_data["performance_metrics"], output_dir)
                visualization_files.append(str(output_dir / "performance_dashboard.png"))
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualization_files
    
    async def create_system_dashboard(self, system_metrics: Dict[str, Any], output_dir: Path):
        """Create system performance dashboard."""
        df = pd.DataFrame(system_metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Performance Dashboard', fontsize=16, fontweight='bold')
        
        # CPU Usage
        axes[0, 0].plot(df.index, df['cpu_usage'], marker='o', linewidth=2)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory Usage
        axes[0, 1].plot(df.index, df['memory_usage'], marker='s', linewidth=2, color='orange')
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_ylabel('Memory %')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Disk Usage
        axes[1, 0].plot(df.index, df['disk_usage'], marker='^', linewidth=2, color='green')
        axes[1, 0].set_title('Disk Usage (%)')
        axes[1, 0].set_ylabel('Disk %')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Network I/O
        axes[1, 1].plot(df.index, df['network_io'], marker='d', linewidth=2, color='red')
        axes[1, 1].set_title('Network I/O (MB/s)')
        axes[1, 1].set_ylabel('MB/s')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "system_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    async def create_application_dashboard(self, app_metrics: Dict[str, Any], output_dir: Path):
        """Create application performance dashboard."""
        df = pd.DataFrame(app_metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Application Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Request Count
        axes[0, 0].plot(df.index, df['request_count'], marker='o', linewidth=2)
        axes[0, 0].set_title('Request Count')
        axes[0, 0].set_ylabel('Requests')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Response Time
        axes[0, 1].plot(df.index, df['response_time'], marker='s', linewidth=2, color='orange')
        axes[0, 1].set_title('Response Time')
        axes[0, 1].set_ylabel('Seconds')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error Rate
        axes[1, 0].plot(df.index, df['error_rate'], marker='^', linewidth=2, color='red')
        axes[1, 0].set_title('Error Rate')
        axes[1, 0].set_ylabel('Error Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Active Users
        axes[1, 1].plot(df.index, df['active_users'], marker='d', linewidth=2, color='green')
        axes[1, 1].set_title('Active Users')
        axes[1, 1].set_ylabel('Users')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "application_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    async def create_user_dashboard(self, user_metrics: Dict[str, Any], output_dir: Path):
        """Create user behavior dashboard."""
        df = pd.DataFrame(user_metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('User Behavior Dashboard', fontsize=16, fontweight='bold')
        
        # Session Duration
        axes[0, 0].plot(df.index, df['session_duration'], marker='o', linewidth=2)
        axes[0, 0].set_title('Session Duration')
        axes[0, 0].set_ylabel('Seconds')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Messages per Session
        axes[0, 1].plot(df.index, df['messages_per_session'], marker='s', linewidth=2, color='orange')
        axes[0, 1].set_title('Messages per Session')
        axes[0, 1].set_ylabel('Messages')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature Usage
        feature_data = df['feature_usage'].apply(pd.Series)
        feature_data.plot(kind='bar', ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title('Feature Usage')
        axes[1, 0].set_ylabel('Usage Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # User Satisfaction
        axes[1, 1].plot(df.index, df['user_satisfaction'], marker='d', linewidth=2, color='green')
        axes[1, 1].set_title('User Satisfaction')
        axes[1, 1].set_ylabel('Rating')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "user_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    async def create_error_dashboard(self, error_metrics: Dict[str, Any], output_dir: Path):
        """Create error analysis dashboard."""
        df = pd.DataFrame(error_metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Error Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Total Errors
        axes[0, 0].plot(df.index, df['total_errors'], marker='o', linewidth=2, color='red')
        axes[0, 0].set_title('Total Errors')
        axes[0, 0].set_ylabel('Error Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error Rate
        axes[0, 1].plot(df.index, df['error_rate'], marker='s', linewidth=2, color='orange')
        axes[0, 1].set_title('Error Rate')
        axes[0, 1].set_ylabel('Error Rate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error Types
        error_types = df['error_types'].apply(pd.Series)
        error_types.plot(kind='bar', ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title('Error Types')
        axes[1, 0].set_ylabel('Error Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Error Distribution
        error_types.sum().plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
        axes[1, 1].set_title('Error Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / "error_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    async def create_performance_dashboard(self, perf_metrics: Dict[str, Any], output_dir: Path):
        """Create performance trends dashboard."""
        df = pd.DataFrame(perf_metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Trends Dashboard', fontsize=16, fontweight='bold')
        
        # WebSocket Connections
        axes[0, 0].plot(df.index, df['websocket_connections'], marker='o', linewidth=2)
        axes[0, 0].set_title('WebSocket Connections')
        axes[0, 0].set_ylabel('Connections')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Message Throughput
        axes[0, 1].plot(df.index, df['message_throughput'], marker='s', linewidth=2, color='orange')
        axes[0, 1].set_title('Message Throughput')
        axes[0, 1].set_ylabel('Messages/Hour')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Processing Times
        axes[1, 0].plot(df.index, df['voice_processing_time'], marker='^', linewidth=2, label='Voice')
        axes[1, 0].plot(df.index, df['analytics_processing_time'], marker='d', linewidth=2, label='Analytics')
        axes[1, 0].set_title('Processing Times')
        axes[1, 0].set_ylabel('Seconds')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance Summary
        summary_data = {
            'Avg Connections': df['websocket_connections'].mean(),
            'Avg Throughput': df['message_throughput'].mean(),
            'Avg Voice Time': df['voice_processing_time'].mean(),
            'Avg Analytics Time': df['analytics_processing_time'].mean()
        }
        
        axes[1, 1].bar(summary_data.keys(), summary_data.values(), color=['blue', 'orange', 'green', 'red'])
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    async def integrate_with_analytics_api(self, monitoring_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate data with analytics API."""
        logger.info("Integrating with analytics API...")
        
        integration_results = {
            "api_calls": 0,
            "successful_integrations": 0,
            "failed_integrations": 0,
            "errors": []
        }
        
        try:
            # Send monitoring data to analytics API
            response = requests.post(
                f"{self.api_base_url}/api/v1/analytics/monitoring",
                json={
                    "time_range": "24h",
                    "metrics": ["response_time", "success_rate", "user_engagement"],
                    "user_id": "system_integration",
                    "monitoring_data": monitoring_data
                },
                timeout=30
            )
            
            integration_results["api_calls"] += 1
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    integration_results["successful_integrations"] += 1
                    logger.info("Successfully integrated with analytics API")
                else:
                    integration_results["failed_integrations"] += 1
                    integration_results["errors"].append(f"API integration failed: {data.get('error')}")
            else:
                integration_results["failed_integrations"] += 1
                integration_results["errors"].append(f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            integration_results["failed_integrations"] += 1
            integration_results["errors"].append(f"Integration error: {str(e)}")
        
        return integration_results
    
    async def run_comprehensive_integration(self) -> Dict[str, Any]:
        """Run comprehensive analytics integration."""
        logger.info("Starting comprehensive analytics integration...")
        
        integration_results = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_data": {},
            "analysis_results": {},
            "visualizations": [],
            "api_integration": {},
            "summary": {},
            "recommendations": []
        }
        
        # Step 1: Collect monitoring data
        logger.info("Step 1: Collecting monitoring data...")
        monitoring_data = await self.collect_monitoring_data()
        integration_results["monitoring_data"] = monitoring_data
        
        # Step 2: Analyze data
        logger.info("Step 2: Analyzing monitoring data...")
        analysis_results = await self.analyze_monitoring_data(monitoring_data)
        integration_results["analysis_results"] = analysis_results
        
        # Step 3: Generate visualizations
        logger.info("Step 3: Generating visualizations...")
        visualization_files = await self.generate_visualizations(monitoring_data, analysis_results)
        integration_results["visualizations"] = visualization_files
        
        # Step 4: Integrate with analytics API
        logger.info("Step 4: Integrating with analytics API...")
        api_integration = await self.integrate_with_analytics_api(monitoring_data, analysis_results)
        integration_results["api_integration"] = api_integration
        
        # Step 5: Generate summary and recommendations
        logger.info("Step 5: Generating summary and recommendations...")
        integration_results["summary"] = self.generate_integration_summary(integration_results)
        integration_results["recommendations"] = self.generate_integration_recommendations(integration_results)
        
        return integration_results
    
    def generate_integration_summary(self, integration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integration summary."""
        summary = {
            "total_metrics_collected": 0,
            "analysis_completed": False,
            "visualizations_generated": len(integration_results.get("visualizations", [])),
            "api_integration_success": False,
            "overall_status": "unknown"
        }
        
        # Count metrics
        monitoring_data = integration_results.get("monitoring_data", {})
        for category in monitoring_data.values():
            if isinstance(category, dict):
                summary["total_metrics_collected"] += len(category)
        
        # Check analysis completion
        if integration_results.get("analysis_results"):
            summary["analysis_completed"] = True
        
        # Check API integration
        api_integration = integration_results.get("api_integration", {})
        if api_integration.get("successful_integrations", 0) > 0:
            summary["api_integration_success"] = True
        
        # Determine overall status
        if (summary["total_metrics_collected"] > 0 and 
            summary["analysis_completed"] and 
            summary["api_integration_success"]):
            summary["overall_status"] = "success"
        elif summary["total_metrics_collected"] > 0:
            summary["overall_status"] = "partial"
        else:
            summary["overall_status"] = "failed"
        
        return summary
    
    def generate_integration_recommendations(self, integration_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on integration results."""
        recommendations = []
        
        # Check for alerts in analysis results
        analysis_results = integration_results.get("analysis_results", {})
        
        for category, analysis in analysis_results.items():
            if isinstance(analysis, dict) and "alerts" in analysis:
                for alert in analysis["alerts"]:
                    recommendations.append(f"Alert: {alert}")
        
        # Check API integration issues
        api_integration = integration_results.get("api_integration", {})
        if api_integration.get("failed_integrations", 0) > 0:
            recommendations.append("API integration issues detected. Review API connectivity.")
        
        # Performance recommendations
        if "performance_analysis" in analysis_results:
            perf_analysis = analysis_results["performance_analysis"]
            if perf_analysis.get("alerts"):
                recommendations.append("Performance optimization recommended based on analysis.")
        
        # Monitoring recommendations
        if integration_results.get("summary", {}).get("total_metrics_collected", 0) < 10:
            recommendations.append("Consider expanding monitoring coverage for better insights.")
        
        if not recommendations:
            recommendations.append("Integration completed successfully. All systems operating normally.")
        
        return recommendations
    
    def save_integration_results(self, results: Dict[str, Any], filename: str = None):
        """Save integration results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_integration_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Integration results saved to: {filename}")
        return filename

async def main():
    """Main integration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analytics Integration with Monitoring")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--time-range", default="24h", help="Time range for data collection")
    
    args = parser.parse_args()
    
    integrator = AnalyticsIntegrator(args.api_url)
    
    # Run comprehensive integration
    results = await integrator.run_comprehensive_integration()
    
    # Save results
    filename = integrator.save_integration_results(results, args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYTICS INTEGRATION SUMMARY")
    print("="*50)
    summary = results["summary"]
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Metrics Collected: {summary['total_metrics_collected']}")
    print(f"Analysis Completed: {summary['analysis_completed']}")
    print(f"Visualizations Generated: {summary['visualizations_generated']}")
    print(f"API Integration Success: {summary['api_integration_success']}")
    print(f"Results saved to: {filename}")
    
    print("\nRECOMMENDATIONS:")
    for rec in results["recommendations"]:
        print(f"- {rec}")
    
    if results["visualizations"]:
        print(f"\nVisualizations generated:")
        for viz in results["visualizations"]:
            print(f"- {viz}")

if __name__ == "__main__":
    asyncio.run(main()) 