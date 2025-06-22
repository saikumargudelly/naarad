"""Analytics router for data analysis and monitoring integration."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import json
import pandas as pd
import io
from datetime import datetime, timedelta

from dependencies import get_analytics_agent
from agent.monitoring.agent_monitor import agent_monitor
from config.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

class AnalyticsRequest(BaseModel):
    """Request model for analytics processing."""
    data: str = Field(..., description="Data to analyze (CSV, JSON, or structured data)")
    analysis_type: str = Field(default="auto", description="Type of analysis")
    generate_chart: bool = Field(default=True, description="Generate visualization")
    user_id: str = Field(default="default", description="User identifier")

class MonitoringDataRequest(BaseModel):
    """Request model for monitoring data analysis."""
    time_range: str = Field(default="24h", description="Time range for analysis")
    metrics: List[str] = Field(default_factory=list, description="Specific metrics to analyze")
    user_id: str = Field(default="default", description="User identifier")

@router.post("/analytics/analyze")
async def analyze_data(request: AnalyticsRequest):
    """
    Analyze data and generate insights with optional visualization.
    """
    logger.info(f"Analyzing data for user: {request.user_id}")
    
    try:
        start_time = datetime.utcnow()
        
        # Analyze data
        analytics_agent = get_analytics_agent()
        result = await analytics_agent.analyze_data(
            data=request.data,
            analysis_type=request.analysis_type,
            generate_chart=request.generate_chart
        )
        
        process_time = (datetime.utcnow() - start_time).total_seconds()
        
        if result.get("success", False):
            return {
                "success": True,
                "analysis": result.get("analysis", ""),
                "chart": result.get("chart"),
                "data_shape": result.get("data_shape", {}),
                "analysis_type": request.analysis_type,
                "processing_time": f"{process_time:.2f}s",
                "user_id": request.user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Data analysis failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing data: {str(e)}"
        )

@router.post("/analytics/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    analysis_type: str = "auto",
    generate_chart: bool = True,
    user_id: str = "default"
):
    """
    Upload data file and analyze it.
    """
    logger.info(f"Processing uploaded data file: {file.filename}")
    
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type and process accordingly
        if file.filename.endswith('.csv'):
            # Read CSV data
            csv_text = content.decode('utf-8')
            analytics_agent = get_analytics_agent()
            result = await analytics_agent.analyze_data(
                data=csv_text,
                analysis_type=analysis_type,
                generate_chart=generate_chart
            )
        elif file.filename.endswith('.json'):
            # Read JSON data
            json_data = json.loads(content.decode('utf-8'))
            analytics_agent = get_analytics_agent()
            result = await analytics_agent.analyze_data(
                data=json.dumps(json_data),
                analysis_type=analysis_type,
                generate_chart=generate_chart
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV or JSON files."
            )
        
        if result.get("success", False):
            return {
                "success": True,
                "analysis": result.get("analysis", ""),
                "chart": result.get("chart"),
                "data_shape": result.get("data_shape", {}),
                "filename": file.filename,
                "user_id": user_id
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Data analysis failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error processing uploaded data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing uploaded data: {str(e)}"
        )

@router.post("/analytics/monitoring")
async def analyze_monitoring_data(request: MonitoringDataRequest):
    """
    Analyze monitoring data and generate insights.
    """
    logger.info(f"Analyzing monitoring data for user: {request.user_id}")
    
    try:
        # Get monitoring data from agent monitor
        monitoring_data = await get_monitoring_data(
            time_range=request.time_range,
            metrics=request.metrics
        )
        
        if not monitoring_data:
            raise HTTPException(
                status_code=404,
                detail="No monitoring data available for the specified time range"
            )
        
        # Convert monitoring data to analysis format
        analysis_data = convert_monitoring_to_analysis(monitoring_data)
        
        # Analyze the data
        analytics_agent = get_analytics_agent()
        result = await analytics_agent.analyze_data(
            data=analysis_data,
            analysis_type="monitoring",
            generate_chart=True
        )
        
        if result.get("success", False):
            return {
                "success": True,
                "analysis": result.get("analysis", ""),
                "chart": result.get("chart"),
                "monitoring_summary": {
                    "time_range": request.time_range,
                    "metrics_analyzed": request.metrics,
                    "data_points": len(monitoring_data),
                    "user_id": request.user_id
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Monitoring analysis failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error analyzing monitoring data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing monitoring data: {str(e)}"
        )

@router.get("/analytics/metrics")
async def get_available_metrics():
    """Get list of available analytics metrics."""
    analytics_agent = get_analytics_agent()
    return {
        "analysis_types": analytics_agent.get_supported_analysis_types(),
        "chart_types": analytics_agent.get_supported_chart_types(),
        "monitoring_metrics": [
            "response_time",
            "success_rate",
            "error_rate",
            "user_engagement",
            "agent_usage",
            "conversation_length"
        ]
    }

@router.get("/analytics/health")
async def analytics_health_check():
    """Health check for analytics services."""
    try:
        # Test basic functionality with sample data
        sample_data = "name,value\nA,10\nB,20\nC,15"
        analytics_agent = get_analytics_agent()
        result = await analytics_agent.analyze_data(
            data=sample_data,
            analysis_type="descriptive",
            generate_chart=False
        )
        
        return {
            "status": "healthy",
            "service": "analytics-processing",
            "timestamp": datetime.utcnow().isoformat(),
            "test_result": result.get("success", False),
            "supported_analysis_types": len(analytics_agent.get_supported_analysis_types()),
            "supported_chart_types": len(analytics_agent.get_supported_chart_types())
        }
    except Exception as e:
        logger.error(f"Analytics health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "analytics-processing",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def get_monitoring_data(time_range: str, metrics: List[str]) -> List[Dict[str, Any]]:
    """Get monitoring data from agent monitor."""
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_range == "24h":
            start_time = end_time - timedelta(days=1)
        elif time_range == "7d":
            start_time = end_time - timedelta(days=7)
        elif time_range == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(hours=1)  # Default to 1 hour
        
        # Get metrics from agent monitor
        monitoring_data = []
        
        # This would integrate with your existing agent_monitor
        # For now, we'll return sample data
        sample_data = [
            {"timestamp": "2024-01-15T10:00:00Z", "response_time": 1.2, "success_rate": 0.95},
            {"timestamp": "2024-01-15T11:00:00Z", "response_time": 1.1, "success_rate": 0.98},
            {"timestamp": "2024-01-15T12:00:00Z", "response_time": 1.3, "success_rate": 0.92},
        ]
        
        return sample_data
        
    except Exception as e:
        logger.error(f"Error getting monitoring data: {str(e)}")
        return []

def convert_monitoring_to_analysis(monitoring_data: List[Dict[str, Any]]) -> str:
    """Convert monitoring data to analysis format."""
    try:
        # Convert to CSV format for analysis
        if not monitoring_data:
            return ""
        
        # Get all unique keys
        all_keys = set()
        for item in monitoring_data:
            all_keys.update(item.keys())
        
        # Create CSV header
        csv_lines = [','.join(all_keys)]
        
        # Add data rows
        for item in monitoring_data:
            row = []
            for key in all_keys:
                value = item.get(key, '')
                row.append(str(value))
            csv_lines.append(','.join(row))
        
        return '\n'.join(csv_lines)
        
    except Exception as e:
        logger.error(f"Error converting monitoring data: {str(e)}")
        return "" 