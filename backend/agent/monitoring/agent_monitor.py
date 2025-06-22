from typing import Dict, List, Any, Optional
from datetime import datetime
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging
from llm.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
AGENT_REQUESTS = Counter(
    'agent_requests_total',
    'Total number of agent requests',
    ['agent_name', 'status']
)

AGENT_PROCESSING_TIME = Histogram(
    'agent_processing_seconds',
    'Time spent processing agent requests',
    ['agent_name']
)

AGENT_ERRORS = Counter(
    'agent_errors_total',
    'Total number of agent errors',
    ['agent_name', 'error_type']
)

CONVERSATION_LENGTH = Gauge(
    'conversation_messages_total',
    'Total number of messages in conversation',
    ['conversation_id']
)

class AgentMonitor:
    def __init__(self, enable_metrics: bool = True, metrics_port: int = 8001):
        self.enable_metrics = enable_metrics
        self.metrics_port = metrics_port
        self._setup_metrics()
        self.agent_performance = {}
    
    def _setup_metrics(self):
        """Initialize metrics server if enabled."""
        if self.enable_metrics:
            try:
                start_http_server(self.metrics_port)
                logger.info(f"Metrics server started on port {self.metrics_port}")
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
    
    def track_request(self, agent_name: str) -> 'AgentRequestTracker':
        """Track the start of an agent request."""
        return AgentRequestTracker(self, agent_name)
    
    def record_success(self, agent_name: str, processing_time: float) -> None:
        """Record a successful agent request."""
        if self.enable_metrics:
            AGENT_REQUESTS.labels(agent_name=agent_name, status='success').inc()
            AGENT_PROCESSING_TIME.labels(agent_name=agent_name).observe(processing_time)
        
        # Update performance tracking
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {
                'total_requests': 0,
                'total_processing_time': 0.0,
                'successful_requests': 0,
                'failed_requests': 0
            }
        
        self.agent_performance[agent_name]['total_requests'] += 1
        self.agent_performance[agent_name]['successful_requests'] += 1
        self.agent_performance[agent_name]['total_processing_time'] += processing_time
    
    def record_error(
        self, 
        agent_name: str, 
        error_type: str, 
        error_message: str = None,
        conversation_id: str = None
    ) -> None:
        """Record an agent error."""
        if self.enable_metrics:
            AGENT_REQUESTS.labels(agent_name=agent_name, status='error').inc()
            AGENT_ERRORS.labels(agent_name=agent_name, error_type=error_type).inc()
        
        # Update performance tracking
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = {
                'total_requests': 0,
                'total_processing_time': 0.0,
                'successful_requests': 0,
                'failed_requests': 0
            }
        
        self.agent_performance[agent_name]['total_requests'] += 1
        self.agent_performance[agent_name]['failed_requests'] += 1
        
        # Log the error with context
        log_context = f" in conversation {conversation_id}" if conversation_id else ""
        logger.error(
            f"Agent {agent_name} error{log_context} ({error_type}): {error_message}",
            extra={
                'agent_name': agent_name,
                'error_type': error_type,
                'conversation_id': conversation_id,
                'error_message': error_message
            }
        )
    
    def record_processing_time(
        self, 
        agent_name: str, 
        processing_time: float,
        conversation_id: str = None
    ) -> None:
        """Record the processing time for an agent."""
        if self.enable_metrics:
            AGENT_PROCESSING_TIME.labels(agent_name=agent_name).observe(processing_time)
        
        logger.debug(
            f"Agent {agent_name} processed request in {processing_time:.2f}s" +
            (f" for conversation {conversation_id}" if conversation_id else "")
        )
    
    def record_processing_error(
        self,
        agent_name: str,
        error_type: str,
        error_message: str,
        conversation_id: str = None
    ) -> None:
        """Record an error that occurred during processing."""
        self.record_error(agent_name, error_type, error_message, conversation_id)
    
    def get_agent_performance(self, agent_name: str = None) -> Dict:
        """Get performance metrics for one or all agents."""
        if agent_name:
            return self.agent_performance.get(agent_name, {})
        return self.agent_performance
    
    def get_average_processing_time(self, agent_name: str = None) -> float:
        """Get the average processing time for an agent or all agents."""
        if agent_name:
            stats = self.agent_performance.get(agent_name, {})
            total = stats.get('total_processing_time', 0)
            count = stats.get('total_requests', 0)
            return total / count if count > 0 else 0.0
        
        total_time = 0.0
        total_requests = 0
        for stats in self.agent_performance.values():
            total_time += stats.get('total_processing_time', 0)
            total_requests += stats.get('total_requests', 0)
        
        return total_time / total_requests if total_requests > 0 else 0.0
    
    def update_conversation_metrics(self, conversation_id: str, message_count: int) -> None:
        """Update conversation-related metrics."""
        if self.enable_metrics:
            CONVERSATION_LENGTH.labels(conversation_id=conversation_id).set(message_count)

class AgentRequestTracker:
    def __init__(self, monitor: 'AgentMonitor', agent_name: str):
        self.monitor = monitor
        self.agent_name = agent_name
        self.start_time = time.time()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        processing_time = time.time() - self.start_time
        
        if exc_type is None:
            self.monitor.record_success(self.agent_name, processing_time)
        else:
            error_type = exc_type.__name__ if exc_type else "unknown"
            error_message = str(exc_val) if exc_val else ""
            self.monitor.record_error(
                self.agent_name,
                error_type=error_type,
                error_message=error_message
            )
        
        return False  # Don't suppress exceptions
    
    def success(self):
        """Mark the request as successful."""
        processing_time = time.time() - self.start_time
        self.monitor.record_success(self.agent_name, processing_time)
    
    def error(self, exception: Exception):
        """Mark the request as failed with an error."""
        processing_time = time.time() - self.start_time
        error_type = type(exception).__name__
        error_message = str(exception)
        self.monitor.record_error(
            self.agent_name,
            error_type=error_type,
            error_message=error_message
        )

# Global monitor instance
agent_monitor = AgentMonitor(enable_metrics=True)
