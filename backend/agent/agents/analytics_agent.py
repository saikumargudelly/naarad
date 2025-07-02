"""Analytics Agent for real-time data analysis and insights."""

from typing import Dict, Any, Optional, Union, List, Tuple
import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os

# LangChain imports
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Local imports
from .base import BaseAgent, AgentConfig
from llm.config import settings
from agent.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class DataAnalysisTool(BaseTool):
    """Tool for analyzing data and generating insights."""
    
    name: str = "data_analysis"
    description: str = "Analyze data and generate insights, charts, and statistics"
    
    def _run(self, data: Union[str, Dict, List], analysis_type: str = "auto") -> str:
        """Analyze data and return insights.
        
        Args:
            data: Data to analyze (CSV string, JSON, or list)
            analysis_type: Type of analysis (auto, descriptive, trend, correlation, etc.)
            
        Returns:
            str: Analysis results with insights
        """
        try:
            # Parse data
            if isinstance(data, str):
                # Try to parse as CSV first
                try:
                    df = pd.read_csv(io.StringIO(data))
                except:
                    # Try JSON
                    try:
                        df = pd.DataFrame(json.loads(data))
                    except:
                        return "Error: Could not parse data format"
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                return "Error: Unsupported data format"
            
            # Perform analysis based on type
            if analysis_type == "auto":
                return self._auto_analyze(df)
            elif analysis_type == "descriptive":
                return self._descriptive_analysis(df)
            elif analysis_type == "trend":
                return self._trend_analysis(df)
            elif analysis_type == "correlation":
                return self._correlation_analysis(df)
            else:
                return self._auto_analyze(df)
                
        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            return f"Error analyzing data: {str(e)}"
    
    def _auto_analyze(self, df: pd.DataFrame) -> str:
        """Perform automatic analysis based on data characteristics."""
        insights = []
        
        # Basic info
        insights.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
        
        # Data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            insights.append(f"Numeric columns: {list(numeric_cols)}")
            insights.append(self._descriptive_analysis(df[numeric_cols]))
        
        if len(categorical_cols) > 0:
            insights.append(f"Categorical columns: {list(categorical_cols)}")
            for col in categorical_cols[:3]:  # Limit to first 3
                value_counts = df[col].value_counts()
                insights.append(f"Top values in {col}: {dict(value_counts.head(3))}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            insights.append(f"Missing values: {dict(missing[missing > 0])}")
        
        return "\n".join(insights)
    
    def _descriptive_analysis(self, df: pd.DataFrame) -> str:
        """Perform descriptive statistical analysis."""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            return "No numeric columns for descriptive analysis"
        
        stats = numeric_df.describe()
        return f"Descriptive Statistics:\n{stats.to_string()}"
    
    def _trend_analysis(self, df: pd.DataFrame) -> str:
        """Analyze trends in time series data."""
        # Look for date/time columns
        date_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                continue
        
        if not date_cols:
            return "No date/time columns found for trend analysis"
        
        insights = []
        for col in date_cols[:2]:  # Limit to first 2 date columns
            df[col] = pd.to_datetime(df[col])
            df_sorted = df.sort_values(col)
            
            # Analyze trends in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                correlation = df_sorted[num_col].corr(pd.to_numeric(df_sorted[col]))
                insights.append(f"Correlation between {col} and {num_col}: {correlation:.3f}")
        
        return "\n".join(insights) if insights else "No significant trends found"
    
    def _correlation_analysis(self, df: pd.DataFrame) -> str:
        """Analyze correlations between variables."""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return "Need at least 2 numeric columns for correlation analysis"
        
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append(f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]}: {corr_val:.3f}")
        
        if strong_corr:
            return f"Strong correlations found:\n" + "\n".join(strong_corr)
        else:
            return "No strong correlations found (|r| > 0.7)"

class ChartGenerationTool(BaseTool):
    """Tool for generating charts and visualizations."""
    
    name: str = "chart_generation"
    description: str = "Generate charts and visualizations from data"
    
    def _run(self, data: Union[str, Dict, List], chart_type: str = "auto") -> str:
        """Generate a chart from data.
        
        Args:
            data: Data to visualize
            chart_type: Type of chart (auto, line, bar, scatter, histogram, etc.)
            
        Returns:
            str: Base64 encoded chart image
        """
        try:
            # Parse data
            if isinstance(data, str):
                try:
                    df = pd.read_csv(io.StringIO(data))
                except:
                    df = pd.DataFrame(json.loads(data))
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                return "Error: Could not parse data format"
            
            # Generate chart
            if chart_type == "auto":
                chart_data = self._auto_chart(df)
            elif chart_type == "line":
                chart_data = self._line_chart(df)
            elif chart_type == "bar":
                chart_data = self._bar_chart(df)
            elif chart_type == "scatter":
                chart_data = self._scatter_chart(df)
            elif chart_type == "histogram":
                chart_data = self._histogram_chart(df)
            else:
                chart_data = self._auto_chart(df)
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return f"Error generating chart: {str(e)}"
    
    def _auto_chart(self, df: pd.DataFrame) -> str:
        """Automatically choose the best chart type."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) >= 2:
            return self._scatter_chart(df)
        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            return self._bar_chart(df)
        elif len(numeric_cols) == 1:
            return self._histogram_chart(df)
        else:
            return self._bar_chart(df)
    
    def _line_chart(self, df: pd.DataFrame) -> str:
        """Generate a line chart."""
        plt.figure(figsize=(10, 6))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return "No numeric columns for line chart"
        
        for col in numeric_cols[:3]:  # Limit to first 3 columns
            plt.plot(df.index, df[col], label=col, marker='o')
        
        plt.title('Line Chart')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return self._save_chart()
    
    def _bar_chart(self, df: pd.DataFrame) -> str:
        """Generate a bar chart."""
        plt.figure(figsize=(10, 6))
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            # Categorical vs numeric
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            df.groupby(cat_col)[num_col].mean().plot(kind='bar')
            plt.title(f'Average {num_col} by {cat_col}')
        elif len(numeric_cols) > 0:
            # Just numeric columns
            df[numeric_cols[0]].plot(kind='bar')
            plt.title(f'Bar Chart - {numeric_cols[0]}')
        else:
            return "No suitable data for bar chart"
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return self._save_chart()
    
    def _scatter_chart(self, df: pd.DataFrame) -> str:
        """Generate a scatter plot."""
        plt.figure(figsize=(10, 6))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return "Need at least 2 numeric columns for scatter plot"
        
        plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
        plt.grid(True, alpha=0.3)
        
        return self._save_chart()
    
    def _histogram_chart(self, df: pd.DataFrame) -> str:
        """Generate a histogram."""
        plt.figure(figsize=(10, 6))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return "No numeric columns for histogram"
        
        df[numeric_cols[0]].hist(bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel(numeric_cols[0])
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {numeric_cols[0]}')
        plt.grid(True, alpha=0.3)
        
        return self._save_chart()
    
    def _save_chart(self) -> str:
        """Save chart to base64 string."""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()  # Close the figure to free memory
        
        return f"data:image/png;base64,{image_base64}"

class AnalyticsAgent(BaseAgent):
    """Agent specialized in real-time data analysis and insights.
    Modular, stateless, and uses injected memory manager for context/state.
    """
    def __init__(self, config: Dict[str, Any], memory_manager: MemoryManager = None):
        """Initialize the analytics agent with configuration.
        Args:
            config: The configuration for the agent. Must be a dictionary.
            memory_manager: The memory manager for the agent.
        """
        # Set default values if not provided
        default_config = {
            'name': 'analytics_agent',
            'description': 'Data analysis and insights agent',
            'model_name': settings.REASONING_MODEL,
            'temperature': 0.2,
            'system_prompt': """You are an analytics agent. Your job is to analyze data, generate insights, and create visualizations.\n\nYou can handle CSV, JSON, and structured data. Use the available tools to perform statistical analysis, generate charts, and summarize findings.""",
            'max_iterations': 5
        }
        # Update with any provided config values
        default_config.update(config)
        config = default_config
        super().__init__(config)
        
        self.memory_manager = memory_manager
        logger.info(f"AnalyticsAgent initialized with memory_manager: {bool(memory_manager)}")
        
        # Add analytics-specific tools
        self.data_analysis = DataAnalysisTool()
        self.chart_generation = ChartGenerationTool()
        
        # Analytics-specific configuration
        self.supported_analysis_types = [
            "auto", "descriptive", "trend", "correlation", 
            "regression", "classification", "clustering"
        ]
        self.supported_chart_types = [
            "auto", "line", "bar", "scatter", "histogram", 
            "box", "heatmap", "pie", "area"
        ]
        
        logger.info("Analytics Agent initialized")
    
    async def process(self, input_text: str, context: Dict[str, Any] = None, conversation_id: str = None, user_id: str = None, conversation_memory=None, **kwargs) -> Dict[str, Any]:
        logger.info(f"AnalyticsAgent.process called | input_text: {input_text} | conversation_id: {conversation_id} | user_id: {user_id}")
        try:
            chat_history = kwargs.get('chat_history', '')
            topic = None
            intent = None
            last_user_message = None
            if conversation_memory:
                topic = conversation_memory.topics[-1] if conversation_memory.topics else None
                intent = conversation_memory.intents[-1] if conversation_memory.intents else None
                for msg in reversed(conversation_memory.messages):
                    if msg['role'] == 'user':
                        last_user_message = msg['content']
                        break
            # Compose a context-aware prompt
            context_snippets = "\n".join([
                f"{m['role'].capitalize()}: {m['content']}" for m in conversation_memory.messages[-6:]
            ]) if conversation_memory else ""
            system_prompt = (
                "You are an expert analytics assistant. Use the conversation context, topic, and intent to answer the user's analytics question as accurately and helpfully as possible. "
                "If the user is following up, use the previous context to disambiguate."
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Conversation context:\n{context_snippets}\n\nTopic: {topic}\nIntent: {intent}\n\nUser question: {input_text}")
            ]
            llm = ChatGroq(
                temperature=0.2,
                model_name=settings.REASONING_MODEL,
                groq_api_key=os.getenv('GROQ_API_KEY')
            )
            result = await llm.ainvoke(messages)
            return {"output": result.content.strip(), "metadata": {"success": True, "topic": topic, "intent": intent}}
        except Exception as e:
            logger.error(f"Async error in analytics process: {str(e)}", exc_info=True)
            return {
                'output': f"I encountered an error while processing your analytics request: {str(e)}",
                'metadata': {
                    'error': str(e),
                    'success': False
                }
            }
    
    async def analyze_data(
        self, 
        data: Union[str, Dict, List],
        analysis_type: str = "auto",
        generate_chart: bool = True
    ) -> Dict[str, Any]:
        """Analyze data and generate insights with optional visualization."""
        try:
            logger.info(f"[ANALYTICS_AGENT] analyze_data called | data: {data} | analysis_type: {analysis_type} | generate_chart: {generate_chart}")
            # Perform data analysis
            analysis_results = self.data_analysis._run(data, analysis_type)
            # Aggressive error detection
            error_phrases = ["error", "could not", "unsupported", "invalid"]
            if (not isinstance(analysis_results, str) or not analysis_results.strip() or
                any(phrase in analysis_results.lower() for phrase in error_phrases)):
                logger.info(f"[ANALYTICS_AGENT] Returning failure: {analysis_results}")
                return {
                    "success": False,
                    "error": str(analysis_results)
                }
            # Generate chart if requested
            chart_data = None
            if generate_chart:
                chart_data = self.chart_generation._run(data, "auto")
            # Get data shape and ensure JSON-serializable
            data_shape = self._get_data_shape(data)
            result = {
                "success": True,
                "analysis": analysis_results,
                "chart": chart_data,
                "data_shape": data_shape,
                "analysis_type": analysis_type,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "analysis_type": analysis_type
                }
            }
            logger.info(f"[ANALYTICS_AGENT] Returning success: {result}")
            return result
        except Exception as e:
            logger.error(f"[ANALYTICS_AGENT] Exception in analyze_data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_data_shape(self, data: Union[str, Dict, List]) -> Dict[str, Any]:
        """Get basic information about the data shape."""
        try:
            if isinstance(data, str):
                try:
                    df = pd.read_csv(io.StringIO(data))
                except:
                    df = pd.DataFrame(json.loads(data))
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                return {"error": "Unknown data format"}
            # Convert dtypes to string for JSON serialization
            return {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": list(df.columns),
                "data_types": {k: str(v) for k, v in df.dtypes.items()}
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_supported_analysis_types(self) -> List[str]:
        """Get list of supported analysis types."""
        return self.supported_analysis_types
    
    def get_supported_chart_types(self) -> List[str]:
        """Get list of supported chart types."""
        return self.supported_chart_types 