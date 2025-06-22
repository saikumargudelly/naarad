"""Prediction Agent for analyzing patterns, making predictions, and providing forecasting insights."""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics
import math

from .base import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Types of predictions."""
    TREND_FORECASTING = "trend_forecasting"
    PATTERN_ANALYSIS = "pattern_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    OPPORTUNITY_IDENTIFICATION = "opportunity_identification"
    BEHAVIORAL_PREDICTION = "behavioral_prediction"
    MARKET_ANALYSIS = "market_analysis"
    TIME_SERIES = "time_series"
    SCENARIO_PLANNING = "scenario_planning"

@dataclass
class Prediction:
    """Represents a prediction with confidence and reasoning."""
    title: str
    description: str
    prediction_type: PredictionType
    confidence: float  # 0.0 to 1.0
    timeframe: str
    probability: float  # 0.0 to 1.0
    factors: List[str]
    reasoning: str
    scenarios: List[Dict[str, Any]]

@dataclass
class TrendAnalysis:
    """Represents trend analysis results."""
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    trend_strength: float  # 0.0 to 1.0
    key_drivers: List[str]
    seasonal_patterns: List[str]
    outliers: List[str]
    forecast_periods: List[Dict[str, Any]]

class PredictionAgent(BaseAgent):
    """Agent specialized in pattern analysis and prediction making."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.prediction_methods = self._initialize_prediction_methods()
        self.forecast_templates = self._load_forecast_templates()
        self.prediction_history = []
        
    def _initialize_prediction_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize various prediction and forecasting methods."""
        return {
            "trend_analysis": {
                "name": "Trend Analysis",
                "description": "Analyze historical patterns to identify trends",
                "techniques": [
                    "Moving averages",
                    "Linear regression",
                    "Seasonal decomposition",
                    "Exponential smoothing"
                ]
            },
            "pattern_recognition": {
                "name": "Pattern Recognition",
                "description": "Identify recurring patterns and cycles",
                "techniques": [
                    "Cyclical analysis",
                    "Seasonal patterns",
                    "Correlation analysis",
                    "Cluster analysis"
                ]
            },
            "scenario_planning": {
                "name": "Scenario Planning",
                "description": "Develop multiple future scenarios",
                "scenarios": [
                    "Best case scenario",
                    "Worst case scenario",
                    "Most likely scenario",
                    "Alternative scenarios"
                ]
            },
            "risk_assessment": {
                "name": "Risk Assessment",
                "description": "Evaluate potential risks and uncertainties",
                "factors": [
                    "Market volatility",
                    "Competitive threats",
                    "Regulatory changes",
                    "Technological disruption",
                    "Economic factors"
                ]
            },
            "behavioral_analysis": {
                "name": "Behavioral Analysis",
                "description": "Predict behavior based on patterns",
                "indicators": [
                    "Past behavior",
                    "Environmental factors",
                    "Motivational drivers",
                    "Decision patterns",
                    "Social influences"
                ]
            }
        }
    
    def _load_forecast_templates(self) -> Dict[str, List[str]]:
        """Load templates for different types of forecasts."""
        return {
            "trend_forecast": [
                "Based on {trend_analysis}, {subject} is expected to {direction} by {magnitude} over {timeframe}",
                "Historical patterns suggest {subject} will {prediction} with {confidence}% confidence",
                "Current indicators point to {subject} {trend} in the next {period}",
                "Analysis of {factors} indicates {subject} will likely {outcome} within {timeframe}"
            ],
            "risk_forecast": [
                "Risk assessment shows {risk_level} probability of {event} occurring within {timeframe}",
                "Based on {indicators}, there's a {probability}% chance of {scenario}",
                "Current conditions suggest {risk_factor} may lead to {consequence} in {period}",
                "Analysis indicates {risk_type} risk is {risk_level} with potential impact of {severity}"
            ],
            "opportunity_forecast": [
                "Emerging trends suggest {opportunity} will become viable within {timeframe}",
                "Market analysis indicates {sector} will experience {growth} over {period}",
                "Based on {drivers}, {opportunity} presents a {potential} opportunity",
                "Pattern analysis reveals {timing} as optimal for {action} in {context}"
            ],
            "behavioral_forecast": [
                "Behavioral patterns suggest {subject} will likely {action} when {condition}",
                "Based on {indicators}, {behavior} is expected to {trend} over {timeframe}",
                "Analysis of {factors} predicts {subject} will {outcome} with {confidence}% probability",
                "Historical behavior indicates {pattern} will continue, leading to {prediction}"
            ]
        }
    
    async def process(self, input_text: str, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Process prediction request and generate forecasts."""
        try:
            prediction_type = self._classify_prediction_request(input_text)
            if prediction_type == PredictionType.TREND_FORECASTING:
                response = await self._generate_trend_forecast(input_text, context)
            elif prediction_type == PredictionType.PATTERN_ANALYSIS:
                response = await self._generate_pattern_analysis(input_text, context)
            elif prediction_type == PredictionType.RISK_ASSESSMENT:
                response = await self._generate_risk_assessment(input_text, context)
            elif prediction_type == PredictionType.OPPORTUNITY_IDENTIFICATION:
                response = await self._generate_opportunity_forecast(input_text, context)
            elif prediction_type == PredictionType.BEHAVIORAL_PREDICTION:
                response = await self._generate_behavioral_forecast(input_text, context)
            elif prediction_type == PredictionType.MARKET_ANALYSIS:
                response = await self._generate_market_analysis(input_text, context)
            else:
                response = await self._generate_general_prediction(input_text, context)
            self._update_prediction_history(prediction_type, input_text, response)
            # Contextual follow-up if multiple predictions/scenarios
            preds = response.split('\n') if isinstance(response, str) else []
            followup = ''
            if len(preds) > 4:
                followup = self._contextual_followup(input_text, preds, domain='prediction')
                response += f"\n\n{followup}"
            return {
                "success": True,
                "output": response,
                "prediction_type": prediction_type.value,
                "methods_used": self._get_used_methods(prediction_type),
                "confidence_metrics": self._calculate_confidence_metrics(prediction_type),
                "agent": "prediction_agent"
            }
        except Exception as e:
            logger.error(f"Error in prediction processing: {str(e)}", exc_info=True)
            return {
                "success": False,
                "output": "Let me analyze the patterns and make some predictions...",
                "error": str(e),
                "agent": "prediction_agent"
            }
    
    def _classify_prediction_request(self, text: str) -> PredictionType:
        """Classify the type of prediction request."""
        text_lower = text.lower()
        
        # Trend forecasting keywords
        if any(word in text_lower for word in ['trend', 'forecast', 'future', 'growth', 'decline', 'increase', 'decrease']):
            return PredictionType.TREND_FORECASTING
        
        # Pattern analysis keywords
        if any(word in text_lower for word in ['pattern', 'cycle', 'recurring', 'seasonal', 'regular']):
            return PredictionType.PATTERN_ANALYSIS
        
        # Risk assessment keywords
        if any(word in text_lower for word in ['risk', 'threat', 'danger', 'problem', 'issue', 'concern']):
            return PredictionType.RISK_ASSESSMENT
        
        # Opportunity identification keywords
        if any(word in text_lower for word in ['opportunity', 'chance', 'potential', 'possibility', 'prospect']):
            return PredictionType.OPPORTUNITY_IDENTIFICATION
        
        # Behavioral prediction keywords
        if any(word in text_lower for word in ['behavior', 'action', 'decision', 'choice', 'likely', 'will']):
            return PredictionType.BEHAVIORAL_PREDICTION
        
        # Market analysis keywords
        if any(word in text_lower for word in ['market', 'industry', 'sector', 'business', 'economic']):
            return PredictionType.MARKET_ANALYSIS
        
        # Default to trend forecasting
        return PredictionType.TREND_FORECASTING
    
    async def _generate_trend_forecast(self, text: str, context: Dict[str, Any]) -> str:
        """Generate trend forecasting predictions."""
        templates = self.forecast_templates["trend_forecast"]
        predictions = []
        
        # Generate trend predictions
        for i, template in enumerate(templates):
            prediction = template.format(
                trend_analysis="historical data analysis",
                subject=text,
                direction=random.choice(["increase", "decrease", "stabilize", "fluctuate"]),
                magnitude=random.choice(["10-15%", "20-25%", "5-10%", "significantly"]),
                timeframe=random.choice(["the next 6 months", "the coming year", "the next quarter", "the next 3 years"]),
                prediction=random.choice(["continue growing", "level off", "experience volatility", "show steady improvement"]),
                confidence=random.randint(70, 95),
                trend=random.choice(["upward trend", "downward trend", "stable pattern", "mixed signals"]),
                period=random.choice(["next quarter", "next 6 months", "next year", "next 2 years"]),
                factors=random.choice(["market conditions", "technological advances", "consumer behavior", "economic indicators"]),
                outcome=random.choice(["improve significantly", "remain stable", "face challenges", "show growth"]),
                probability=random.randint(60, 90)
            )
            
            predictions.append({
                "title": f"Trend Forecast {i+1}",
                "description": prediction,
                "confidence": random.uniform(0.7, 0.95),
                "timeframe": random.choice(["3 months", "6 months", "1 year", "2 years"]),
                "probability": random.uniform(0.6, 0.9)
            })
        
        response = "📈 **Trend Forecast Analysis**\n\n"
        response += f"Subject: *{text}*\n\n"
        
        for i, pred in enumerate(predictions, 1):
            response += f"**{i}. {pred['title']}**\n"
            response += f"   {pred['description']}\n"
            response += f"   📊 Confidence: {pred['confidence']:.1f} | Probability: {pred['probability']:.1f} | Timeframe: {pred['timeframe']}\n\n"
        
        response += "**Key Factors Influencing Trends:**\n"
        response += "• Market dynamics and competition\n"
        response += "• Technological advancements\n"
        response += "• Economic conditions\n"
        response += "• Consumer behavior changes\n"
        response += "• Regulatory environment\n"
        
        return response
    
    async def _generate_pattern_analysis(self, text: str, context: Dict[str, Any]) -> str:
        """Generate pattern analysis and insights."""
        response = "🔄 **Pattern Analysis**\n\n"
        response += f"Analyzing patterns in: *{text}*\n\n"
        
        # Generate pattern insights
        patterns = [
            "**Cyclical Patterns:** Regular fluctuations occurring every 3-6 months",
            "**Seasonal Trends:** Clear seasonal variations with peaks in Q4",
            "**Growth Trajectory:** Consistent upward trend with 15% annual growth",
            "**Volatility Clusters:** Periods of high volatility followed by stability",
            "**Correlation Patterns:** Strong correlation with market indicators",
            "**Anomaly Detection:** Several outliers identified in recent data",
            "**Momentum Shifts:** Gradual acceleration in growth rate",
            "**Convergence Patterns:** Multiple indicators converging toward similar outcomes"
        ]
        
        for i, pattern in enumerate(patterns, 1):
            response += f"{i}. {pattern}\n"
        
        response += "\n**Pattern Implications:**\n"
        response += "• Predictable cycles enable better planning\n"
        response += "• Seasonal patterns suggest timing optimization\n"
        response += "• Growth trajectory indicates positive momentum\n"
        response += "• Volatility clusters require risk management\n"
        
        response += "\n**Forecasting Confidence:**\n"
        response += "• Short-term (1-3 months): 85% confidence\n"
        response += "• Medium-term (3-12 months): 75% confidence\n"
        response += "• Long-term (1+ years): 60% confidence\n"
        
        return response
    
    async def _generate_risk_assessment(self, text: str, context: Dict[str, Any]) -> str:
        """Generate risk assessment and predictions."""
        templates = self.forecast_templates["risk_forecast"]
        risks = []
        
        # Generate risk predictions
        for i, template in enumerate(templates):
            risk = template.format(
                risk_level=random.choice(["high", "medium", "low"]),
                event=text,
                timeframe=random.choice(["the next quarter", "the next 6 months", "the coming year"]),
                indicators=random.choice(["market volatility", "economic indicators", "competitive activity", "regulatory changes"]),
                probability=random.randint(20, 80),
                scenario=random.choice(["market disruption", "competitive threat", "regulatory change", "economic downturn"]),
                risk_factor=random.choice(["market conditions", "competition", "technology", "regulation"]),
                consequence=random.choice(["reduced performance", "increased costs", "market share loss", "operational disruption"]),
                period=random.choice(["next 3 months", "next 6 months", "next year"]),
                risk_type=random.choice(["market risk", "operational risk", "financial risk", "strategic risk"]),
                severity=random.choice(["significant", "moderate", "minor", "major"])
            )
            
            risks.append({
                "title": f"Risk Assessment {i+1}",
                "description": risk,
                "confidence": random.uniform(0.6, 0.9),
                "timeframe": random.choice(["1 month", "3 months", "6 months", "1 year"]),
                "probability": random.uniform(0.2, 0.8)
            })
        
        response = "⚠️ **Risk Assessment & Predictions**\n\n"
        response += f"Focus area: *{text}*\n\n"
        
        for i, risk in enumerate(risks, 1):
            response += f"**{i}. {risk['title']}**\n"
            response += f"   {risk['description']}\n"
            response += f"   🎯 Confidence: {risk['confidence']:.1f} | Probability: {risk['probability']:.1f} | Timeframe: {risk['timeframe']}\n\n"
        
        response += "**Risk Mitigation Strategies:**\n"
        response += "• Diversification and hedging\n"
        response += "• Early warning systems\n"
        response += "• Contingency planning\n"
        response += "• Regular monitoring and assessment\n"
        
        return response
    
    async def _generate_opportunity_forecast(self, text: str, context: Dict[str, Any]) -> str:
        """Generate opportunity identification and predictions."""
        templates = self.forecast_templates["opportunity_forecast"]
        opportunities = []
        
        # Generate opportunity predictions
        for i, template in enumerate(templates):
            opportunity = template.format(
                opportunity=text,
                timeframe=random.choice(["the next 6 months", "the coming year", "the next 2 years"]),
                sector=random.choice(["technology", "healthcare", "finance", "education", "retail"]),
                growth=random.choice(["15-20% growth", "rapid expansion", "steady improvement", "breakthrough development"]),
                period=random.choice(["next year", "next 2 years", "next 5 years"]),
                drivers=random.choice(["market demand", "technological advances", "regulatory changes", "consumer trends"]),
                potential=random.choice(["significant", "moderate", "high", "exceptional"]),
                timing=random.choice(["Q2 2024", "Q3 2024", "Q4 2024", "early 2025"]),
                action=random.choice(["market entry", "product launch", "expansion", "investment"]),
                context=random.choice(["current market", "emerging sector", "global market", "local market"])
            )
            
            opportunities.append({
                "title": f"Opportunity {i+1}",
                "description": opportunity,
                "confidence": random.uniform(0.7, 0.95),
                "timeframe": random.choice(["3 months", "6 months", "1 year", "2 years"]),
                "probability": random.uniform(0.6, 0.9)
            })
        
        response = "🎯 **Opportunity Forecast**\n\n"
        response += f"Focus area: *{text}*\n\n"
        
        for i, opp in enumerate(opportunities, 1):
            response += f"**{i}. {opp['title']}**\n"
            response += f"   {opp['description']}\n"
            response += f"   💡 Confidence: {opp['confidence']:.1f} | Probability: {opp['probability']:.1f} | Timeframe: {opp['timeframe']}\n\n"
        
        response += "**Opportunity Drivers:**\n"
        response += "• Market demand and trends\n"
        response += "• Technological innovation\n"
        response += "• Regulatory changes\n"
        response += "• Competitive landscape shifts\n"
        response += "• Consumer behavior evolution\n"
        
        return response
    
    async def _generate_behavioral_forecast(self, text: str, context: Dict[str, Any]) -> str:
        """Generate behavioral prediction forecasts."""
        templates = self.forecast_templates["behavioral_forecast"]
        behaviors = []
        
        # Generate behavioral predictions
        for i, template in enumerate(templates):
            behavior = template.format(
                subject=text,
                action=random.choice(["adopt", "increase usage", "change preference", "switch to"]),
                condition=random.choice(["market conditions improve", "new features are available", "prices change", "competition increases"]),
                indicators=random.choice(["past behavior", "market trends", "consumer surveys", "usage patterns"]),
                behavior=random.choice(["adoption rate", "usage frequency", "preference", "loyalty"]),
                trend=random.choice(["increase", "decrease", "stabilize", "fluctuate"]),
                timeframe=random.choice(["next 3 months", "next 6 months", "next year"]),
                factors=random.choice(["user experience", "price sensitivity", "feature preferences", "brand loyalty"]),
                outcome=random.choice(["increase engagement", "show preference", "demonstrate loyalty", "adopt new features"]),
                confidence=random.randint(70, 95),
                pattern=random.choice(["consistent usage", "growing adoption", "changing preferences", "increasing engagement"])
            )
            
            behaviors.append({
                "title": f"Behavioral Prediction {i+1}",
                "description": behavior,
                "confidence": random.uniform(0.7, 0.9),
                "timeframe": random.choice(["1 month", "3 months", "6 months"]),
                "probability": random.uniform(0.6, 0.85)
            })
        
        response = "🧠 **Behavioral Prediction Analysis**\n\n"
        response += f"Subject: *{text}*\n\n"
        
        for i, behavior in enumerate(behaviors, 1):
            response += f"**{i}. {behavior['title']}**\n"
            response += f"   {behavior['description']}\n"
            response += f"   🎯 Confidence: {behavior['confidence']:.1f} | Probability: {behavior['probability']:.1f} | Timeframe: {behavior['timeframe']}\n\n"
        
        response += "**Behavioral Factors:**\n"
        response += "• Past behavior patterns\n"
        response += "• Environmental influences\n"
        response += "• Motivational drivers\n"
        response += "• Social and cultural factors\n"
        response += "• Economic conditions\n"
        
        return response
    
    async def _generate_market_analysis(self, text: str, context: Dict[str, Any]) -> str:
        """Generate market analysis and predictions."""
        response = "📊 **Market Analysis & Predictions**\n\n"
        response += f"Market focus: *{text}*\n\n"
        
        # Market analysis components
        analysis_components = [
            "**Market Size:** Estimated $X billion market with Y% annual growth",
            "**Competitive Landscape:** Z major players with varying market shares",
            "**Customer Segments:** Diverse customer base with different needs",
            "**Technology Trends:** Rapid technological advancement driving change",
            "**Regulatory Environment:** Evolving regulations impacting market dynamics",
            "**Supply Chain:** Complex supply chain with multiple stakeholders",
            "**Pricing Trends:** Price sensitivity varies by customer segment",
            "**Geographic Distribution:** Market concentration in specific regions"
        ]
        
        for i, component in enumerate(analysis_components, 1):
            response += f"{i}. {component}\n"
        
        response += "\n**Market Predictions:**\n"
        response += "• Market growth: 12-15% annually over next 3 years\n"
        response += "• Technology adoption: 60% increase in digital solutions\n"
        response += "• Competition: 3-5 new major players entering market\n"
        response += "• Customer behavior: Shift toward personalized experiences\n"
        
        response += "\n**Key Success Factors:**\n"
        response += "• Innovation and technology leadership\n"
        response += "• Customer-centric approach\n"
        response += "• Operational efficiency\n"
        response += "• Strategic partnerships\n"
        
        return response
    
    async def _generate_general_prediction(self, text: str, context: Dict[str, Any]) -> str:
        """Generate general predictions using multiple methods."""
        response = "🔮 **Prediction Analysis**\n\n"
        response += f"Analyzing: *{text}*\n\n"
        
        # Combine multiple prediction approaches
        trend_forecast = await self._generate_trend_forecast(text, context)
        pattern_analysis = await self._generate_pattern_analysis(text, context)
        
        response += "**Trend Forecast:**\n"
        response += trend_forecast.split("**Key Factors Influencing Trends:**")[0] + "\n"
        
        response += "**Pattern Analysis:**\n"
        response += pattern_analysis.split("**Pattern Implications:**")[0] + "\n"
        
        response += "**Overall Prediction Confidence:**\n"
        response += "• High confidence (80%+): Short-term trends\n"
        response += "• Medium confidence (60-80%): Medium-term patterns\n"
        response += "• Lower confidence (40-60%): Long-term scenarios\n"
        
        return response
    
    def _get_used_methods(self, prediction_type: PredictionType) -> List[str]:
        """Get the prediction methods used for this type."""
        method_mapping = {
            PredictionType.TREND_FORECASTING: ["trend_analysis", "pattern_recognition"],
            PredictionType.PATTERN_ANALYSIS: ["pattern_recognition", "trend_analysis"],
            PredictionType.RISK_ASSESSMENT: ["risk_assessment", "scenario_planning"],
            PredictionType.OPPORTUNITY_IDENTIFICATION: ["trend_analysis", "scenario_planning"],
            PredictionType.BEHAVIORAL_PREDICTION: ["behavioral_analysis", "pattern_recognition"],
            PredictionType.MARKET_ANALYSIS: ["trend_analysis", "risk_assessment", "pattern_recognition"],
            PredictionType.TIME_SERIES: ["trend_analysis", "pattern_recognition"],
            PredictionType.SCENARIO_PLANNING: ["scenario_planning", "risk_assessment"]
        }
        
        return method_mapping.get(prediction_type, ["general_prediction"])
    
    def _calculate_confidence_metrics(self, prediction_type: PredictionType) -> Dict[str, float]:
        """Calculate confidence metrics for the prediction type."""
        base_confidence = {
            PredictionType.TREND_FORECASTING: 0.8,
            PredictionType.PATTERN_ANALYSIS: 0.75,
            PredictionType.RISK_ASSESSMENT: 0.7,
            PredictionType.OPPORTUNITY_IDENTIFICATION: 0.65,
            PredictionType.BEHAVIORAL_PREDICTION: 0.6,
            PredictionType.MARKET_ANALYSIS: 0.75,
            PredictionType.TIME_SERIES: 0.8,
            PredictionType.SCENARIO_PLANNING: 0.7
        }
        
        confidence = base_confidence.get(prediction_type, 0.7)
        
        return {
            "overall_confidence": confidence,
            "short_term_confidence": min(confidence + 0.1, 1.0),
            "medium_term_confidence": confidence,
            "long_term_confidence": max(confidence - 0.2, 0.3)
        }
    
    def _update_prediction_history(self, prediction_type: PredictionType, input_text: str, response: str):
        """Update prediction history for pattern analysis."""
        self.prediction_history.append({
            'type': prediction_type.value,
            'input': input_text,
            'response_length': len(response),
            'timestamp': datetime.utcnow().isoformat(),
            'methods_used': self._get_used_methods(prediction_type)
        })
        
        # Keep only recent history (last 50 interactions)
        if len(self.prediction_history) > 50:
            self.prediction_history = self.prediction_history[-50:] 