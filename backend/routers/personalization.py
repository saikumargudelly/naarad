"""Personalization router for user preference learning and validation."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import json
from datetime import datetime, timedelta

from dependencies import get_personalization_agent
from config.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

class UserInteractionRequest(BaseModel):
    """Request model for user interaction learning."""
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., description="User message")
    response: str = Field(..., description="Agent response")
    interaction_type: str = Field(default="chat", description="Type of interaction")
    timestamp: Optional[str] = Field(default=None, description="Interaction timestamp")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")

class PreferenceUpdateRequest(BaseModel):
    """Request model for preference updates."""
    user_id: str = Field(..., description="User identifier")
    preferences: Dict[str, Any] = Field(..., description="User preferences")
    source: str = Field(default="manual", description="Source of preference update")

class ValidationRequest(BaseModel):
    """Request model for algorithm validation."""
    user_id: str = Field(..., description="User identifier")
    test_data: List[Dict[str, Any]] = Field(..., description="Test interaction data")
    validation_type: str = Field(default="accuracy", description="Type of validation")

@router.post("/personalization/learn")
async def learn_from_interaction(request: UserInteractionRequest):
    """
    Learn from user interaction to improve personalization.
    """
    logger.info(f"Learning from interaction for user: {request.user_id}")
    
    try:
        # Learn from interaction
        interaction_data = {
            "message": request.message,
            "response": request.response,
            "interaction_type": request.interaction_type,
            "context": request.context or {}
        }
        
        personalization_agent = get_personalization_agent()
        result = await personalization_agent.learn_from_interaction(
            user_id=request.user_id,
            interaction_data=interaction_data
        )
        
        if result.get("success", False):
            return {
                "success": True,
                "learning_outcome": result.get("learning_outcome", ""),
                "preferences_updated": result.get("preferences_updated", False),
                "confidence_score": result.get("confidence_score", 0.0),
                "user_id": request.user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Learning failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error learning from interaction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error learning from interaction: {str(e)}"
        )

@router.post("/personalization/update-preferences")
async def update_user_preferences(request: PreferenceUpdateRequest):
    """
    Manually update user preferences.
    """
    logger.info(f"Updating preferences for user: {request.user_id}")
    
    try:
        personalization_agent = get_personalization_agent()
        result = await personalization_agent.update_preferences(
            user_id=request.user_id,
            preferences=request.preferences,
            source=request.source
        )
        
        if result.get("success", False):
            return {
                "success": True,
                "preferences_updated": result.get("preferences_updated", {}),
                "total_preferences": result.get("total_preferences", 0),
                "user_id": request.user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Preference update failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error updating preferences: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error updating preferences: {str(e)}"
        )

@router.get("/personalization/preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """
    Get current user preferences and learning insights.
    """
    logger.info(f"Getting preferences for user: {user_id}")
    
    try:
        # Use the correct method or return a clear error
        personalization_agent = get_personalization_agent()
        if hasattr(personalization_agent, 'get_user_preferences'):
            result = await personalization_agent.get_user_preferences(user_id=user_id)
        elif hasattr(personalization_agent, 'user_preference'):
            result = await personalization_agent.user_preference.invoke({"user_id": user_id, "action": "get"})
        else:
            return {"error": "PersonalizationAgent does not support getting user preferences."}
        
        if result.get("success", False):
            return {
                "success": True,
                "preferences": result.get("preferences", {}),
                "learning_insights": result.get("learning_insights", {}),
                "confidence_scores": result.get("confidence_scores", {}),
                "last_updated": result.get("last_updated"),
                "user_id": user_id
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"User preferences not found: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting user preferences: {str(e)}"
        )

@router.post("/personalization/personalize")
async def personalize_response(
    user_id: str,
    message: str,
    context: Optional[Dict[str, Any]] = None
):
    """
    Get personalized response based on user preferences.
    """
    logger.info(f"Personalizing response for user: {user_id}")
    
    try:
        personalization_agent = get_personalization_agent()
        result = await personalization_agent.personalize_response(
            user_id=user_id,
            message=message,
            context=context or {}
        )
        
        if result.get("success", False):
            return {
                "success": True,
                "personalized_response": result.get("personalized_response", ""),
                "personalization_factors": result.get("personalization_factors", {}),
                "confidence_score": result.get("confidence_score", 0.0),
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Personalization failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error personalizing response: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error personalizing response: {str(e)}"
        )

@router.post("/personalization/validate")
async def validate_learning_algorithm(request: ValidationRequest):
    """
    Validate personalization learning algorithm with test data.
    """
    logger.info(f"Validating learning algorithm for user: {request.user_id}")
    
    try:
        personalization_agent = get_personalization_agent()
        result = await personalization_agent.validate_algorithm(
            user_id=request.user_id,
            test_data=request.test_data,
            validation_type=request.validation_type
        )
        
        if result.get("success", False):
            return {
                "success": True,
                "validation_results": result.get("validation_results", {}),
                "accuracy_score": result.get("accuracy_score", 0.0),
                "precision_score": result.get("precision_score", 0.0),
                "recall_score": result.get("recall_score", 0.0),
                "f1_score": result.get("f1_score", 0.0),
                "recommendations": result.get("recommendations", []),
                "user_id": request.user_id,
                "validation_type": request.validation_type,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Validation failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error validating algorithm: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error validating algorithm: {str(e)}"
        )

@router.get("/personalization/insights/{user_id}")
async def get_learning_insights(user_id: str):
    """
    Get detailed learning insights and patterns for a user.
    """
    logger.info(f"Getting learning insights for user: {user_id}")
    
    try:
        personalization_agent = get_personalization_agent()
        result = await personalization_agent.get_learning_insights(user_id=user_id)
        
        if result.get("success", False):
            return {
                "success": True,
                "topic_preferences": result.get("topic_preferences", {}),
                "interaction_patterns": result.get("interaction_patterns", {}),
                "time_patterns": result.get("time_patterns", {}),
                "style_preferences": result.get("style_preferences", {}),
                "learning_progress": result.get("learning_progress", {}),
                "user_id": user_id,
                "last_analyzed": result.get("last_analyzed")
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Learning insights not found: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error getting learning insights: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting learning insights: {str(e)}"
        )

@router.post("/personalization/reset/{user_id}")
async def reset_user_preferences(user_id: str):
    """
    Reset user preferences and learning data.
    """
    logger.info(f"Resetting preferences for user: {user_id}")
    
    try:
        personalization_agent = get_personalization_agent()
        result = await personalization_agent.reset_user_profile(user_id=user_id)
        
        if result.get("success", False):
            return {
                "success": True,
                "message": "User preferences and learning data reset successfully",
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Reset failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Error resetting preferences: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting preferences: {str(e)}"
        )

@router.get("/personalization/health")
async def personalization_health_check():
    """Health check for personalization services."""
    try:
        # Test basic functionality
        user_id = "health_check_user"
        personalization_agent = get_personalization_agent()
        await personalization_agent.update_preferences(
            user_id=user_id,
            preferences={"topic": "testing"},
            source="manual"
        )
        
        return {
            "status": "healthy",
            "service": "personalization-processing",
            "timestamp": datetime.utcnow().isoformat(),
            "test_result": True,
            "learning_algorithm": "preference-based",
            "supported_features": [
                "topic_preferences",
                "interaction_patterns",
                "time_patterns",
                "style_preferences"
            ]
        }
    except Exception as e:
        logger.error(f"Personalization health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "personalization-processing",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        } 