#!/usr/bin/env python3
"""
Personalization Learning Algorithm Validation Script

This script validates the personalization learning algorithms by testing
various scenarios and measuring accuracy, precision, recall, and F1 scores.
"""

import asyncio
import json
import time
import random
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PersonalizationValidator:
    """Personalization learning algorithm validator."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.test_results = {}
        self.validation_data = self.generate_test_data()
    
    def generate_test_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate comprehensive test data for validation."""
        
        # User interaction patterns
        interaction_patterns = [
            {
                "user_id": "user_1",
                "message": "Tell me about artificial intelligence",
                "response": "AI is a branch of computer science...",
                "interaction_type": "chat",
                "expected_preferences": ["technology", "ai", "learning"]
            },
            {
                "user_id": "user_1",
                "message": "What's the weather like?",
                "response": "I don't have access to real-time weather data...",
                "interaction_type": "chat",
                "expected_preferences": ["weather", "information"]
            },
            {
                "user_id": "user_2",
                "message": "Help me with my math homework",
                "response": "I'd be happy to help with your math homework...",
                "interaction_type": "chat",
                "expected_preferences": ["education", "mathematics", "homework"]
            },
            {
                "user_id": "user_2",
                "message": "What's the best restaurant in town?",
                "response": "I can't provide real-time restaurant recommendations...",
                "interaction_type": "chat",
                "expected_preferences": ["food", "restaurants", "recommendations"]
            },
            {
                "user_id": "user_3",
                "message": "Explain quantum physics",
                "response": "Quantum physics is a fundamental theory...",
                "interaction_type": "chat",
                "expected_preferences": ["science", "physics", "quantum"]
            },
            {
                "user_id": "user_3",
                "message": "How do I cook pasta?",
                "response": "Here's a basic guide to cooking pasta...",
                "interaction_type": "chat",
                "expected_preferences": ["cooking", "food", "pasta"]
            }
        ]
        
        # Time-based patterns
        time_patterns = [
            {
                "user_id": "user_1",
                "timestamp": "2024-01-15T09:00:00Z",
                "interaction_type": "morning",
                "expected_pattern": "morning_user"
            },
            {
                "user_id": "user_2",
                "timestamp": "2024-01-15T14:00:00Z",
                "interaction_type": "afternoon",
                "expected_pattern": "afternoon_user"
            },
            {
                "user_id": "user_3",
                "timestamp": "2024-01-15T22:00:00Z",
                "interaction_type": "evening",
                "expected_pattern": "evening_user"
            }
        ]
        
        # Style preferences
        style_patterns = [
            {
                "user_id": "user_1",
                "message": "Give me a detailed explanation",
                "response": "Here's a comprehensive breakdown...",
                "style": "detailed",
                "expected_style": "detailed"
            },
            {
                "user_id": "user_2",
                "message": "Keep it simple",
                "response": "Here's the simple answer...",
                "style": "concise",
                "expected_style": "concise"
            },
            {
                "user_id": "user_3",
                "message": "Use examples",
                "response": "Let me explain with examples...",
                "style": "examples",
                "expected_style": "examples"
            }
        ]
        
        return {
            "interaction_patterns": interaction_patterns,
            "time_patterns": time_patterns,
            "style_patterns": style_patterns
        }
    
    async def test_learning_algorithm(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test the learning algorithm with provided data."""
        logger.info(f"Testing learning algorithm with {len(test_data)} interactions")
        
        results = {
            "total_interactions": len(test_data),
            "successful_learnings": 0,
            "failed_learnings": 0,
            "learning_times": [],
            "preferences_updated": 0,
            "errors": []
        }
        
        for interaction in test_data:
            try:
                start_time = time.time()
                
                # Send learning request
                response = requests.post(
                    f"{self.api_base_url}/api/v1/personalization/learn",
                    json=interaction,
                    timeout=10
                )
                
                learning_time = time.time() - start_time
                results["learning_times"].append(learning_time)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        results["successful_learnings"] += 1
                        if data.get("preferences_updated"):
                            results["preferences_updated"] += 1
                    else:
                        results["failed_learnings"] += 1
                        results["errors"].append(f"Learning failed: {data.get('error')}")
                else:
                    results["failed_learnings"] += 1
                    results["errors"].append(f"HTTP {response.status_code}: {response.text}")
                
            except Exception as e:
                results["failed_learnings"] += 1
                results["errors"].append(f"Exception: {str(e)}")
        
        # Calculate average learning time
        if results["learning_times"]:
            results["avg_learning_time"] = sum(results["learning_times"]) / len(results["learning_times"])
        else:
            results["avg_learning_time"] = 0
        
        return results
    
    async def test_preference_retrieval(self, user_ids: List[str]) -> Dict[str, Any]:
        """Test preference retrieval accuracy."""
        logger.info(f"Testing preference retrieval for {len(user_ids)} users")
        
        results = {
            "total_users": len(user_ids),
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "preferences_found": 0,
            "retrieval_times": [],
            "errors": []
        }
        
        for user_id in user_ids:
            try:
                start_time = time.time()
                
                response = requests.get(
                    f"{self.api_base_url}/api/v1/personalization/preferences/{user_id}",
                    timeout=10
                )
                
                retrieval_time = time.time() - start_time
                results["retrieval_times"].append(retrieval_time)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        results["successful_retrievals"] += 1
                        preferences = data.get("preferences", {})
                        if preferences:
                            results["preferences_found"] += 1
                    else:
                        results["failed_retrievals"] += 1
                        results["errors"].append(f"Retrieval failed: {data.get('error')}")
                else:
                    results["failed_retrievals"] += 1
                    results["errors"].append(f"HTTP {response.status_code}: {response.text}")
                
            except Exception as e:
                results["failed_retrievals"] += 1
                results["errors"].append(f"Exception: {str(e)}")
        
        # Calculate average retrieval time
        if results["retrieval_times"]:
            results["avg_retrieval_time"] = sum(results["retrieval_times"]) / len(results["retrieval_times"])
        else:
            results["avg_retrieval_time"] = 0
        
        return results
    
    async def test_personalization_accuracy(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test personalization accuracy with known test cases."""
        logger.info(f"Testing personalization accuracy with {len(test_cases)} test cases")
        
        results = {
            "total_cases": len(test_cases),
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "prediction_times": [],
            "errors": []
        }
        
        y_true = []
        y_pred = []
        
        for test_case in test_cases:
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.api_base_url}/api/v1/personalization/personalize",
                    params={
                        "user_id": test_case["user_id"],
                        "message": test_case["message"]
                    },
                    timeout=10
                )
                
                prediction_time = time.time() - start_time
                results["prediction_times"].append(prediction_time)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        # Compare predicted preferences with expected
                        predicted_factors = data.get("personalization_factors", {})
                        expected_preferences = test_case.get("expected_preferences", [])
                        
                        # Simple accuracy check (can be enhanced)
                        if any(pref in str(predicted_factors) for pref in expected_preferences):
                            results["correct_predictions"] += 1
                            y_true.append(1)
                            y_pred.append(1)
                        else:
                            results["incorrect_predictions"] += 1
                            y_true.append(1)
                            y_pred.append(0)
                    else:
                        results["errors"].append(f"Personalization failed: {data.get('error')}")
                else:
                    results["errors"].append(f"HTTP {response.status_code}: {response.text}")
                
            except Exception as e:
                results["errors"].append(f"Exception: {str(e)}")
        
        # Calculate metrics
        if y_true and y_pred:
            results["accuracy"] = accuracy_score(y_true, y_pred)
            results["precision"] = precision_score(y_true, y_pred, zero_division=0)
            results["recall"] = recall_score(y_true, y_pred, zero_division=0)
            results["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate average prediction time
        if results["prediction_times"]:
            results["avg_prediction_time"] = sum(results["prediction_times"]) / len(results["prediction_times"])
        else:
            results["avg_prediction_time"] = 0
        
        return results
    
    async def test_algorithm_validation(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test the algorithm validation endpoint."""
        logger.info("Testing algorithm validation endpoint")
        
        results = {
            "validation_tests": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "validation_times": [],
            "errors": []
        }
        
        # Test different validation types
        validation_types = ["accuracy", "precision", "recall", "f1"]
        
        for val_type in validation_types:
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.api_base_url}/api/v1/personalization/validate",
                    json={
                        "user_id": "test_user",
                        "test_data": validation_data["interaction_patterns"][:3],
                        "validation_type": val_type
                    },
                    timeout=30
                )
                
                validation_time = time.time() - start_time
                results["validation_times"].append(validation_time)
                results["validation_tests"] += 1
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        results["successful_validations"] += 1
                    else:
                        results["failed_validations"] += 1
                        results["errors"].append(f"Validation failed: {data.get('error')}")
                else:
                    results["failed_validations"] += 1
                    results["errors"].append(f"HTTP {response.status_code}: {response.text}")
                
            except Exception as e:
                results["failed_validations"] += 1
                results["errors"].append(f"Exception: {str(e)}")
        
        # Calculate average validation time
        if results["validation_times"]:
            results["avg_validation_time"] = sum(results["validation_times"]) / len(results["validation_times"])
        else:
            results["avg_validation_time"] = 0
        
        return results
    
    async def test_learning_insights(self, user_ids: List[str]) -> Dict[str, Any]:
        """Test learning insights retrieval."""
        logger.info(f"Testing learning insights for {len(user_ids)} users")
        
        results = {
            "total_users": len(user_ids),
            "successful_insights": 0,
            "failed_insights": 0,
            "insights_retrieved": 0,
            "retrieval_times": [],
            "errors": []
        }
        
        for user_id in user_ids:
            try:
                start_time = time.time()
                
                response = requests.get(
                    f"{self.api_base_url}/api/v1/personalization/insights/{user_id}",
                    timeout=10
                )
                
                retrieval_time = time.time() - start_time
                results["retrieval_times"].append(retrieval_time)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        results["successful_insights"] += 1
                        insights = data.get("topic_preferences", {})
                        if insights:
                            results["insights_retrieved"] += 1
                    else:
                        results["failed_insights"] += 1
                        results["errors"].append(f"Insights failed: {data.get('error')}")
                else:
                    results["failed_insights"] += 1
                    results["errors"].append(f"HTTP {response.status_code}: {response.text}")
                
            except Exception as e:
                results["failed_insights"] += 1
                results["errors"].append(f"Exception: {str(e)}")
        
        # Calculate average retrieval time
        if results["retrieval_times"]:
            results["avg_retrieval_time"] = sum(results["retrieval_times"]) / len(results["retrieval_times"])
        else:
            results["avg_retrieval_time"] = 0
        
        return results
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of personalization algorithms."""
        logger.info("Starting comprehensive personalization validation...")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {},
            "recommendations": []
        }
        
        # Test 1: Learning Algorithm
        logger.info("Test 1: Learning Algorithm")
        learning_results = await self.test_learning_algorithm(
            self.validation_data["interaction_patterns"]
        )
        validation_results["tests"]["learning_algorithm"] = learning_results
        
        # Test 2: Preference Retrieval
        logger.info("Test 2: Preference Retrieval")
        user_ids = ["user_1", "user_2", "user_3"]
        retrieval_results = await self.test_preference_retrieval(user_ids)
        validation_results["tests"]["preference_retrieval"] = retrieval_results
        
        # Test 3: Personalization Accuracy
        logger.info("Test 3: Personalization Accuracy")
        accuracy_results = await self.test_personalization_accuracy(
            self.validation_data["interaction_patterns"]
        )
        validation_results["tests"]["personalization_accuracy"] = accuracy_results
        
        # Test 4: Algorithm Validation
        logger.info("Test 4: Algorithm Validation")
        validation_test_results = await self.test_algorithm_validation(self.validation_data)
        validation_results["tests"]["algorithm_validation"] = validation_test_results
        
        # Test 5: Learning Insights
        logger.info("Test 5: Learning Insights")
        insights_results = await self.test_learning_insights(user_ids)
        validation_results["tests"]["learning_insights"] = insights_results
        
        # Generate summary
        validation_results["summary"] = self.generate_summary(validation_results["tests"])
        
        # Generate recommendations
        validation_results["recommendations"] = self.generate_recommendations(validation_results["tests"])
        
        return validation_results
    
    def generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        summary = {
            "total_tests": len(test_results),
            "overall_success_rate": 0.0,
            "performance_metrics": {},
            "error_count": 0
        }
        
        total_errors = 0
        success_rates = []
        
        for test_name, results in test_results.items():
            if "errors" in results:
                total_errors += len(results["errors"])
            
            # Calculate success rate for each test
            if "total_interactions" in results:
                success_rate = results["successful_learnings"] / results["total_interactions"]
                success_rates.append(success_rate)
            elif "total_users" in results:
                success_rate = results["successful_retrievals"] / results["total_users"]
                success_rates.append(success_rate)
            elif "total_cases" in results:
                success_rate = results["correct_predictions"] / results["total_cases"]
                success_rates.append(success_rate)
            elif "validation_tests" in results:
                success_rate = results["successful_validations"] / results["validation_tests"]
                success_rates.append(success_rate)
        
        summary["error_count"] = total_errors
        if success_rates:
            summary["overall_success_rate"] = sum(success_rates) / len(success_rates)
        
        # Performance metrics
        performance_metrics = {}
        for test_name, results in test_results.items():
            if "avg_learning_time" in results:
                performance_metrics[f"{test_name}_avg_time"] = results["avg_learning_time"]
            elif "avg_retrieval_time" in results:
                performance_metrics[f"{test_name}_avg_time"] = results["avg_retrieval_time"]
            elif "avg_prediction_time" in results:
                performance_metrics[f"{test_name}_avg_time"] = results["avg_prediction_time"]
            elif "avg_validation_time" in results:
                performance_metrics[f"{test_name}_avg_time"] = results["avg_validation_time"]
        
        summary["performance_metrics"] = performance_metrics
        
        return summary
    
    def generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check learning algorithm performance
        if "learning_algorithm" in test_results:
            learning_results = test_results["learning_algorithm"]
            success_rate = learning_results["successful_learnings"] / learning_results["total_interactions"]
            
            if success_rate < 0.8:
                recommendations.append("Learning algorithm success rate is below 80%. Consider improving error handling.")
            
            if learning_results["avg_learning_time"] > 1.0:
                recommendations.append("Learning algorithm is slow. Consider optimizing performance.")
        
        # Check personalization accuracy
        if "personalization_accuracy" in test_results:
            accuracy_results = test_results["personalization_accuracy"]
            
            if accuracy_results["accuracy"] < 0.7:
                recommendations.append("Personalization accuracy is below 70%. Consider improving the algorithm.")
            
            if accuracy_results["f1_score"] < 0.6:
                recommendations.append("F1 score is low. Consider balancing precision and recall.")
        
        # Check error rates
        total_errors = sum(len(results.get("errors", [])) for results in test_results.values())
        if total_errors > 10:
            recommendations.append("High error rate detected. Review error handling and API stability.")
        
        # Performance recommendations
        for test_name, results in test_results.items():
            if "avg_learning_time" in results and results["avg_learning_time"] > 2.0:
                recommendations.append(f"{test_name} is too slow. Consider caching or optimization.")
        
        if not recommendations:
            recommendations.append("All tests passed successfully. Personalization algorithm is working well.")
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save validation results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"personalization_validation_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Validation results saved to: {filename}")
        return filename

async def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Personalization Algorithm Validation")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--test-only", choices=["learning", "retrieval", "accuracy", "validation", "insights"], 
                       help="Run only specific test")
    
    args = parser.parse_args()
    
    validator = PersonalizationValidator(args.api_url)
    
    if args.test_only:
        # Run specific test
        if args.test_only == "learning":
            results = await validator.test_learning_algorithm(validator.validation_data["interaction_patterns"])
        elif args.test_only == "retrieval":
            results = await validator.test_preference_retrieval(["user_1", "user_2", "user_3"])
        elif args.test_only == "accuracy":
            results = await validator.test_personalization_accuracy(validator.validation_data["interaction_patterns"])
        elif args.test_only == "validation":
            results = await validator.test_algorithm_validation(validator.validation_data)
        elif args.test_only == "insights":
            results = await validator.test_learning_insights(["user_1", "user_2", "user_3"])
        
        print(json.dumps(results, indent=2))
    else:
        # Run comprehensive validation
        results = await validator.run_comprehensive_validation()
        
        # Save results
        filename = validator.save_results(results, args.output)
        
        # Print summary
        print("\n" + "="*50)
        print("PERSONALIZATION VALIDATION SUMMARY")
        print("="*50)
        print(f"Overall Success Rate: {results['summary']['overall_success_rate']:.2%}")
        print(f"Total Errors: {results['summary']['error_count']}")
        print(f"Results saved to: {filename}")
        
        print("\nRECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"- {rec}")

if __name__ == "__main__":
    asyncio.run(main()) 