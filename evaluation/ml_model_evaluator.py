"""
ML Model Evaluation Framework

Comprehensive evaluation system for machine learning models including LLM performance,
embedding drift detection, model degradation monitoring, and automated retraining triggers.
"""

import asyncio
import json
import logging
import pickle
import statistics
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import plotly.graph_objs as go
import plotly.express as px

from .golden_dataset import GoldenDatasetManager, GoldenMatch
from ..llm_client.ollama_client import OllamaClient
from ..vector_db.embedding_service import EmbeddingService
from ..agents.matcher_agent import MatcherAgent


@dataclass
class ModelMetric:
    """Individual model performance metric."""
    
    metric_name: str
    current_value: float
    baseline_value: Optional[float] = None
    threshold_warning: float = 0.1  # 10% degradation
    threshold_critical: float = 0.2  # 20% degradation
    higher_is_better: bool = True
    unit: str = "score"
    
    # Drift detection
    drift_detected: bool = False
    drift_severity: str = "none"  # none, mild, moderate, severe
    drift_magnitude: float = 0.0
    
    # Historical tracking
    historical_values: List[Tuple[datetime, float]] = field(default_factory=list)
    
    def add_measurement(self, value: float):
        """Add new measurement and detect drift."""
        self.historical_values.append((datetime.now(), value))
        
        if self.baseline_value is not None:
            # Calculate drift magnitude
            if self.baseline_value != 0:
                if self.higher_is_better:
                    self.drift_magnitude = (self.baseline_value - value) / self.baseline_value
                else:
                    self.drift_magnitude = (value - self.baseline_value) / self.baseline_value
            
            # Determine drift severity
            if abs(self.drift_magnitude) >= self.threshold_critical:
                self.drift_severity = "severe"
                self.drift_detected = True
            elif abs(self.drift_magnitude) >= self.threshold_warning:
                self.drift_severity = "moderate"
                self.drift_detected = True
            elif abs(self.drift_magnitude) >= 0.05:  # 5% threshold for mild drift
                self.drift_severity = "mild"
                self.drift_detected = True
            else:
                self.drift_severity = "none"
                self.drift_detected = False
        
        self.current_value = value
        
        # Keep only last 50 measurements
        if len(self.historical_values) > 50:
            self.historical_values = self.historical_values[-50:]
    
    def get_status(self) -> str:
        """Get current metric status."""
        if self.drift_severity == "severe":
            return "critical"
        elif self.drift_severity == "moderate":
            return "warning"
        elif self.drift_severity == "mild":
            return "caution"
        else:
            return "healthy"


@dataclass
class ModelEvaluationResult:
    """Complete model evaluation result."""
    
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    model_type: str = ""  # llm, embedding, classification
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Performance metrics
    metrics: Dict[str, ModelMetric] = field(default_factory=dict)
    overall_score: float = 0.0
    
    # Drift analysis
    drift_detected: bool = False
    drift_components: List[str] = field(default_factory=list)
    
    # Quality indicators
    response_quality_score: float = 0.0
    consistency_score: float = 0.0
    latency_score: float = 0.0
    
    # Recommendations
    retraining_recommended: bool = False
    retraining_urgency: str = "none"  # none, low, medium, high, critical
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution details
    execution_time: float = 0.0
    test_samples: int = 0
    errors_encountered: int = 0
    
    def add_recommendation(self, priority: str, category: str, title: str, description: str, actions: List[str]):
        """Add improvement recommendation."""
        self.recommendations.append({
            'priority': priority,
            'category': category,
            'title': title,
            'description': description,
            'actions': actions,
            'timestamp': datetime.now().isoformat()
        })
    
    def calculate_overall_score(self):
        """Calculate overall model health score."""
        if not self.metrics:
            self.overall_score = 0.0
            return
        
        # Weight different metric types
        weights = {
            'accuracy': 0.3,
            'response_quality': 0.25,
            'consistency': 0.2,
            'latency': 0.15,
            'stability': 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, metric in self.metrics.items():
            # Determine weight category
            weight_key = 'accuracy'  # default
            if 'quality' in metric_name.lower():
                weight_key = 'response_quality'
            elif 'consistency' in metric_name.lower():
                weight_key = 'consistency'
            elif 'latency' in metric_name.lower() or 'response_time' in metric_name.lower():
                weight_key = 'latency'
            elif 'stability' in metric_name.lower() or 'drift' in metric_name.lower():
                weight_key = 'stability'
            
            weight = weights.get(weight_key, 0.1)
            
            # Normalize metric value to 0-100 scale
            normalized_value = min(100, max(0, metric.current_value * 100))
            
            # Apply drift penalty
            if metric.drift_detected:
                penalty = 0.8 if metric.drift_severity == "severe" else 0.9 if metric.drift_severity == "moderate" else 0.95
                normalized_value *= penalty
            
            weighted_sum += normalized_value * weight
            total_weight += weight
        
        self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0


class LLMEvaluator:
    """Evaluator for Large Language Models (Ollama)."""
    
    def __init__(self):
        self.llm_client = OllamaClient()
        self.golden_dataset = GoldenDatasetManager()
        self.logger = logging.getLogger(__name__)
        
    async def evaluate_llm_performance(self, model_name: str = "qwen2.5:1.5b") -> ModelEvaluationResult:
        """Evaluate LLM performance on grocery-specific tasks."""
        self.logger.info(f"Evaluating LLM model: {model_name}")
        
        start_time = time.time()
        result = ModelEvaluationResult(
            model_name=model_name,
            model_type="llm"
        )
        
        try:
            # Test different task categories
            await self._test_ingredient_normalization(result, model_name)
            await self._test_product_classification(result, model_name)
            await self._test_response_consistency(result, model_name)
            await self._test_response_quality(result, model_name)
            await self._test_response_latency(result, model_name)
            
            result.execution_time = time.time() - start_time
            result.calculate_overall_score()
            
            # Generate recommendations
            self._generate_llm_recommendations(result)
            
        except Exception as e:
            self.logger.error(f"LLM evaluation failed: {e}")
            result.errors_encountered += 1
            result.add_recommendation(
                'high', 'system', 'LLM Evaluation Failed',
                f'Could not complete LLM evaluation: {str(e)}',
                ['Check LLM service connectivity', 'Verify model availability', 'Review error logs']
            )
        
        return result
    
    async def _test_ingredient_normalization(self, result: ModelEvaluationResult, model_name: str):
        """Test ingredient normalization accuracy."""
        test_ingredients = [
            ("milk", "milk"),
            ("organic skim milk", "milk"),
            ("2% milk", "milk"),
            ("chicken breast", "chicken breast"),
            ("boneless skinless chicken breast", "chicken breast"),
            ("ground beef 80/20", "ground beef"),
            ("lean ground beef", "ground beef")
        ]
        
        correct_predictions = 0
        response_times = []
        
        for input_ingredient, expected_normalized in test_ingredients:
            start_time = time.time()
            
            try:
                prompt = f"Normalize this ingredient to its base form: '{input_ingredient}'. Return only the normalized ingredient name."
                response = await self.llm_client.generate_response(prompt, model_name)
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                # Simple matching - in production, use more sophisticated comparison
                if expected_normalized.lower() in response.lower():
                    correct_predictions += 1
                    
            except Exception as e:
                self.logger.warning(f"Ingredient normalization test failed for '{input_ingredient}': {e}")
                response_times.append(5.0)  # Penalty time
                result.errors_encountered += 1
        
        # Calculate metrics
        accuracy = correct_predictions / len(test_ingredients) if test_ingredients else 0.0
        avg_response_time = statistics.mean(response_times) if response_times else 5.0
        
        # Add metrics
        result.metrics['ingredient_normalization_accuracy'] = ModelMetric(
            metric_name="Ingredient Normalization Accuracy",
            current_value=accuracy,
            baseline_value=0.85,  # Expected baseline
            higher_is_better=True,
            unit="accuracy"
        )
        
        result.metrics['ingredient_normalization_latency'] = ModelMetric(
            metric_name="Ingredient Normalization Latency",
            current_value=avg_response_time,
            baseline_value=0.5,  # Expected 500ms
            threshold_warning=1.0,  # 1s warning
            threshold_critical=2.0,  # 2s critical
            higher_is_better=False,
            unit="seconds"
        )
        
        result.test_samples += len(test_ingredients)
    
    async def _test_product_classification(self, result: ModelEvaluationResult, model_name: str):
        """Test product classification accuracy."""
        test_products = [
            ("Lactantia 2% Milk 4L", "dairy"),
            ("Organic Ground Beef 1lb", "meat"),
            ("Fresh Bananas", "produce"),
            ("Whole Wheat Bread", "bakery"),
            ("Extra Virgin Olive Oil", "pantry")
        ]
        
        correct_classifications = 0
        response_times = []
        
        for product, expected_category in test_products:
            start_time = time.time()
            
            try:
                prompt = f"Classify this grocery product into a category: '{product}'. Choose from: dairy, meat, produce, bakery, pantry, frozen. Return only the category name."
                response = await self.llm_client.generate_response(prompt, model_name)
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if expected_category.lower() in response.lower():
                    correct_classifications += 1
                    
            except Exception as e:
                self.logger.warning(f"Product classification test failed for '{product}': {e}")
                response_times.append(5.0)
                result.errors_encountered += 1
        
        # Calculate metrics
        accuracy = correct_classifications / len(test_products) if test_products else 0.0
        avg_response_time = statistics.mean(response_times) if response_times else 5.0
        
        result.metrics['product_classification_accuracy'] = ModelMetric(
            metric_name="Product Classification Accuracy",
            current_value=accuracy,
            baseline_value=0.90,  # Expected baseline
            higher_is_better=True,
            unit="accuracy"
        )
        
        result.metrics['product_classification_latency'] = ModelMetric(
            metric_name="Product Classification Latency",
            current_value=avg_response_time,
            baseline_value=0.6,
            higher_is_better=False,
            unit="seconds"
        )
        
        result.test_samples += len(test_products)
    
    async def _test_response_consistency(self, result: ModelEvaluationResult, model_name: str):
        """Test response consistency for identical queries."""
        test_query = "What is the best way to store fresh herbs?"
        responses = []
        
        # Run same query multiple times
        for _ in range(5):
            try:
                response = await self.llm_client.generate_response(test_query, model_name)
                responses.append(response.lower().strip())
            except Exception as e:
                self.logger.warning(f"Consistency test failed: {e}")
                result.errors_encountered += 1
        
        # Calculate consistency (simple word overlap method)
        if len(responses) >= 2:
            consistency_scores = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    words_i = set(responses[i].split())
                    words_j = set(responses[j].split())
                    
                    if words_i and words_j:
                        overlap = len(words_i & words_j)
                        total = len(words_i | words_j)
                        consistency = overlap / total if total > 0 else 0.0
                        consistency_scores.append(consistency)
            
            avg_consistency = statistics.mean(consistency_scores) if consistency_scores else 0.0
        else:
            avg_consistency = 0.0
        
        result.metrics['response_consistency'] = ModelMetric(
            metric_name="Response Consistency",
            current_value=avg_consistency,
            baseline_value=0.75,  # Expected baseline
            higher_is_better=True,
            unit="consistency_score"
        )
        
        result.consistency_score = avg_consistency
        result.test_samples += len(responses)
    
    async def _test_response_quality(self, result: ModelEvaluationResult, model_name: str):
        """Test response quality using heuristics."""
        test_queries = [
            "How do I choose ripe avocados?",
            "What's the difference between organic and conventional produce?",
            "How long can I store ground beef in the refrigerator?"
        ]
        
        quality_scores = []
        
        for query in test_queries:
            try:
                response = await self.llm_client.generate_response(query, model_name)
                
                # Simple quality heuristics
                quality_score = 0.0
                
                # Length check (not too short, not too long)
                if 50 <= len(response) <= 500:
                    quality_score += 0.3
                
                # Contains specific food-related terms
                food_terms = ['food', 'fresh', 'store', 'refrigerat', 'organic', 'ripe']
                if any(term in response.lower() for term in food_terms):
                    quality_score += 0.3
                
                # Sentence structure (contains periods)
                if '.' in response:
                    quality_score += 0.2
                
                # No obvious errors or gibberish
                if not any(char * 3 in response for char in 'abcdefghijklmnopqrstuvwxyz'):
                    quality_score += 0.2
                
                quality_scores.append(quality_score)
                
            except Exception as e:
                self.logger.warning(f"Quality test failed for query '{query}': {e}")
                quality_scores.append(0.0)
                result.errors_encountered += 1
        
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        
        result.metrics['response_quality'] = ModelMetric(
            metric_name="Response Quality",
            current_value=avg_quality,
            baseline_value=0.80,  # Expected baseline
            higher_is_better=True,
            unit="quality_score"
        )
        
        result.response_quality_score = avg_quality
        result.test_samples += len(test_queries)
    
    async def _test_response_latency(self, result: ModelEvaluationResult, model_name: str):
        """Test response latency performance."""
        test_queries = [
            "Normalize: organic milk",
            "Category: fresh strawberries",
            "Store: ground turkey"
        ]
        
        response_times = []
        
        for query in test_queries:
            start_time = time.time()
            
            try:
                await self.llm_client.generate_response(query, model_name)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
            except Exception as e:
                self.logger.warning(f"Latency test failed for query '{query}': {e}")
                response_times.append(10.0)  # Penalty time
                result.errors_encountered += 1
        
        # Calculate latency metrics
        if response_times:
            avg_latency = statistics.mean(response_times)
            p95_latency = np.percentile(response_times, 95)
        else:
            avg_latency = 10.0
            p95_latency = 10.0
        
        result.metrics['avg_response_latency'] = ModelMetric(
            metric_name="Average Response Latency",
            current_value=avg_latency,
            baseline_value=1.0,  # Expected 1s
            threshold_warning=2.0,
            threshold_critical=5.0,
            higher_is_better=False,
            unit="seconds"
        )
        
        result.metrics['p95_response_latency'] = ModelMetric(
            metric_name="P95 Response Latency",
            current_value=p95_latency,
            baseline_value=2.0,  # Expected 2s
            threshold_warning=4.0,
            threshold_critical=8.0,
            higher_is_better=False,
            unit="seconds"
        )
        
        result.latency_score = max(0.0, 1.0 - (avg_latency - 1.0) / 4.0)  # Score based on target of 1s
        result.test_samples += len(test_queries)
    
    def _generate_llm_recommendations(self, result: ModelEvaluationResult):
        """Generate LLM-specific recommendations."""
        # Check for performance issues
        if result.overall_score < 70:
            result.retraining_recommended = True
            result.retraining_urgency = "high" if result.overall_score < 50 else "medium"
            
            result.add_recommendation(
                'high', 'model', 'LLM Performance Below Threshold',
                f'Overall model score ({result.overall_score:.1f}) is below acceptable threshold',
                [
                    'Consider switching to a larger model variant',
                    'Review and update prompt templates',
                    'Fine-tune model on grocery-specific data',
                    'Implement response caching for common queries'
                ]
            )
        
        # Check latency issues
        avg_latency = result.metrics.get('avg_response_latency')
        if avg_latency and avg_latency.current_value > 3.0:
            result.add_recommendation(
                'medium', 'performance', 'High Response Latency',
                f'Average response time ({avg_latency.current_value:.2f}s) exceeds target',
                [
                    'Optimize prompt length and complexity',
                    'Implement response caching',
                    'Consider using a faster model variant',
                    'Add request batching for multiple queries'
                ]
            )
        
        # Check consistency issues
        consistency = result.metrics.get('response_consistency')
        if consistency and consistency.current_value < 0.6:
            result.add_recommendation(
                'medium', 'quality', 'Low Response Consistency',
                f'Response consistency ({consistency.current_value:.2f}) is below target',
                [
                    'Add temperature control to model configuration',
                    'Implement response validation and retry logic',
                    'Use more specific prompts with examples',
                    'Consider deterministic sampling methods'
                ]
            )


class EmbeddingEvaluator:
    """Evaluator for embedding models and vector similarity."""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.golden_dataset = GoldenDatasetManager()
        self.logger = logging.getLogger(__name__)
        
        # Store historical embeddings for drift detection
        self.embedding_history: List[Tuple[datetime, np.ndarray]] = []
    
    async def evaluate_embedding_performance(self) -> ModelEvaluationResult:
        """Evaluate embedding model performance."""
        self.logger.info("Evaluating embedding model performance")
        
        start_time = time.time()
        result = ModelEvaluationResult(
            model_name="all-MiniLM-L6-v2",
            model_type="embedding"
        )
        
        try:
            # Test embedding quality
            await self._test_semantic_similarity(result)
            await self._test_embedding_consistency(result)
            await self._test_embedding_drift(result)
            await self._test_embedding_latency(result)
            
            result.execution_time = time.time() - start_time
            result.calculate_overall_score()
            
            # Generate recommendations
            self._generate_embedding_recommendations(result)
            
        except Exception as e:
            self.logger.error(f"Embedding evaluation failed: {e}")
            result.errors_encountered += 1
            result.add_recommendation(
                'high', 'system', 'Embedding Evaluation Failed',
                f'Could not complete embedding evaluation: {str(e)}',
                ['Check embedding service availability', 'Verify model loading', 'Review error logs']
            )
        
        return result
    
    async def _test_semantic_similarity(self, result: ModelEvaluationResult):
        """Test semantic similarity accuracy."""
        # Test pairs: (text1, text2, expected_similarity_high)
        similarity_tests = [
            ("milk", "dairy milk", True),
            ("chicken breast", "boneless chicken", True),
            ("organic apples", "fresh apples", True),
            ("bread", "whole wheat bread", True),
            ("milk", "chicken breast", False),
            ("apples", "ground beef", False),
            ("pasta", "motor oil", False)
        ]
        
        correct_predictions = 0
        similarity_scores = []
        
        for text1, text2, should_be_similar in similarity_tests:
            try:
                # Generate embeddings
                embedding1 = await self.embedding_service.generate_embedding(text1)
                embedding2 = await self.embedding_service.generate_embedding(text2)
                
                # Calculate similarity
                similarity = cosine_similarity(
                    embedding1.reshape(1, -1),
                    embedding2.reshape(1, -1)
                )[0][0]
                
                similarity_scores.append(similarity)
                
                # Check if prediction matches expectation
                predicted_similar = similarity > 0.7  # Threshold for similarity
                if predicted_similar == should_be_similar:
                    correct_predictions += 1
                    
            except Exception as e:
                self.logger.warning(f"Similarity test failed for '{text1}' vs '{text2}': {e}")
                similarity_scores.append(0.0)
                result.errors_encountered += 1
        
        # Calculate metrics
        accuracy = correct_predictions / len(similarity_tests) if similarity_tests else 0.0
        avg_similarity = statistics.mean(similarity_scores) if similarity_scores else 0.0
        
        result.metrics['semantic_similarity_accuracy'] = ModelMetric(
            metric_name="Semantic Similarity Accuracy",
            current_value=accuracy,
            baseline_value=0.85,
            higher_is_better=True,
            unit="accuracy"
        )
        
        result.test_samples += len(similarity_tests)
    
    async def _test_embedding_consistency(self, result: ModelEvaluationResult):
        """Test embedding consistency for identical inputs."""
        test_text = "organic whole milk"
        embeddings = []
        
        # Generate embeddings multiple times
        for _ in range(5):
            try:
                embedding = await self.embedding_service.generate_embedding(test_text)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.warning(f"Consistency test failed: {e}")
                result.errors_encountered += 1
        
        # Calculate consistency (embeddings should be identical)
        if len(embeddings) >= 2:
            consistency_scores = []
            base_embedding = embeddings[0]
            
            for embedding in embeddings[1:]:
                similarity = cosine_similarity(
                    base_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                consistency_scores.append(similarity)
            
            avg_consistency = statistics.mean(consistency_scores) if consistency_scores else 0.0
        else:
            avg_consistency = 0.0
        
        result.metrics['embedding_consistency'] = ModelMetric(
            metric_name="Embedding Consistency",
            current_value=avg_consistency,
            baseline_value=0.99,  # Should be nearly identical
            higher_is_better=True,
            unit="consistency_score"
        )
        
        result.consistency_score = avg_consistency
        result.test_samples += len(embeddings)
    
    async def _test_embedding_drift(self, result: ModelEvaluationResult):
        """Test for embedding drift over time."""
        # Reference ingredients for drift detection
        reference_ingredients = ["milk", "bread", "chicken", "apple", "rice"]
        
        current_embeddings = []
        for ingredient in reference_ingredients:
            try:
                embedding = await self.embedding_service.generate_embedding(ingredient)
                current_embeddings.append(embedding)
            except Exception as e:
                self.logger.warning(f"Drift test failed for '{ingredient}': {e}")
                result.errors_encountered += 1
        
        if current_embeddings:
            # Calculate centroid of current embeddings
            current_centroid = np.mean(current_embeddings, axis=0)
            
            # Compare with historical embeddings if available
            if self.embedding_history:
                # Get most recent historical centroid
                historical_centroids = [emb for _, emb in self.embedding_history[-10:]]  # Last 10
                if historical_centroids:
                    historical_centroid = np.mean(historical_centroids, axis=0)
                    
                    # Calculate drift magnitude
                    drift_similarity = cosine_similarity(
                        current_centroid.reshape(1, -1),
                        historical_centroid.reshape(1, -1)
                    )[0][0]
                    
                    drift_magnitude = 1.0 - drift_similarity
                    
                    result.metrics['embedding_drift'] = ModelMetric(
                        metric_name="Embedding Drift",
                        current_value=drift_magnitude,
                        baseline_value=0.02,  # Expected 2% drift
                        threshold_warning=0.05,  # 5% drift warning
                        threshold_critical=0.10,  # 10% drift critical
                        higher_is_better=False,
                        unit="drift_magnitude"
                    )
                    
                    if drift_magnitude > 0.05:
                        result.drift_detected = True
                        result.drift_components.append("embedding_space")
            
            # Store current centroid for future drift detection
            self.embedding_history.append((datetime.now(), current_centroid))
            
            # Keep only last 50 entries
            if len(self.embedding_history) > 50:
                self.embedding_history = self.embedding_history[-50:]
        
        result.test_samples += len(current_embeddings)
    
    async def _test_embedding_latency(self, result: ModelEvaluationResult):
        """Test embedding generation latency."""
        test_texts = [
            "milk",
            "organic chicken breast",
            "fresh strawberries from california",
            "whole wheat bread with added fiber and nutrients"
        ]
        
        response_times = []
        
        for text in test_texts:
            start_time = time.time()
            
            try:
                await self.embedding_service.generate_embedding(text)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
            except Exception as e:
                self.logger.warning(f"Latency test failed for '{text}': {e}")
                response_times.append(5.0)  # Penalty time
                result.errors_encountered += 1
        
        # Calculate latency metrics
        if response_times:
            avg_latency = statistics.mean(response_times)
            max_latency = max(response_times)
        else:
            avg_latency = 5.0
            max_latency = 5.0
        
        result.metrics['embedding_avg_latency'] = ModelMetric(
            metric_name="Embedding Average Latency",
            current_value=avg_latency,
            baseline_value=0.1,  # Expected 100ms
            threshold_warning=0.5,  # 500ms warning
            threshold_critical=1.0,  # 1s critical
            higher_is_better=False,
            unit="seconds"
        )
        
        result.latency_score = max(0.0, 1.0 - (avg_latency - 0.1) / 0.9)
        result.test_samples += len(test_texts)
    
    def _generate_embedding_recommendations(self, result: ModelEvaluationResult):
        """Generate embedding-specific recommendations."""
        # Check for drift
        if result.drift_detected:
            result.add_recommendation(
                'high', 'model', 'Embedding Drift Detected',
                'Significant drift detected in embedding space',
                [
                    'Retrain embedding model with recent data',
                    'Update vector database with new embeddings',
                    'Review data preprocessing pipeline',
                    'Consider ensemble embedding approaches'
                ]
            )
        
        # Check consistency
        consistency = result.metrics.get('embedding_consistency')
        if consistency and consistency.current_value < 0.95:
            result.add_recommendation(
                'medium', 'quality', 'Low Embedding Consistency',
                f'Embedding consistency ({consistency.current_value:.3f}) is below target',
                [
                    'Check for model loading issues',
                    'Verify deterministic behavior',
                    'Review random seed configuration',
                    'Monitor model serving infrastructure'
                ]
            )


class MLModelEvaluator:
    """Comprehensive ML model evaluation coordinator."""
    
    def __init__(self):
        self.llm_evaluator = LLMEvaluator()
        self.embedding_evaluator = EmbeddingEvaluator()
        self.logger = logging.getLogger(__name__)
        
        # Historical results for trend analysis
        self.evaluation_history: List[Tuple[datetime, Dict[str, ModelEvaluationResult]]] = []
    
    async def run_comprehensive_evaluation(self) -> Dict[str, ModelEvaluationResult]:
        """Run comprehensive evaluation of all ML models."""
        self.logger.info("Starting comprehensive ML model evaluation")
        
        results = {}
        
        try:
            # Run evaluations in parallel
            llm_task = self.llm_evaluator.evaluate_llm_performance()
            embedding_task = self.embedding_evaluator.evaluate_embedding_performance()
            
            llm_result, embedding_result = await asyncio.gather(
                llm_task, embedding_task, return_exceptions=True
            )
            
            if isinstance(llm_result, ModelEvaluationResult):
                results['llm'] = llm_result
            else:
                self.logger.error(f"LLM evaluation failed: {llm_result}")
            
            if isinstance(embedding_result, ModelEvaluationResult):
                results['embedding'] = embedding_result
            else:
                self.logger.error(f"Embedding evaluation failed: {embedding_result}")
            
            # Store results for trend analysis
            self.evaluation_history.append((datetime.now(), results))
            
            # Keep only last 20 evaluations
            if len(self.evaluation_history) > 20:
                self.evaluation_history = self.evaluation_history[-20:]
            
            self.logger.info(f"ML evaluation completed - {len(results)} models evaluated")
            
        except Exception as e:
            self.logger.error(f"ML model evaluation failed: {e}")
        
        return results
    
    def generate_evaluation_summary(self, results: Dict[str, ModelEvaluationResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': len(results),
            'overall_health': 'unknown',
            'critical_issues': [],
            'recommendations': [],
            'model_scores': {}
        }
        
        if not results:
            summary['overall_health'] = 'no_data'
            return summary
        
        # Calculate overall health
        total_score = 0.0
        model_count = 0
        
        for model_name, result in results.items():
            summary['model_scores'][model_name] = result.overall_score
            total_score += result.overall_score
            model_count += 1
            
            # Collect critical issues
            for metric_name, metric in result.metrics.items():
                if metric.drift_severity == "severe":
                    summary['critical_issues'].append({
                        'model': model_name,
                        'metric': metric_name,
                        'issue': f'Severe drift detected ({metric.drift_magnitude:.2%})'
                    })
            
            # Collect recommendations
            for rec in result.recommendations:
                if rec['priority'] == 'high':
                    summary['recommendations'].append({
                        'model': model_name,
                        'title': rec['title'],
                        'description': rec['description']
                    })
        
        if model_count > 0:
            avg_score = total_score / model_count
            
            if avg_score >= 80:
                summary['overall_health'] = 'healthy'
            elif avg_score >= 60:
                summary['overall_health'] = 'warning'
            else:
                summary['overall_health'] = 'critical'
        
        return summary
    
    def save_results(self, results: Dict[str, ModelEvaluationResult], output_file: str = None):
        """Save evaluation results to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation/results/ml_evaluation_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'evaluation_id': result.evaluation_id,
                'model_name': result.model_name,
                'model_type': result.model_type,
                'timestamp': result.timestamp.isoformat(),
                'overall_score': result.overall_score,
                'drift_detected': result.drift_detected,
                'drift_components': result.drift_components,
                'response_quality_score': result.response_quality_score,
                'consistency_score': result.consistency_score,
                'latency_score': result.latency_score,
                'retraining_recommended': result.retraining_recommended,
                'retraining_urgency': result.retraining_urgency,
                'execution_time': result.execution_time,
                'test_samples': result.test_samples,
                'errors_encountered': result.errors_encountered,
                'metrics': {
                    name: {
                        'metric_name': metric.metric_name,
                        'current_value': metric.current_value,
                        'baseline_value': metric.baseline_value,
                        'drift_detected': metric.drift_detected,
                        'drift_severity': metric.drift_severity,
                        'drift_magnitude': metric.drift_magnitude,
                        'unit': metric.unit
                    }
                    for name, metric in result.metrics.items()
                },
                'recommendations': result.recommendations
            }
        
        # Add summary
        summary = self.generate_evaluation_summary(results)
        
        final_data = {
            'summary': summary,
            'results': serializable_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2, default=str)
        
        print(f"✅ ML evaluation results saved to {output_path}")


if __name__ == "__main__":
    async def test_ml_evaluation():
        evaluator = MLModelEvaluator()
        results = await evaluator.run_comprehensive_evaluation()
        
        print(f"\n🤖 ML Model Evaluation Results:")
        
        for model_name, result in results.items():
            print(f"\n{model_name.upper()} Model:")
            print(f"  Overall Score: {result.overall_score:.1f}/100")
            print(f"  Test Samples: {result.test_samples}")
            print(f"  Execution Time: {result.execution_time:.2f}s")
            print(f"  Errors: {result.errors_encountered}")
            print(f"  Drift Detected: {'Yes' if result.drift_detected else 'No'}")
            print(f"  Retraining Recommended: {'Yes' if result.retraining_recommended else 'No'}")
            
            if result.recommendations:
                print(f"  High Priority Recommendations: {len([r for r in result.recommendations if r['priority'] == 'high'])}")
        
        # Generate and display summary
        summary = evaluator.generate_evaluation_summary(results)
        print(f"\n📊 Overall Health: {summary['overall_health'].upper()}")
        print(f"Critical Issues: {len(summary['critical_issues'])}")
        
        # Save results
        evaluator.save_results(results)
    
    asyncio.run(test_ml_evaluation())