"""
Ollama client for local LLM inference with model routing and specialized tasks.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

import aiohttp


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types with their specializations."""
    QWEN_1_5B = "qwen2.5:1.5b"  # Fast ingredient matching, brand normalization
    PHI3_5_MINI = "phi3.5:latest"  # Complex reasoning, optimization decisions
    PHI3_3_8B = "phi3:3.8b"  # Alternative reasoning model


@dataclass
class ModelCapabilities:
    """Defines model capabilities and use cases."""
    model: ModelType
    strengths: List[str]
    max_tokens: int
    use_cases: List[str]


# Model capability definitions
MODEL_SPECS = {
    ModelType.QWEN_1_5B: ModelCapabilities(
        model=ModelType.QWEN_1_5B,
        strengths=["fast_inference", "pattern_matching", "normalization"],
        max_tokens=4096,
        use_cases=["ingredient_matching", "brand_normalization", "quick_classification"]
    ),
    ModelType.PHI3_5_MINI: ModelCapabilities(
        model=ModelType.PHI3_5_MINI,
        strengths=["reasoning", "analysis", "decision_making"],
        max_tokens=8192,
        use_cases=["complex_reasoning", "optimization", "strategy_decisions"]
    ),
    ModelType.PHI3_3_8B: ModelCapabilities(
        model=ModelType.PHI3_3_8B,
        strengths=["reasoning", "analysis", "code_generation"],
        max_tokens=8192,
        use_cases=["complex_reasoning", "code_analysis", "detailed_explanations"]
    )
}


class OllamaClient:
    """
    Async client for Ollama local LLM inference with intelligent model routing.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        max_retries: int = 3,
        enable_caching: bool = True
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_caching = enable_caching
        
        # Response cache for repeated queries
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}
        self.cache_duration = 300  # 5 minutes
        
        # Model availability tracking
        self._available_models: Optional[List[str]] = None
        self._model_load_times: Dict[str, float] = {}
        
    async def _make_request(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make async HTTP request to Ollama API."""
        url = f"{self.base_url}{endpoint}"
        request_timeout = timeout or self.timeout
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout)
        ) as session:
            try:
                if method.upper() == "GET":
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise Exception(f"HTTP {response.status}: {error_text}")
                else:
                    async with session.post(url, json=data) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise Exception(f"HTTP {response.status}: {error_text}")
            except aiohttp.ClientError as e:
                raise Exception(f"Request failed: {str(e)}")
    
    async def check_model_availability(self) -> List[str]:
        """Check which models are available in Ollama."""
        if self._available_models is not None:
            return self._available_models
            
        try:
            response = await self._make_request("/api/tags", method="GET")
            models = [model["name"] for model in response.get("models", [])]
            self._available_models = models
            logger.info(f"Available models: {models}")
            return models
        except Exception as e:
            logger.error(f"Failed to fetch available models: {e}")
            return []
    
    def _get_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key for request."""
        key_data = {"prompt": prompt, "model": model, **kwargs}
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached response is still valid."""
        if not self.enable_caching or cache_key not in self._cache:
            return False
        return time.time() - self._cache_ttl[cache_key] < self.cache_duration
    
    def _cache_response(self, cache_key: str, response: Any) -> None:
        """Cache response with timestamp."""
        if self.enable_caching:
            self._cache[cache_key] = response
            self._cache_ttl[cache_key] = time.time()
    
    async def generate(
        self,
        prompt: str,
        model: Optional[ModelType] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response using specified model or auto-select best model.
        
        Args:
            prompt: Input prompt
            model: Specific model to use (auto-selected if None)
            system_prompt: System prompt to set context
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Auto-select model if not specified
        if model is None:
            model = await self._select_best_model(prompt)
        
        model_name = model.value
        
        # Check cache first
        cache_key = self._get_cache_key(
            prompt, model_name, system_prompt=system_prompt, 
            temperature=temperature, **kwargs
        )
        
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {model_name}")
            return self._cache[cache_key]
        
        # Prepare request data
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                **({"num_predict": max_tokens} if max_tokens else {}),
                **kwargs
            }
        }
        
        if system_prompt:
            request_data["system"] = system_prompt
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = await self._make_request("/api/generate", request_data)
                inference_time = time.time() - start_time
                
                self._model_load_times[model_name] = inference_time
                
                generated_text = response.get("response", "")
                
                # Cache successful response
                self._cache_response(cache_key, generated_text)
                
                logger.debug(
                    f"Generated {len(generated_text)} chars with {model_name} "
                    f"in {inference_time:.2f}s"
                )
                
                return generated_text
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"All {self.max_retries} attempts failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return ""
    
    async def structured_output(
        self,
        prompt: str,
        response_schema: Dict[str, Any],
        model: Optional[ModelType] = None,
        max_attempts: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response matching provided schema.
        
        Args:
            prompt: Input prompt
            response_schema: Expected JSON schema
            model: Model to use (auto-selected if None)
            max_attempts: Maximum parsing attempts
            **kwargs: Additional generation parameters
            
        Returns:
            Parsed JSON response matching schema
        """
        schema_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema: {json.dumps(response_schema)}\n"
            f"Return ONLY the JSON object, no additional text."
        )
        
        for attempt in range(max_attempts):
            try:
                response = await self.generate(
                    schema_prompt,
                    model=model,
                    temperature=0.1,  # Low temperature for structured output
                    **kwargs
                )
                
                # Clean response and parse JSON
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                
                parsed_response = json.loads(cleaned_response.strip())
                
                # Basic schema validation
                if self._validate_schema(parsed_response, response_schema):
                    return parsed_response
                else:
                    logger.warning(f"Response doesn't match schema (attempt {attempt + 1})")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.warning(f"Structured output failed (attempt {attempt + 1}): {e}")
        
        # Fallback: return empty structure matching schema
        return self._create_empty_schema(response_schema)
    
    def _validate_schema(self, response: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Basic schema validation."""
        if not isinstance(response, dict):
            return False
        
        # Check required keys exist
        required_keys = schema.get("required", [])
        for key in required_keys:
            if key not in response:
                return False
        
        return True
    
    def _create_empty_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create empty response matching schema."""
        empty_response = {}
        properties = schema.get("properties", {})
        
        for key, prop in properties.items():
            prop_type = prop.get("type", "string")
            if prop_type == "string":
                empty_response[key] = ""
            elif prop_type == "number":
                empty_response[key] = 0
            elif prop_type == "boolean":
                empty_response[key] = False
            elif prop_type == "array":
                empty_response[key] = []
            elif prop_type == "object":
                empty_response[key] = {}
        
        return empty_response
    
    async def _select_best_model(self, prompt: str) -> ModelType:
        """
        Intelligently select best model for the given prompt.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            Best model type for the task
        """
        # Ensure models are available
        available_models = await self.check_model_availability()
        
        # Analyze prompt characteristics
        prompt_lower = prompt.lower()
        
        # Fast pattern matching tasks -> Qwen
        if any(keyword in prompt_lower for keyword in [
            "normalize", "match", "classify", "extract", "ingredient", 
            "brand", "quick", "simple", "fast"
        ]):
            if ModelType.QWEN_1_5B.value in available_models:
                return ModelType.QWEN_1_5B
        
        # Complex reasoning tasks -> Phi-3.5
        if any(keyword in prompt_lower for keyword in [
            "analyze", "reason", "decide", "optimize", "strategy", 
            "complex", "explain", "compare", "evaluate"
        ]):
            if ModelType.PHI3_5_MINI.value in available_models:
                return ModelType.PHI3_5_MINI
            elif ModelType.PHI3_3_8B.value in available_models:
                return ModelType.PHI3_3_8B
        
        # Fallback selection based on availability and performance
        model_preferences = [
            ModelType.PHI3_5_MINI,
            ModelType.QWEN_1_5B,
            ModelType.PHI3_3_8B
        ]
        
        for model in model_preferences:
            if model.value in available_models:
                return model
        
        # If no preferred models available, use the first available
        if available_models:
            for model_type in ModelType:
                if model_type.value in available_models:
                    return model_type
        
        # Default fallback
        return ModelType.QWEN_1_5B
    
    async def batch_generate(
        self,
        prompts: List[str],
        model: Optional[ModelType] = None,
        max_concurrent: int = 3,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts concurrently.
        
        Args:
            prompts: List of input prompts
            model: Model to use (auto-selected if None)
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt, model=model, **kwargs)
        
        tasks = [_generate_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get performance statistics for loaded models."""
        return {
            "available_models": self._available_models or [],
            "load_times": self._model_load_times,
            "cache_size": len(self._cache),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self._cache:
            return 0.0
        
        # This is a simplified calculation
        # In a real implementation, you'd track hits/misses
        return len(self._cache) / max(len(self._cache) * 2, 1)
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        self._cache.clear()
        self._cache_ttl.clear()
        logger.info("Cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health and model availability."""
        try:
            # Test basic connectivity
            response = await self._make_request("/api/tags", method="GET")
            
            # Test model inference
            test_model = await self._select_best_model("test")
            test_response = await self.generate(
                "Say 'OK' if you can respond",
                model=test_model,
                max_tokens=5
            )
            
            return {
                "status": "healthy",
                "service_available": True,
                "models_available": len(self._available_models or []),
                "test_inference": "OK" in test_response.upper()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "service_available": False,
                "error": str(e)
            }