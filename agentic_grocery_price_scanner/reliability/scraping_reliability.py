"""
Advanced scraping reliability framework with circuit breakers, progressive degradation,
and intelligent fallback mechanisms for production-level performance.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, AsyncIterator
from decimal import Decimal
import json
import aiohttp
import random

from ..config.store_profiles import StoreProfileManager, store_profile_manager
from ..data_models import Product

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of scraping failures."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error" 
    HTTP_ERROR = "http_error"
    PARSING_ERROR = "parsing_error"
    RATE_LIMITED = "rate_limited"
    BOT_DETECTED = "bot_detected"
    CONTENT_BLOCKED = "content_blocked"
    SERVER_OVERLOAD = "server_overload"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    IMMEDIATE_RETRY = "immediate_retry"
    DELAYED_RETRY = "delayed_retry"
    FALLBACK_METHOD = "fallback_method"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAK = "circuit_break"
    HUMAN_ESCALATION = "human_escalation"


@dataclass
class FailureContext:
    """Context information about a scraping failure."""
    
    store_id: str
    query: str
    failure_mode: FailureMode
    error_message: str
    timestamp: datetime
    response_time: Optional[float] = None
    http_status: Optional[int] = None
    retry_attempt: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestPoolEntry:
    """Entry in the request prioritization pool."""
    
    priority: int  # Higher = more urgent
    store_id: str
    query: str
    callback: Callable
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Priority queue ordering (higher priority first)."""
        return self.priority > other.priority


class ScrapingReliabilityManager:
    """Advanced scraping reliability framework with production-grade features."""
    
    def __init__(
        self,
        profile_manager: StoreProfileManager = None,
        max_concurrent_requests: int = 10,
        enable_progressive_degradation: bool = True,
        enable_cache_fallback: bool = True
    ):
        self.profile_manager = profile_manager or store_profile_manager
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_progressive_degradation = enable_progressive_degradation
        self.enable_cache_fallback = enable_cache_fallback
        
        # Request management
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.request_queue = asyncio.PriorityQueue()
        self.active_requests: Dict[str, int] = {}  # store_id -> active_count
        
        # Failure tracking
        self.recent_failures: List[FailureContext] = []
        self.failure_patterns: Dict[str, List[FailureContext]] = {}
        
        # Cache management
        self.price_cache: Dict[str, Dict[str, Any]] = {}  # store_id -> {query -> cached_data}
        self.cache_timestamps: Dict[str, Dict[str, datetime]] = {}
        self.cache_max_age = timedelta(hours=6)  # Reasonable for price data
        
        # Performance tracking
        self.request_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "circuit_breaker_activations": 0,
            "avg_response_time": 0.0
        }
        
        # Recovery handlers
        self.recovery_strategies: Dict[FailureMode, RecoveryStrategy] = {
            FailureMode.TIMEOUT: RecoveryStrategy.DELAYED_RETRY,
            FailureMode.CONNECTION_ERROR: RecoveryStrategy.IMMEDIATE_RETRY,
            FailureMode.HTTP_ERROR: RecoveryStrategy.FALLBACK_METHOD,
            FailureMode.PARSING_ERROR: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureMode.RATE_LIMITED: RecoveryStrategy.CIRCUIT_BREAK,
            FailureMode.BOT_DETECTED: RecoveryStrategy.HUMAN_ESCALATION,
            FailureMode.CONTENT_BLOCKED: RecoveryStrategy.FALLBACK_METHOD,
            FailureMode.SERVER_OVERLOAD: RecoveryStrategy.DELAYED_RETRY
        }
        
        logger.info("Initialized ScrapingReliabilityManager with production features")
    
    async def execute_reliable_scraping(
        self,
        store_id: str,
        query: str,
        scraping_method: Callable,
        priority: int = 5,
        max_retries: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        fallback_data: Optional[List[Product]] = None
    ) -> List[Product]:
        """Execute scraping with full reliability framework."""
        
        # Check circuit breaker
        if not self.profile_manager.is_store_available(store_id):
            logger.warning(f"Store {store_id} circuit breaker is open")
            if self.enable_cache_fallback:
                cached_data = await self._get_cached_data(store_id, query)
                if cached_data:
                    return cached_data
            if fallback_data:
                return fallback_data
            raise Exception(f"Store {store_id} is unavailable and no fallback data provided")
        
        # Check cache first
        if self.enable_cache_fallback:
            cached_data = await self._get_cached_data(store_id, query)
            if cached_data:
                self.performance_metrics["cache_hits"] += 1
                logger.debug(f"Cache hit for {store_id}:{query}")
                return cached_data
        
        # Execute with reliability framework
        profile = self.profile_manager.get_profile(store_id)
        max_retries = max_retries or profile.max_retries
        timeout_seconds = timeout_seconds or 30.0
        
        for attempt in range(max_retries + 1):
            try:
                # Wait for rate limiting
                await self._enforce_rate_limiting(store_id, attempt)
                
                # Execute request with timeout
                start_time = time.time()
                
                async with self.request_semaphore:
                    self._track_active_request(store_id, 1)
                    try:
                        result = await asyncio.wait_for(
                            scraping_method(store_id, query),
                            timeout=timeout_seconds
                        )
                        
                        response_time = time.time() - start_time
                        
                        # Record success
                        self.profile_manager.record_request(store_id, response_time, True)
                        self._update_performance_metrics(response_time, True)
                        
                        # Cache results
                        if result:
                            await self._cache_data(store_id, query, result)
                        
                        logger.debug(f"Successful scraping for {store_id}:{query} in {response_time:.2f}s")
                        return result
                        
                    finally:
                        self._track_active_request(store_id, -1)
            
            except asyncio.TimeoutError:
                await self._handle_failure(
                    FailureContext(
                        store_id=store_id,
                        query=query,
                        failure_mode=FailureMode.TIMEOUT,
                        error_message="Request timed out",
                        timestamp=datetime.now(),
                        response_time=timeout_seconds,
                        retry_attempt=attempt
                    ),
                    attempt,
                    max_retries
                )
                
            except aiohttp.ClientError as e:
                await self._handle_failure(
                    FailureContext(
                        store_id=store_id,
                        query=query,
                        failure_mode=FailureMode.CONNECTION_ERROR,
                        error_message=str(e),
                        timestamp=datetime.now(),
                        retry_attempt=attempt
                    ),
                    attempt,
                    max_retries
                )
                
            except Exception as e:
                failure_mode = self._classify_error(str(e))
                await self._handle_failure(
                    FailureContext(
                        store_id=store_id,
                        query=query,
                        failure_mode=failure_mode,
                        error_message=str(e),
                        timestamp=datetime.now(),
                        retry_attempt=attempt
                    ),
                    attempt,
                    max_retries
                )
        
        # All retries exhausted - try fallback strategies
        return await self._execute_fallback_strategies(store_id, query, fallback_data)
    
    async def _enforce_rate_limiting(self, store_id: str, attempt: int):
        """Enforce intelligent rate limiting based on store profile."""
        delay = self.profile_manager.get_request_delay(store_id, attempt)
        
        if delay > 0:
            # Add jitter to prevent thundering herd
            jittered_delay = delay * (0.8 + 0.4 * random.random())
            logger.debug(f"Rate limiting delay for {store_id}: {jittered_delay:.2f}s")
            await asyncio.sleep(jittered_delay)
    
    def _track_active_request(self, store_id: str, delta: int):
        """Track active requests per store."""
        self.active_requests[store_id] = self.active_requests.get(store_id, 0) + delta
        self.active_requests[store_id] = max(0, self.active_requests[store_id])
    
    def _classify_error(self, error_message: str) -> FailureMode:
        """Classify error into failure mode."""
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in ["rate limit", "too many requests"]):
            return FailureMode.RATE_LIMITED
        elif any(keyword in error_lower for keyword in ["captcha", "bot detected", "blocked"]):
            return FailureMode.BOT_DETECTED
        elif any(keyword in error_lower for keyword in ["timeout", "timed out"]):
            return FailureMode.TIMEOUT
        elif any(keyword in error_lower for keyword in ["connection", "network", "dns"]):
            return FailureMode.CONNECTION_ERROR
        elif any(keyword in error_lower for keyword in ["parse", "selector", "element"]):
            return FailureMode.PARSING_ERROR
        elif any(keyword in error_lower for keyword in ["503", "502", "overload", "unavailable"]):
            return FailureMode.SERVER_OVERLOAD
        else:
            return FailureMode.HTTP_ERROR
    
    async def _handle_failure(
        self,
        failure_context: FailureContext,
        attempt: int,
        max_retries: int
    ):
        """Handle scraping failure with appropriate recovery strategy."""
        
        # Record failure
        response_time = failure_context.response_time or 0.0
        self.profile_manager.record_request(
            failure_context.store_id,
            response_time,
            False,
            failure_context.failure_mode.value
        )
        self._update_performance_metrics(response_time, False)
        
        # Track failure patterns
        self.recent_failures.append(failure_context)
        if len(self.recent_failures) > 100:
            self.recent_failures = self.recent_failures[-50:]
        
        store_failures = self.failure_patterns.setdefault(failure_context.store_id, [])
        store_failures.append(failure_context)
        if len(store_failures) > 50:
            self.failure_patterns[failure_context.store_id] = store_failures[-25:]
        
        # Execute recovery strategy
        strategy = self.recovery_strategies.get(
            failure_context.failure_mode, 
            RecoveryStrategy.DELAYED_RETRY
        )
        
        logger.warning(
            f"Failure in {failure_context.store_id}:{failure_context.query} "
            f"(attempt {attempt + 1}/{max_retries + 1}): {failure_context.error_message}. "
            f"Strategy: {strategy.value}"
        )
        
        if attempt < max_retries:
            if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
                # No additional delay
                pass
            elif strategy == RecoveryStrategy.DELAYED_RETRY:
                profile = self.profile_manager.get_profile(failure_context.store_id)
                delay = profile.get_retry_delay(attempt + 1)
                await asyncio.sleep(delay)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
                # Circuit breaker will be triggered by profile manager
                pass
    
    async def _execute_fallback_strategies(
        self,
        store_id: str,
        query: str,
        fallback_data: Optional[List[Product]] = None
    ) -> List[Product]:
        """Execute fallback strategies when all retries are exhausted."""
        
        logger.warning(f"All retries exhausted for {store_id}:{query}, executing fallbacks")
        
        # Try cache with extended age
        if self.enable_cache_fallback:
            cached_data = await self._get_cached_data(store_id, query, max_age=timedelta(days=1))
            if cached_data:
                logger.info(f"Using stale cache data for {store_id}:{query}")
                return cached_data
        
        # Try progressive degradation
        if self.enable_progressive_degradation:
            degraded_data = await self._get_degraded_data(store_id, query)
            if degraded_data:
                logger.info(f"Using degraded data for {store_id}:{query}")
                return degraded_data
        
        # Use provided fallback data
        if fallback_data:
            logger.info(f"Using provided fallback data for {store_id}:{query}")
            return fallback_data
        
        # Last resort - return empty list
        logger.error(f"No fallback options available for {store_id}:{query}")
        return []
    
    async def _get_cached_data(
        self,
        store_id: str,
        query: str,
        max_age: Optional[timedelta] = None
    ) -> Optional[List[Product]]:
        """Retrieve cached data if available and fresh."""
        max_age = max_age or self.cache_max_age
        
        if store_id not in self.price_cache:
            return None
            
        store_cache = self.price_cache[store_id]
        if query not in store_cache:
            return None
            
        # Check cache freshness
        timestamps = self.cache_timestamps.get(store_id, {})
        cache_time = timestamps.get(query)
        
        if not cache_time or datetime.now() - cache_time > max_age:
            # Cache expired
            return None
        
        return store_cache[query]
    
    async def _cache_data(
        self,
        store_id: str,
        query: str,
        data: List[Product]
    ):
        """Cache scraped data for future use."""
        if store_id not in self.price_cache:
            self.price_cache[store_id] = {}
            self.cache_timestamps[store_id] = {}
        
        self.price_cache[store_id][query] = data
        self.cache_timestamps[store_id][query] = datetime.now()
        
        # Limit cache size per store
        store_cache = self.price_cache[store_id]
        if len(store_cache) > 100:
            # Remove oldest entries
            timestamps = self.cache_timestamps[store_id]
            oldest_queries = sorted(timestamps.keys(), key=lambda q: timestamps[q])[:20]
            
            for old_query in oldest_queries:
                del store_cache[old_query]
                del timestamps[old_query]
    
    async def _get_degraded_data(
        self,
        store_id: str,
        query: str
    ) -> Optional[List[Product]]:
        """Get degraded data when primary scraping fails."""
        
        # Could implement various degradation strategies:
        # 1. Use similar product queries from cache
        # 2. Use generic pricing estimates
        # 3. Use competitor data as proxy
        
        # For now, try to find similar cached queries
        if store_id not in self.price_cache:
            return None
        
        store_cache = self.price_cache[store_id]
        query_lower = query.lower()
        
        # Look for partial matches in cached queries
        for cached_query, cached_data in store_cache.items():
            if (query_lower in cached_query.lower() or 
                cached_query.lower() in query_lower):
                
                logger.info(f"Using similar cached data: '{cached_query}' for query '{query}'")
                return cached_data
        
        return None
    
    def _update_performance_metrics(self, response_time: float, success: bool):
        """Update global performance metrics."""
        self.performance_metrics["total_requests"] += 1
        
        if success:
            self.performance_metrics["successful_requests"] += 1
        else:
            self.performance_metrics["failed_requests"] += 1
        
        # Update average response time (exponential moving average)
        current_avg = self.performance_metrics["avg_response_time"]
        if current_avg == 0:
            self.performance_metrics["avg_response_time"] = response_time
        else:
            self.performance_metrics["avg_response_time"] = 0.9 * current_avg + 0.1 * response_time
    
    def get_reliability_report(self) -> Dict[str, Any]:
        """Generate comprehensive reliability report."""
        total_requests = self.performance_metrics["total_requests"]
        successful_requests = self.performance_metrics["successful_requests"]
        
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        # Analyze recent failures
        recent_failures_by_mode = {}
        for failure in self.recent_failures[-50:]:  # Last 50 failures
            mode = failure.failure_mode.value
            recent_failures_by_mode[mode] = recent_failures_by_mode.get(mode, 0) + 1
        
        # Store health summary
        store_health = self.profile_manager.get_store_health_report()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": {
                "total_requests": total_requests,
                "success_rate": success_rate,
                "avg_response_time": self.performance_metrics["avg_response_time"],
                "cache_hit_rate": (self.performance_metrics["cache_hits"] / max(total_requests, 1)) * 100
            },
            "failure_analysis": {
                "recent_failure_modes": recent_failures_by_mode,
                "total_failures": len(self.recent_failures)
            },
            "store_health": store_health,
            "cache_statistics": {
                "stores_cached": len(self.price_cache),
                "total_cached_queries": sum(len(cache) for cache in self.price_cache.values())
            },
            "active_requests": dict(self.active_requests)
        }
    
    async def perform_health_checks(self) -> Dict[str, Any]:
        """Perform comprehensive health checks."""
        logger.info("Starting reliability health checks")
        
        # Check store availability
        store_health = await self.profile_manager.health_check_all_stores()
        
        # Check cache health
        cache_stats = {
            "total_entries": sum(len(cache) for cache in self.price_cache.values()),
            "stores_with_cache": len(self.price_cache),
            "avg_entries_per_store": sum(len(cache) for cache in self.price_cache.values()) / max(len(self.price_cache), 1)
        }
        
        # Check recent performance
        recent_success_rate = 0
        if self.performance_metrics["total_requests"] > 0:
            recent_success_rate = (
                self.performance_metrics["successful_requests"] / 
                self.performance_metrics["total_requests"] * 100
            )
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy" if recent_success_rate > 80 else "degraded" if recent_success_rate > 50 else "unhealthy",
            "store_availability": store_health,
            "cache_health": cache_stats,
            "performance_summary": {
                "recent_success_rate": recent_success_rate,
                "avg_response_time": self.performance_metrics["avg_response_time"],
                "active_requests": sum(self.active_requests.values())
            }
        }
        
        logger.info(f"Health check completed: {health_report['overall_health']}")
        return health_report


# Global instance for easy access
scraping_reliability_manager = ScrapingReliabilityManager()