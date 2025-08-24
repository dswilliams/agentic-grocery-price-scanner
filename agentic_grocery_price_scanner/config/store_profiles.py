"""
Store-specific optimization profiles for production-level reliability.
Profiles each store's response patterns and implements targeted optimizations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal

logger = logging.getLogger(__name__)


class StoreHealth(Enum):
    """Health status of a store endpoint."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class RetryStrategy(Enum):
    """Retry strategies for different store behaviors."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"


@dataclass
class StorePerformanceProfile:
    """Detailed performance profile for a specific store."""
    
    store_id: str
    
    # Response characteristics
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    success_rate: float = 100.0
    timeout_rate: float = 0.0
    error_rate: float = 0.0
    
    # Rate limiting behavior
    optimal_request_rate: float = 1.0
    burst_capacity: int = 5
    rate_limit_recovery_time: float = 60.0
    
    # Reliability patterns
    peak_hours: List[int] = field(default_factory=lambda: [9, 12, 18, 21])  # Hours when slow
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)
    known_outage_patterns: List[str] = field(default_factory=list)
    
    # Circuit breaker thresholds
    circuit_breaker_threshold: int = 5  # Failed requests before opening
    circuit_breaker_timeout: float = 30.0  # Seconds before retry
    circuit_breaker_recovery_requests: int = 3  # Successful requests to close
    
    # Retry configuration
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_retries: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    
    # Content parsing reliability
    selector_reliability: Dict[str, float] = field(default_factory=dict)
    fallback_selectors: Dict[str, List[str]] = field(default_factory=dict)
    
    # Quality metrics
    data_completeness_rate: float = 90.0
    price_accuracy_confidence: float = 95.0
    image_availability_rate: float = 80.0
    
    # Anti-bot detection patterns
    bot_detection_triggers: List[str] = field(default_factory=list)
    stealth_requirements: List[str] = field(default_factory=list)
    
    # Performance optimization
    concurrent_request_limit: int = 3
    connection_pool_size: int = 10
    keep_alive_duration: float = 300.0
    
    # Health monitoring
    health_status: StoreHealth = StoreHealth.HEALTHY
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    
    def update_performance_metrics(
        self, 
        response_time: float, 
        success: bool, 
        error_type: Optional[str] = None
    ):
        """Update performance metrics based on request outcome."""
        # Update response time (exponential moving average)
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = 0.9 * self.avg_response_time + 0.1 * response_time
        
        # Update success rate
        if success:
            self.consecutive_failures = 0
            self.success_rate = min(100.0, self.success_rate + 0.1)
        else:
            self.consecutive_failures += 1
            self.success_rate = max(0.0, self.success_rate - 1.0)
            
            if error_type == "timeout":
                self.timeout_rate = min(100.0, self.timeout_rate + 0.5)
            else:
                self.error_rate = min(100.0, self.error_rate + 0.5)
        
        # Update health status
        self._update_health_status()
    
    def _update_health_status(self):
        """Update health status based on current metrics."""
        if self.consecutive_failures >= self.circuit_breaker_threshold:
            self.health_status = StoreHealth.UNHEALTHY
        elif self.success_rate < 80.0 or self.error_rate > 20.0:
            self.health_status = StoreHealth.DEGRADED
        else:
            self.health_status = StoreHealth.HEALTHY
    
    def get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy and attempt number."""
        if self.retry_strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_retry_delay * (2 ** attempt)
        elif self.retry_strategy == RetryStrategy.LINEAR:
            delay = self.base_retry_delay * attempt
        elif self.retry_strategy == RetryStrategy.AGGRESSIVE:
            delay = self.base_retry_delay * 0.5
        elif self.retry_strategy == RetryStrategy.CONSERVATIVE:
            delay = self.base_retry_delay * 2
        else:  # FIXED
            delay = self.base_retry_delay
        
        return min(delay, self.max_retry_delay)
    
    def should_circuit_break(self) -> bool:
        """Check if circuit breaker should open."""
        return (
            self.consecutive_failures >= self.circuit_breaker_threshold or
            self.health_status == StoreHealth.UNHEALTHY
        )
    
    def is_in_peak_hours(self) -> bool:
        """Check if current time is in peak hours."""
        current_hour = datetime.now().hour
        return current_hour in self.peak_hours
    
    def get_optimal_request_rate(self) -> float:
        """Get optimal request rate based on current conditions."""
        base_rate = self.optimal_request_rate
        
        # Reduce rate during peak hours
        if self.is_in_peak_hours():
            base_rate *= 0.7
        
        # Reduce rate if degraded
        if self.health_status == StoreHealth.DEGRADED:
            base_rate *= 0.5
        elif self.health_status == StoreHealth.UNHEALTHY:
            base_rate = 0.0
        
        return base_rate


@dataclass 
class CircuitBreakerState:
    """Circuit breaker state for a store."""
    
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    next_retry_time: Optional[datetime] = None
    consecutive_successes: int = 0


class StoreProfileManager:
    """Manages store performance profiles and optimization strategies."""
    
    def __init__(self):
        self.profiles: Dict[str, StorePerformanceProfile] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize default profiles
        self._initialize_default_profiles()
        
        logger.info("Initialized StoreProfileManager with optimized profiles")
    
    def _initialize_default_profiles(self):
        """Initialize default performance profiles for known stores."""
        
        # Metro Canada - Generally reliable but slower during peak
        self.profiles["metro_ca"] = StorePerformanceProfile(
            store_id="metro_ca",
            avg_response_time=2.5,
            p95_response_time=5.0,
            success_rate=92.0,
            optimal_request_rate=0.5,  # Conservative rate
            peak_hours=[11, 12, 17, 18, 19, 20],
            circuit_breaker_threshold=3,
            retry_strategy=RetryStrategy.EXPONENTIAL,
            max_retries=4,
            base_retry_delay=2.0,
            selector_reliability={
                ".product-tile": 95.0,
                ".product-name": 98.0,
                ".price-update": 90.0,
                ".product-brand": 85.0
            },
            fallback_selectors={
                ".price-update": [".price", ".current-price", "[data-price]"],
                ".product-name": [".title", "h2", "h3", "[data-name]"]
            },
            bot_detection_triggers=["captcha", "access denied", "rate limit"],
            stealth_requirements=["user-agent", "headers", "delays"]
        )
        
        # Walmart Canada - Fast but aggressive bot detection
        self.profiles["walmart_ca"] = StorePerformanceProfile(
            store_id="walmart_ca", 
            avg_response_time=1.8,
            p95_response_time=3.5,
            success_rate=88.0,
            optimal_request_rate=0.8,
            peak_hours=[12, 18, 19, 20],
            circuit_breaker_threshold=4,
            retry_strategy=RetryStrategy.CONSERVATIVE,
            max_retries=3,
            base_retry_delay=3.0,
            selector_reliability={
                ".product-item": 92.0,
                ".product-name": 96.0,
                ".price-current": 94.0,
                ".product-brand": 88.0
            },
            fallback_selectors={
                ".price-current": [".price", "[data-automation-id*='price']", ".sr-only"],
                ".product-name": ["[data-automation-id*='name']", "h3", "h4"]
            },
            bot_detection_triggers=["blocked", "captcha", "unusual traffic"],
            stealth_requirements=["canadian-headers", "session-cookies", "human-delays"],
            concurrent_request_limit=2  # More conservative due to bot detection
        )
        
        # FreshCo - Smaller chain, less sophisticated but can be unstable
        self.profiles["freshco_com"] = StorePerformanceProfile(
            store_id="freshco_com",
            avg_response_time=3.2,
            p95_response_time=8.0,
            success_rate=85.0,
            optimal_request_rate=0.4,  # Very conservative
            peak_hours=[17, 18, 19],
            circuit_breaker_threshold=2,  # Lower threshold due to instability
            retry_strategy=RetryStrategy.LINEAR,
            max_retries=5,
            base_retry_delay=4.0,
            selector_reliability={
                ".product-card": 88.0,
                ".product-title": 92.0,
                ".price": 85.0,
                ".brand-name": 80.0
            },
            fallback_selectors={
                ".price": [".current-price", "[data-price]", ".cost"],
                ".product-title": [".name", "h2", "h3"]
            },
            bot_detection_triggers=["service unavailable", "maintenance"],
            stealth_requirements=["realistic-delays", "basic-headers"]
        )
        
        # Initialize circuit breakers
        for store_id in self.profiles:
            self.circuit_breakers[store_id] = CircuitBreakerState()
            self.performance_history[store_id] = []
    
    def get_profile(self, store_id: str) -> StorePerformanceProfile:
        """Get performance profile for a store."""
        return self.profiles.get(store_id, self._create_default_profile(store_id))
    
    def _create_default_profile(self, store_id: str) -> StorePerformanceProfile:
        """Create a default profile for unknown stores."""
        profile = StorePerformanceProfile(
            store_id=store_id,
            retry_strategy=RetryStrategy.CONSERVATIVE,
            max_retries=2,
            base_retry_delay=5.0,
            optimal_request_rate=0.3,  # Very conservative for unknown stores
            circuit_breaker_threshold=2
        )
        self.profiles[store_id] = profile
        self.circuit_breakers[store_id] = CircuitBreakerState()
        return profile
    
    def record_request(
        self,
        store_id: str,
        response_time: float,
        success: bool,
        error_type: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a request outcome and update profiles."""
        profile = self.get_profile(store_id)
        circuit_breaker = self.circuit_breakers[store_id]
        
        # Update profile metrics
        profile.update_performance_metrics(response_time, success, error_type)
        
        # Update circuit breaker
        if success:
            circuit_breaker.failure_count = 0
            circuit_breaker.consecutive_successes += 1
            
            # Close circuit breaker if enough successes
            if circuit_breaker.is_open and circuit_breaker.consecutive_successes >= profile.circuit_breaker_recovery_requests:
                circuit_breaker.is_open = False
                circuit_breaker.next_retry_time = None
                logger.info(f"Circuit breaker closed for {store_id}")
                
        else:
            circuit_breaker.failure_count += 1
            circuit_breaker.consecutive_successes = 0
            circuit_breaker.last_failure_time = datetime.now()
            
            # Open circuit breaker if threshold reached
            if circuit_breaker.failure_count >= profile.circuit_breaker_threshold:
                circuit_breaker.is_open = True
                circuit_breaker.next_retry_time = datetime.now() + timedelta(seconds=profile.circuit_breaker_timeout)
                logger.warning(f"Circuit breaker opened for {store_id}")
        
        # Record in history
        self.performance_history[store_id].append({
            "timestamp": datetime.now().isoformat(),
            "response_time": response_time,
            "success": success,
            "error_type": error_type,
            "metadata": additional_metadata or {}
        })
        
        # Limit history size
        if len(self.performance_history[store_id]) > 1000:
            self.performance_history[store_id] = self.performance_history[store_id][-500:]
    
    def is_store_available(self, store_id: str) -> bool:
        """Check if store is available for requests."""
        circuit_breaker = self.circuit_breakers.get(store_id)
        if not circuit_breaker:
            return True
            
        if circuit_breaker.is_open:
            if circuit_breaker.next_retry_time and datetime.now() >= circuit_breaker.next_retry_time:
                # Time to try again
                return True
            return False
            
        return True
    
    def get_request_delay(self, store_id: str, attempt: int = 0) -> float:
        """Get recommended delay before next request."""
        profile = self.get_profile(store_id)
        
        # Base delay from rate limiting
        base_delay = 1.0 / profile.get_optimal_request_rate() if profile.get_optimal_request_rate() > 0 else 60.0
        
        # Add retry delay if this is a retry
        if attempt > 0:
            retry_delay = profile.get_retry_delay(attempt)
            return base_delay + retry_delay
            
        return base_delay
    
    def get_store_health_report(self, store_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive health report for stores."""
        if store_id:
            stores = [store_id] if store_id in self.profiles else []
        else:
            stores = list(self.profiles.keys())
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "stores": {}
        }
        
        for sid in stores:
            profile = self.profiles[sid]
            circuit_breaker = self.circuit_breakers[sid]
            
            report["stores"][sid] = {
                "health_status": profile.health_status.value,
                "success_rate": profile.success_rate,
                "avg_response_time": profile.avg_response_time,
                "error_rate": profile.error_rate,
                "circuit_breaker_open": circuit_breaker.is_open,
                "consecutive_failures": profile.consecutive_failures,
                "available": self.is_store_available(sid),
                "optimal_rate": profile.get_optimal_request_rate()
            }
        
        return report
    
    def optimize_store_selection(
        self, 
        target_stores: List[str], 
        priority_criteria: str = "balanced"
    ) -> List[str]:
        """Optimize store selection based on current performance."""
        available_stores = [s for s in target_stores if self.is_store_available(s)]
        
        if not available_stores:
            logger.warning("No stores available, returning original list")
            return target_stores
        
        profiles = [(s, self.get_profile(s)) for s in available_stores]
        
        if priority_criteria == "speed":
            # Sort by response time
            profiles.sort(key=lambda x: x[1].avg_response_time)
        elif priority_criteria == "reliability":
            # Sort by success rate
            profiles.sort(key=lambda x: x[1].success_rate, reverse=True)
        elif priority_criteria == "balanced":
            # Composite score: success_rate / avg_response_time
            profiles.sort(key=lambda x: x[1].success_rate / max(x[1].avg_response_time, 0.1), reverse=True)
        
        return [s for s, _ in profiles]
    
    async def health_check_all_stores(self) -> Dict[str, bool]:
        """Perform health checks on all stores."""
        results = {}
        
        for store_id in self.profiles:
            try:
                # Simple health check - could be expanded to actual HTTP check
                profile = self.profiles[store_id]
                is_healthy = (
                    profile.health_status in [StoreHealth.HEALTHY, StoreHealth.DEGRADED] and
                    self.is_store_available(store_id)
                )
                results[store_id] = is_healthy
                profile.last_health_check = datetime.now()
                
            except Exception as e:
                logger.error(f"Health check failed for {store_id}: {e}")
                results[store_id] = False
        
        return results


# Global instance
store_profile_manager = StoreProfileManager()