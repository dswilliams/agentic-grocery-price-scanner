"""
Production-level reliability framework for grocery price scraping.
Provides circuit breakers, progressive degradation, and intelligent fallback mechanisms.
"""

from .scraping_reliability import (
    ScrapingReliabilityManager,
    scraping_reliability_manager,
    FailureMode,
    RecoveryStrategy,
    FailureContext
)

__all__ = [
    "ScrapingReliabilityManager",
    "scraping_reliability_manager", 
    "FailureMode",
    "RecoveryStrategy",
    "FailureContext"
]