"""
Multi-tier intelligent caching system with cache warming, invalidation,
and distributed cache preparation for production-level performance.
"""

import asyncio
import logging
import json
import pickle
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from collections import defaultdict
import threading
import sqlite3
import tempfile
import os

from ..data_models import Product
from ..config.store_profiles import store_profile_manager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheLevel(Enum):
    """Cache tier levels."""
    MEMORY = "memory"
    DISK = "disk"
    VECTOR_DB = "vector_db"
    DISTRIBUTED = "distributed"  # For future use


class CachePolicy(Enum):
    """Cache invalidation policies."""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    LFU = "lfu"  # Least Frequently Used
    ADAPTIVE = "adaptive"  # Intelligent adaptive policy


@dataclass
class CacheEntry(Generic[T]):
    """Individual cache entry with metadata."""
    
    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        age_seconds = (datetime.now() - self.created_at).total_seconds()
        return age_seconds > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def calculate_size(self):
        """Calculate approximate size of cached value."""
        try:
            self.size_bytes = len(pickle.dumps(self.value))
        except Exception:
            # Fallback size estimation
            self.size_bytes = len(str(self.value)) * 2  # Rough estimate


class MemoryCache:
    """High-performance in-memory cache tier."""
    
    def __init__(self, max_size_mb: float = 100.0, policy: CachePolicy = CachePolicy.LRU):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.policy = policy
        
        self.entries: Dict[str, CacheEntry] = {}
        self.current_size_bytes = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"Initialized MemoryCache with {max_size_mb}MB capacity")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from memory cache."""
        with self.lock:
            if key not in self.entries:
                self.misses += 1
                return None
            
            entry = self.entries[key]
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access metadata
            entry.update_access()
            self.hits += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None, tags: Optional[Dict[str, str]] = None):
        """Store value in memory cache."""
        with self.lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl_seconds,
                tags=tags or {}
            )
            entry.calculate_size()
            
            # Remove existing entry if present
            if key in self.entries:
                self._remove_entry(key)
            
            # Check if we need to make space
            while (self.current_size_bytes + entry.size_bytes > self.max_size_bytes and 
                   len(self.entries) > 0):
                self._evict_entry()
            
            # Store entry
            self.entries[key] = entry
            self.current_size_bytes += entry.size_bytes
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self.lock:
            if key in self.entries:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self, tag_filter: Optional[str] = None):
        """Clear cache entries, optionally filtered by tag."""
        with self.lock:
            if tag_filter is None:
                self.entries.clear()
                self.current_size_bytes = 0
            else:
                keys_to_remove = []
                for key, entry in self.entries.items():
                    if tag_filter in entry.tags.values():
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """Internal method to remove entry."""
        if key in self.entries:
            entry = self.entries.pop(key)
            self.current_size_bytes -= entry.size_bytes
    
    def _evict_entry(self):
        """Evict an entry based on cache policy."""
        if not self.entries:
            return
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            oldest_key = min(self.entries.keys(), 
                           key=lambda k: self.entries[k].last_accessed)
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            oldest_key = min(self.entries.keys(),
                           key=lambda k: self.entries[k].access_count)
        else:  # Default to LRU
            oldest_key = min(self.entries.keys(),
                           key=lambda k: self.entries[k].last_accessed)
        
        self._remove_entry(oldest_key)
        self.evictions += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "entries": len(self.entries),
                "size_mb": self.current_size_bytes / 1024 / 1024,
                "max_size_mb": self.max_size_bytes / 1024 / 1024,
                "utilization": (self.current_size_bytes / self.max_size_bytes) * 100,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions
            }


class DiskCache:
    """Persistent disk-based cache tier."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: float = 500.0):
        self.cache_dir = Path(cache_dir or tempfile.gettempdir()) / "grocery_scanner_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.db_path = self.cache_dir / "cache_metadata.db"
        
        self.lock = threading.RLock()
        self._init_database()
        
        logger.info(f"Initialized DiskCache at {self.cache_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    ttl_seconds INTEGER,
                    size_bytes INTEGER NOT NULL,
                    tags TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_metadata(last_accessed)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache_metadata(created_at)')
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from disk cache."""
        with self.lock:
            # Get metadata from database
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM cache_metadata WHERE key = ?', 
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                # Check expiration
                created_at = datetime.fromisoformat(row['created_at'])
                if row['ttl_seconds'] and (datetime.now() - created_at).total_seconds() > row['ttl_seconds']:
                    self._remove_entry(key, conn)
                    return None
                
                # Load value from disk
                filename = row['filename']
                file_path = self.cache_dir / filename
                
                if not file_path.exists():
                    self._remove_entry(key, conn)
                    return None
                
                try:
                    with open(file_path, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Update access metadata
                    conn.execute('''
                        UPDATE cache_metadata 
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE key = ?
                    ''', (datetime.now().isoformat(), key))
                    
                    return value
                
                except Exception as e:
                    logger.error(f"Error loading cached value for key {key}: {e}")
                    self._remove_entry(key, conn)
                    return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None, tags: Optional[Dict[str, str]] = None):
        """Store value in disk cache."""
        with self.lock:
            # Generate filename
            filename = hashlib.md5(key.encode()).hexdigest() + ".pkl"
            file_path = self.cache_dir / filename
            
            try:
                # Serialize to disk
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                size_bytes = file_path.stat().st_size
                current_time = datetime.now().isoformat()
                
                # Store metadata
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO cache_metadata
                        (key, filename, created_at, last_accessed, ttl_seconds, size_bytes, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        key, filename, current_time, current_time, 
                        ttl_seconds, size_bytes, json.dumps(tags or {})
                    ))
                    
                    # Check if we need to evict entries
                    self._enforce_size_limit(conn)
                
            except Exception as e:
                logger.error(f"Error storing value in disk cache: {e}")
                if file_path.exists():
                    file_path.unlink()
    
    def remove(self, key: str) -> bool:
        """Remove entry from disk cache."""
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                return self._remove_entry(key, conn)
    
    def _remove_entry(self, key: str, conn: sqlite3.Connection) -> bool:
        """Remove entry (internal method)."""
        cursor = conn.execute('SELECT filename FROM cache_metadata WHERE key = ?', (key,))
        row = cursor.fetchone()
        
        if row:
            filename = row[0]
            file_path = self.cache_dir / filename
            
            # Remove file
            if file_path.exists():
                file_path.unlink()
            
            # Remove metadata
            conn.execute('DELETE FROM cache_metadata WHERE key = ?', (key,))
            return True
        
        return False
    
    def _enforce_size_limit(self, conn: sqlite3.Connection):
        """Enforce cache size limits by evicting old entries."""
        # Get current cache size
        cursor = conn.execute('SELECT SUM(size_bytes) FROM cache_metadata')
        current_size = cursor.fetchone()[0] or 0
        
        if current_size <= self.max_size_bytes:
            return
        
        # Evict entries (LRU order) until under limit
        cursor = conn.execute('''
            SELECT key, filename, size_bytes 
            FROM cache_metadata 
            ORDER BY last_accessed ASC
        ''')
        
        for key, filename, size_bytes in cursor:
            if current_size <= self.max_size_bytes:
                break
            
            # Remove file
            file_path = self.cache_dir / filename
            if file_path.exists():
                file_path.unlink()
            
            # Remove metadata
            conn.execute('DELETE FROM cache_metadata WHERE key = ?', (key,))
            current_size -= size_bytes
    
    def clear(self, tag_filter: Optional[str] = None):
        """Clear disk cache entries."""
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                if tag_filter is None:
                    # Clear everything
                    cursor = conn.execute('SELECT filename FROM cache_metadata')
                    for (filename,) in cursor:
                        file_path = self.cache_dir / filename
                        if file_path.exists():
                            file_path.unlink()
                    conn.execute('DELETE FROM cache_metadata')
                else:
                    # Clear by tag filter
                    cursor = conn.execute('SELECT key, filename, tags FROM cache_metadata')
                    keys_to_remove = []
                    
                    for key, filename, tags_json in cursor:
                        tags = json.loads(tags_json or '{}')
                        if tag_filter in tags.values():
                            keys_to_remove.append((key, filename))
                    
                    for key, filename in keys_to_remove:
                        file_path = self.cache_dir / filename
                        if file_path.exists():
                            file_path.unlink()
                        conn.execute('DELETE FROM cache_metadata WHERE key = ?', (key,))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get disk cache statistics."""
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                
                # Basic counts
                cursor = conn.execute('SELECT COUNT(*), SUM(size_bytes), AVG(access_count) FROM cache_metadata')
                row = cursor.fetchone()
                
                entry_count = row[0] or 0
                total_size_bytes = row[1] or 0
                avg_access_count = row[2] or 0
                
                # Recent activity
                one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM cache_metadata WHERE last_accessed > ?',
                    (one_hour_ago,)
                )
                recent_accesses = cursor.fetchone()[0] or 0
                
                return {
                    "entries": entry_count,
                    "size_mb": total_size_bytes / 1024 / 1024,
                    "max_size_mb": self.max_size_bytes / 1024 / 1024,
                    "utilization": (total_size_bytes / self.max_size_bytes) * 100,
                    "avg_access_count": avg_access_count,
                    "recent_accesses": recent_accesses,
                    "cache_dir": str(self.cache_dir)
                }


class CacheManager:
    """Multi-tier intelligent cache manager."""
    
    def __init__(
        self,
        memory_cache_mb: float = 100.0,
        disk_cache_mb: float = 500.0,
        enable_cache_warming: bool = True
    ):
        self.memory_cache = MemoryCache(memory_cache_mb)
        self.disk_cache = DiskCache(max_size_mb=disk_cache_mb)
        self.enable_cache_warming = enable_cache_warming
        
        # Cache warming and invalidation
        self.warming_tasks: Dict[str, asyncio.Task] = {}
        self.invalidation_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Default TTL values by data type
        self.default_ttls = {
            "product": 3600,  # 1 hour
            "search_results": 1800,  # 30 minutes
            "store_status": 300,  # 5 minutes
            "quality_metrics": 3600,  # 1 hour
            "optimization_results": 7200  # 2 hours
        }
        
        # Intelligent caching patterns
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.popular_keys: Dict[str, float] = {}  # key -> popularity score
        
        logger.info("Initialized multi-tier CacheManager")
    
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from multi-tier cache."""
        # Try memory cache first (fastest)
        value = self.memory_cache.get(key)
        if value is not None:
            self._record_access(key)
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            data_type = self._extract_data_type(key)
            ttl = self.default_ttls.get(data_type, 3600)
            self.memory_cache.put(key, value, ttl_seconds=ttl)
            self._record_access(key)
            return value
        
        return None
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        cache_levels: Optional[List[CacheLevel]] = None
    ):
        """Store value in specified cache levels."""
        cache_levels = cache_levels or [CacheLevel.MEMORY, CacheLevel.DISK]
        
        # Determine TTL
        if ttl_seconds is None:
            data_type = self._extract_data_type(key)
            ttl_seconds = self.default_ttls.get(data_type, 3600)
        
        # Store in specified levels
        if CacheLevel.MEMORY in cache_levels:
            self.memory_cache.put(key, value, ttl_seconds, tags)
        
        if CacheLevel.DISK in cache_levels:
            self.disk_cache.put(key, value, ttl_seconds, tags)
        
        self._record_access(key)
    
    async def remove(self, key: str) -> bool:
        """Remove from all cache levels."""
        memory_removed = self.memory_cache.remove(key)
        disk_removed = self.disk_cache.remove(key)
        
        # Execute invalidation callbacks
        for callback in self.invalidation_callbacks.get(key, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(key)
                else:
                    callback(key)
            except Exception as e:
                logger.error(f"Error in invalidation callback for {key}: {e}")
        
        return memory_removed or disk_removed
    
    async def clear(self, pattern: Optional[str] = None, tag_filter: Optional[str] = None):
        """Clear cache entries matching pattern or tag."""
        if pattern:
            # Clear by key pattern (simplified - just check if pattern in key)
            memory_keys = list(self.memory_cache.entries.keys())
            for key in memory_keys:
                if pattern in key:
                    await self.remove(key)
        elif tag_filter:
            self.memory_cache.clear(tag_filter)
            self.disk_cache.clear(tag_filter)
        else:
            self.memory_cache.clear()
            self.disk_cache.clear()
    
    def _record_access(self, key: str):
        """Record access for intelligence gathering."""
        current_time = datetime.now()
        self.access_patterns[key].append(current_time)
        
        # Limit access history
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-50:]
        
        # Update popularity score
        recent_accesses = [
            t for t in self.access_patterns[key]
            if current_time - t < timedelta(hours=1)
        ]
        self.popular_keys[key] = len(recent_accesses)
    
    def _extract_data_type(self, key: str) -> str:
        """Extract data type from cache key."""
        # Simple heuristic based on key patterns
        if "product:" in key:
            return "product"
        elif "search:" in key:
            return "search_results"
        elif "store:" in key:
            return "store_status"
        elif "quality:" in key:
            return "quality_metrics"
        elif "optimization:" in key:
            return "optimization_results"
        else:
            return "general"
    
    async def warm_cache(self, key_patterns: List[str], data_loader: Callable[[str], Any]):
        """Proactive cache warming for popular or critical data."""
        if not self.enable_cache_warming:
            return
        
        logger.info(f"Starting cache warming for {len(key_patterns)} patterns")
        
        async def warm_pattern(pattern: str):
            """Warm cache for a specific pattern."""
            try:
                # Load data
                data = data_loader(pattern)
                if data is not None:
                    await self.put(
                        pattern,
                        data,
                        tags={"cache_warmed": "true", "pattern": pattern}
                    )
                    logger.debug(f"Warmed cache for pattern: {pattern}")
            except Exception as e:
                logger.error(f"Error warming cache for pattern {pattern}: {e}")
        
        # Execute warming tasks concurrently
        warming_tasks = [warm_pattern(pattern) for pattern in key_patterns]
        await asyncio.gather(*warming_tasks, return_exceptions=True)
        
        logger.info(f"Cache warming completed for {len(key_patterns)} patterns")
    
    def add_invalidation_callback(self, key: str, callback: Callable):
        """Add callback to be executed when key is invalidated."""
        self.invalidation_callbacks[key].append(callback)
    
    async def intelligent_prefetch(self, store_id: str, query: str):
        """Intelligently prefetch related data based on access patterns."""
        # Analyze access patterns to predict what might be needed
        related_keys = []
        
        # Find related queries for the same store
        for key in self.access_patterns:
            if f"store:{store_id}" in key and key != f"search:{store_id}:{query}":
                related_keys.append(key)
        
        # Prefetch popular related items
        popular_related = sorted(
            related_keys,
            key=lambda k: self.popular_keys.get(k, 0),
            reverse=True
        )[:3]  # Top 3 related items
        
        for key in popular_related:
            if await self.get(key) is None:
                # Item not in cache, could trigger prefetch
                logger.debug(f"Could prefetch related item: {key}")
    
    async def analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache performance and provide recommendations."""
        memory_stats = self.memory_cache.get_statistics()
        disk_stats = self.disk_cache.get_statistics()
        
        # Access pattern analysis
        total_accesses = sum(len(accesses) for accesses in self.access_patterns.values())
        unique_keys = len(self.access_patterns)
        
        # Popular keys analysis
        top_keys = sorted(
            self.popular_keys.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Cache efficiency metrics
        total_hit_rate = 0
        if memory_stats["hits"] + memory_stats["misses"] > 0:
            total_hit_rate = memory_stats["hit_rate"]
        
        # Recommendations
        recommendations = []
        
        if memory_stats["hit_rate"] < 80:
            recommendations.append("Consider increasing memory cache size")
        
        if memory_stats["utilization"] > 90:
            recommendations.append("Memory cache is near capacity - monitor for evictions")
        
        if disk_stats["utilization"] > 80:
            recommendations.append("Disk cache is filling up - consider cleanup or expansion")
        
        if len(top_keys) > 0 and top_keys[0][1] > 10:
            recommendations.append(f"Consider cache warming for popular key: {top_keys[0][0]}")
        
        return {
            "memory_cache": memory_stats,
            "disk_cache": disk_stats,
            "access_patterns": {
                "total_accesses": total_accesses,
                "unique_keys": unique_keys,
                "avg_accesses_per_key": total_accesses / max(unique_keys, 1)
            },
            "popular_keys": dict(top_keys),
            "performance_metrics": {
                "overall_hit_rate": total_hit_rate,
                "cache_efficiency_score": min(100, total_hit_rate * (100 - memory_stats["utilization"]) / 100)
            },
            "recommendations": recommendations
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        memory_stats = self.memory_cache.get_statistics()
        disk_stats = self.disk_cache.get_statistics()
        
        # Health indicators
        memory_healthy = memory_stats["hit_rate"] > 70 and memory_stats["utilization"] < 95
        disk_healthy = disk_stats["utilization"] < 90
        
        overall_health = "healthy" if memory_healthy and disk_healthy else "degraded"
        if memory_stats["hit_rate"] < 50 or disk_stats["utilization"] > 95:
            overall_health = "unhealthy"
        
        return {
            "status": overall_health,
            "memory_cache_healthy": memory_healthy,
            "disk_cache_healthy": disk_healthy,
            "memory_hit_rate": memory_stats["hit_rate"],
            "memory_utilization": memory_stats["utilization"],
            "disk_utilization": disk_stats["utilization"],
            "timestamp": datetime.now().isoformat()
        }


# Global instance
cache_manager = CacheManager()