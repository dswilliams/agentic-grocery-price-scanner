"""
Advanced error recovery system with checkpointing, dead letter queue,
and intelligent workflow restart capabilities for production reliability.
"""

import asyncio
import logging
import json
import pickle
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, TypeVar, Union
import tempfile
import sqlite3
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RETRY = "retry"
    RESTART_FROM_CHECKPOINT = "restart_from_checkpoint"
    FALLBACK_STRATEGY = "fallback_strategy"
    MANUAL_INTERVENTION = "manual_intervention"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ABORT_WORKFLOW = "abort_workflow"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WorkflowCheckpoint:
    """Workflow execution checkpoint."""
    
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    stage: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Workflow state snapshot
    state_data: Dict[str, Any] = field(default_factory=dict)
    completed_stages: List[str] = field(default_factory=list)
    current_progress: float = 0.0
    
    # Metadata
    agent_states: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowCheckpoint':
        """Create checkpoint from dictionary."""
        data = data.copy()
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ErrorContext:
    """Context information about an error occurrence."""
    
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    stage: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Error details
    error_type: str = ""
    error_message: str = ""
    stack_trace: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    
    # Recovery information
    retry_count: int = 0
    max_retries: int = 3
    recovery_actions_attempted: List[RecoveryAction] = field(default_factory=list)
    
    # Context data
    execution_context: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeadLetterItem:
    """Item in the dead letter queue."""
    
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    original_data: Dict[str, Any] = field(default_factory=dict)
    error_context: ErrorContext = field(default_factory=ErrorContext)
    timestamp: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    requires_manual_review: bool = True


class CheckpointManager:
    """Manages workflow checkpoints for recovery."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or tempfile.gettempdir()) / "grocery_scanner_checkpoints"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_dir / "checkpoints.db"
        self.lock = threading.RLock()
        
        self._init_database()
        logger.info(f"Initialized CheckpointManager at {self.storage_dir}")
    
    def _init_database(self):
        """Initialize checkpoint database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data_file TEXT NOT NULL,
                    progress REAL DEFAULT 0.0
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_workflow_id ON checkpoints(workflow_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON checkpoints(timestamp)')
    
    async def create_checkpoint(
        self,
        workflow_id: str,
        stage: str,
        state_data: Dict[str, Any],
        completed_stages: List[str],
        progress: float = 0.0,
        agent_states: Optional[Dict[str, Any]] = None,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a workflow checkpoint."""
        
        checkpoint = WorkflowCheckpoint(
            workflow_id=workflow_id,
            stage=stage,
            state_data=state_data,
            completed_stages=completed_stages,
            current_progress=progress,
            agent_states=agent_states or {},
            execution_context=execution_context or {}
        )
        
        with self.lock:
            # Save checkpoint data to file
            data_file = f"checkpoint_{checkpoint.checkpoint_id}.pkl"
            data_path = self.storage_dir / data_file
            
            with open(data_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # Store metadata in database
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO checkpoints 
                    (checkpoint_id, workflow_id, stage, timestamp, data_file, progress)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    checkpoint.checkpoint_id,
                    workflow_id,
                    stage,
                    checkpoint.timestamp.isoformat(),
                    data_file,
                    progress
                ))
        
        logger.info(f"Created checkpoint {checkpoint.checkpoint_id} for workflow {workflow_id} at stage {stage}")
        return checkpoint.checkpoint_id
    
    async def get_latest_checkpoint(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Get the latest checkpoint for a workflow."""
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM checkpoints 
                    WHERE workflow_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (workflow_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Load checkpoint data
                data_path = self.storage_dir / row['data_file']
                if not data_path.exists():
                    logger.error(f"Checkpoint data file not found: {data_path}")
                    return None
                
                try:
                    with open(data_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                    return checkpoint
                except Exception as e:
                    logger.error(f"Error loading checkpoint: {e}")
                    return None
    
    async def get_checkpoint_by_id(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Get a specific checkpoint by ID."""
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM checkpoints WHERE checkpoint_id = ?',
                    (checkpoint_id,)
                )
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                data_path = self.storage_dir / row['data_file']
                if not data_path.exists():
                    return None
                
                try:
                    with open(data_path, 'rb') as f:
                        checkpoint = pickle.load(f)
                    return checkpoint
                except Exception as e:
                    logger.error(f"Error loading checkpoint: {e}")
                    return None
    
    async def list_checkpoints(self, workflow_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a workflow."""
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT checkpoint_id, stage, timestamp, progress 
                    FROM checkpoints 
                    WHERE workflow_id = ? 
                    ORDER BY timestamp DESC
                ''', (workflow_id,))
                
                return [dict(row) for row in cursor.fetchall()]
    
    async def cleanup_old_checkpoints(self, retention_days: int = 7):
        """Clean up old checkpoints."""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get old checkpoint files
                cursor = conn.execute(
                    'SELECT data_file FROM checkpoints WHERE timestamp < ?',
                    (cutoff_time.isoformat(),)
                )
                
                old_files = [row[0] for row in cursor.fetchall()]
                
                # Delete files
                for filename in old_files:
                    file_path = self.storage_dir / filename
                    if file_path.exists():
                        file_path.unlink()
                
                # Delete database records
                conn.execute(
                    'DELETE FROM checkpoints WHERE timestamp < ?',
                    (cutoff_time.isoformat(),)
                )
                
                logger.info(f"Cleaned up {len(old_files)} old checkpoints")


class DeadLetterQueue:
    """Dead letter queue for failed operations requiring manual intervention."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or tempfile.gettempdir()) / "grocery_scanner_dlq"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_dir / "dead_letter_queue.db"
        self.lock = threading.RLock()
        
        self._init_database()
        logger.info(f"Initialized DeadLetterQueue at {self.storage_dir}")
    
    def _init_database(self):
        """Initialize dead letter queue database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS dead_letter_items (
                    item_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    data_file TEXT NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    requires_manual_review BOOLEAN DEFAULT TRUE,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_workflow_id ON dead_letter_items(workflow_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON dead_letter_items(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_resolved ON dead_letter_items(resolved)')
    
    async def add_item(
        self,
        workflow_id: str,
        data: Dict[str, Any],
        error_context: ErrorContext,
        requires_manual_review: bool = True
    ) -> str:
        """Add item to dead letter queue."""
        
        item = DeadLetterItem(
            workflow_id=workflow_id,
            original_data=data,
            error_context=error_context,
            requires_manual_review=requires_manual_review
        )
        
        with self.lock:
            # Save item data to file
            data_file = f"dlq_item_{item.item_id}.pkl"
            data_path = self.storage_dir / data_file
            
            with open(data_path, 'wb') as f:
                pickle.dump(item, f)
            
            # Store metadata in database
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO dead_letter_items 
                    (item_id, workflow_id, timestamp, data_file, requires_manual_review)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    item.item_id,
                    workflow_id,
                    item.timestamp.isoformat(),
                    data_file,
                    requires_manual_review
                ))
        
        logger.warning(f"Added item {item.item_id} to dead letter queue for workflow {workflow_id}")
        return item.item_id
    
    async def get_pending_items(self, limit: int = 50) -> List[DeadLetterItem]:
        """Get pending items from dead letter queue."""
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM dead_letter_items 
                    WHERE resolved = FALSE 
                    ORDER BY timestamp ASC 
                    LIMIT ?
                ''', (limit,))
                
                items = []
                for row in cursor.fetchall():
                    data_path = self.storage_dir / row['data_file']
                    if data_path.exists():
                        try:
                            with open(data_path, 'rb') as f:
                                item = pickle.load(f)
                            items.append(item)
                        except Exception as e:
                            logger.error(f"Error loading DLQ item: {e}")
                
                return items
    
    async def resolve_item(self, item_id: str, resolved: bool = True):
        """Mark item as resolved."""
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    'UPDATE dead_letter_items SET resolved = ? WHERE item_id = ?',
                    (resolved, item_id)
                )
        
        logger.info(f"Marked DLQ item {item_id} as {'resolved' if resolved else 'unresolved'}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get dead letter queue statistics."""
        with self.lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Total counts
                cursor = conn.execute('SELECT COUNT(*) FROM dead_letter_items')
                total_items = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(*) FROM dead_letter_items WHERE resolved = FALSE')
                pending_items = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT COUNT(*) FROM dead_letter_items WHERE requires_manual_review = TRUE AND resolved = FALSE')
                manual_review_needed = cursor.fetchone()[0]
                
                # Recent activity
                one_day_ago = (datetime.now() - timedelta(days=1)).isoformat()
                cursor = conn.execute('SELECT COUNT(*) FROM dead_letter_items WHERE timestamp > ?', (one_day_ago,))
                recent_items = cursor.fetchone()[0]
                
                return {
                    "total_items": total_items,
                    "pending_items": pending_items,
                    "manual_review_needed": manual_review_needed,
                    "recent_items_24h": recent_items,
                    "resolution_rate": ((total_items - pending_items) / max(total_items, 1)) * 100
                }


class ErrorRecoveryManager:
    """Main error recovery and workflow restart manager."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.checkpoint_manager = CheckpointManager(storage_dir)
        self.dead_letter_queue = DeadLetterQueue(storage_dir)
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {
            "timeout": self._handle_timeout_error,
            "connection": self._handle_connection_error,
            "parsing": self._handle_parsing_error,
            "rate_limit": self._handle_rate_limit_error,
            "data_quality": self._handle_data_quality_error,
            "resource_exhaustion": self._handle_resource_error
        }
        
        # Error classification patterns
        self.error_patterns = {
            "timeout": ["timeout", "timed out", "deadline exceeded"],
            "connection": ["connection", "network", "dns", "unreachable"],
            "parsing": ["parse", "json", "xml", "selector", "element not found"],
            "rate_limit": ["rate limit", "too many requests", "429", "throttle"],
            "data_quality": ["quality", "validation", "anomaly", "inconsistent"],
            "resource_exhaustion": ["memory", "disk space", "cpu", "resource"]
        }
        
        logger.info("Initialized ErrorRecoveryManager")
    
    async def handle_error(
        self,
        workflow_id: str,
        stage: str,
        error: Exception,
        execution_context: Dict[str, Any],
        state_data: Optional[Dict[str, Any]] = None
    ) -> RecoveryAction:
        """Handle an error and determine recovery action."""
        
        # Create error context
        error_context = ErrorContext(
            workflow_id=workflow_id,
            stage=stage,
            error_type=type(error).__name__,
            error_message=str(error),
            execution_context=execution_context
        )
        
        # Classify error
        error_category = self._classify_error(str(error))
        
        # Determine severity
        error_context.severity = self._determine_severity(error, error_category)
        
        logger.error(f"Handling {error_context.severity.value} error in {stage}: {error}")
        
        # Get recovery strategy
        strategy_func = self.recovery_strategies.get(error_category, self._handle_generic_error)
        
        try:
            recovery_action = await strategy_func(error_context, state_data)
            error_context.recovery_actions_attempted.append(recovery_action)
            
            # Log recovery decision
            logger.info(f"Recovery action for {workflow_id}: {recovery_action.value}")
            
            return recovery_action
            
        except Exception as recovery_error:
            logger.error(f"Error in recovery strategy: {recovery_error}")
            
            # Fallback to dead letter queue
            if state_data:
                await self.dead_letter_queue.add_item(
                    workflow_id,
                    state_data,
                    error_context
                )
            
            return RecoveryAction.MANUAL_INTERVENTION
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error into category."""
        error_lower = error_message.lower()
        
        for category, patterns in self.error_patterns.items():
            if any(pattern in error_lower for pattern in patterns):
                return category
        
        return "generic"
    
    def _determine_severity(self, error: Exception, category: str) -> ErrorSeverity:
        """Determine error severity."""
        if isinstance(error, (MemoryError, SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        
        if category in ["resource_exhaustion", "data_quality"]:
            return ErrorSeverity.HIGH
        
        if category in ["connection", "timeout"]:
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    async def _handle_timeout_error(
        self, 
        error_context: ErrorContext, 
        state_data: Optional[Dict[str, Any]]
    ) -> RecoveryAction:
        """Handle timeout errors."""
        if error_context.retry_count < 2:
            return RecoveryAction.RETRY
        else:
            return RecoveryAction.FALLBACK_STRATEGY
    
    async def _handle_connection_error(
        self,
        error_context: ErrorContext,
        state_data: Optional[Dict[str, Any]]
    ) -> RecoveryAction:
        """Handle connection errors."""
        if error_context.retry_count < 3:
            return RecoveryAction.RETRY
        else:
            return RecoveryAction.FALLBACK_STRATEGY
    
    async def _handle_parsing_error(
        self,
        error_context: ErrorContext,
        state_data: Optional[Dict[str, Any]]
    ) -> RecoveryAction:
        """Handle parsing errors."""
        return RecoveryAction.GRACEFUL_DEGRADATION
    
    async def _handle_rate_limit_error(
        self,
        error_context: ErrorContext,
        state_data: Optional[Dict[str, Any]]
    ) -> RecoveryAction:
        """Handle rate limiting errors."""
        return RecoveryAction.GRACEFUL_DEGRADATION  # Circuit breaker should handle this
    
    async def _handle_data_quality_error(
        self,
        error_context: ErrorContext,
        state_data: Optional[Dict[str, Any]]
    ) -> RecoveryAction:
        """Handle data quality errors."""
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.MANUAL_INTERVENTION
        else:
            return RecoveryAction.GRACEFUL_DEGRADATION
    
    async def _handle_resource_error(
        self,
        error_context: ErrorContext,
        state_data: Optional[Dict[str, Any]]
    ) -> RecoveryAction:
        """Handle resource exhaustion errors."""
        return RecoveryAction.RESTART_FROM_CHECKPOINT
    
    async def _handle_generic_error(
        self,
        error_context: ErrorContext,
        state_data: Optional[Dict[str, Any]]
    ) -> RecoveryAction:
        """Handle generic/unclassified errors."""
        if error_context.retry_count < 1:
            return RecoveryAction.RETRY
        else:
            return RecoveryAction.MANUAL_INTERVENTION
    
    async def restart_workflow_from_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[WorkflowCheckpoint]:
        """Restart workflow from a checkpoint."""
        
        if checkpoint_id:
            checkpoint = await self.checkpoint_manager.get_checkpoint_by_id(checkpoint_id)
        else:
            checkpoint = await self.checkpoint_manager.get_latest_checkpoint(workflow_id)
        
        if not checkpoint:
            logger.error(f"No checkpoint found for workflow {workflow_id}")
            return None
        
        logger.info(f"Restarting workflow {workflow_id} from checkpoint {checkpoint.checkpoint_id} at stage {checkpoint.stage}")
        
        return checkpoint
    
    async def create_workflow_checkpoint(
        self,
        workflow_id: str,
        stage: str,
        state_data: Dict[str, Any],
        completed_stages: List[str],
        progress: float = 0.0,
        agent_states: Optional[Dict[str, Any]] = None,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a workflow checkpoint."""
        return await self.checkpoint_manager.create_checkpoint(
            workflow_id,
            stage,
            state_data,
            completed_stages,
            progress,
            agent_states,
            execution_context
        )
    
    async def get_recovery_report(self) -> Dict[str, Any]:
        """Get comprehensive recovery system report."""
        dlq_stats = await self.dead_letter_queue.get_statistics()
        
        # Get checkpoint statistics
        checkpoint_stats = {
            "storage_directory": str(self.checkpoint_manager.storage_dir),
            "database_path": str(self.checkpoint_manager.db_path)
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "dead_letter_queue": dlq_stats,
            "checkpoints": checkpoint_stats,
            "recovery_strategies": list(self.recovery_strategies.keys()),
            "error_categories": list(self.error_patterns.keys())
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform recovery system health check."""
        dlq_stats = await self.dead_letter_queue.get_statistics()
        
        # Health indicators
        dlq_healthy = dlq_stats["manual_review_needed"] < 10
        checkpoint_storage_healthy = self.checkpoint_manager.storage_dir.exists()
        
        overall_health = "healthy" if dlq_healthy and checkpoint_storage_healthy else "degraded"
        
        if dlq_stats["manual_review_needed"] > 50:
            overall_health = "unhealthy"
        
        return {
            "status": overall_health,
            "dead_letter_queue_healthy": dlq_healthy,
            "checkpoint_storage_healthy": checkpoint_storage_healthy,
            "manual_review_items": dlq_stats["manual_review_needed"],
            "recent_failures_24h": dlq_stats["recent_items_24h"],
            "timestamp": datetime.now().isoformat()
        }


# Global instance
error_recovery_manager = ErrorRecoveryManager()