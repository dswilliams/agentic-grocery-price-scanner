"""
Evaluation Framework

Comprehensive quality monitoring, regression testing, and continuous improvement
system for the Agentic Grocery Price Scanner.
"""

from .quality_monitor import QualityMonitor, QualityReport
from .regression_tester import RegressionTester, RegressionTestSuite
from .ml_model_evaluator import MLModelEvaluator, ModelEvaluationResult
from .business_metrics_validator import BusinessMetricsValidator, BusinessValidationReport
from .continuous_improvement import ContinuousImprovementPipeline, ImprovementPlan
from .reporting_system import ComprehensiveReportingSystem, ReportType, ReportPriority
from .golden_dataset import GoldenDatasetManager, GoldenMatch

__version__ = "1.0.0"
__author__ = "Agentic Grocery Price Scanner Team"

__all__ = [
    'QualityMonitor',
    'QualityReport', 
    'RegressionTester',
    'RegressionTestSuite',
    'MLModelEvaluator',
    'ModelEvaluationResult',
    'BusinessMetricsValidator',
    'BusinessValidationReport',
    'ContinuousImprovementPipeline',
    'ImprovementPlan',
    'ComprehensiveReportingSystem',
    'ReportType',
    'ReportPriority',
    'GoldenDatasetManager',
    'GoldenMatch'
]