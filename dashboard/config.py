"""
Dashboard configuration and utilities.
"""

import streamlit as st
from typing import Dict, Any, List
from datetime import datetime
import json
import os

class DashboardConfig:
    """Configuration class for the dashboard."""
    
    # Dashboard settings
    PAGE_TITLE = "🛒 Agentic Grocery Scanner Dashboard"
    PAGE_ICON = "🛒"
    LAYOUT = "wide"
    
    # Colors and themes
    COLORS = {
        'primary': '#1f77b4',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8',
        'secondary': '#6c757d'
    }
    
    # Pipeline stages
    PIPELINE_STAGES = [
        ("🎯 Initialize", "System setup and configuration"),
        ("🥘 Extract Ingredients", "Parse and normalize recipe ingredients"), 
        ("🕷️ Parallel Scraping", "Multi-store concurrent product collection"),
        ("📦 Aggregate Products", "Normalize and deduplicate product data"),
        ("🎯 Parallel Matching", "Semantic ingredient-to-product matching"),
        ("🔗 Aggregate Matches", "Confidence-weighted match consolidation"),
        ("⚖️ Optimize Shopping", "Multi-store trip optimization"),
        ("📊 Strategy Analysis", "Compare 6 optimization strategies"),
        ("✅ Finalize Results", "Generate shopping recommendations"),
        ("📈 Analytics", "Performance tracking and metrics"),
        ("🛡️ Error Recovery", "Graceful degradation and retry logic")
    ]
    
    # Store configurations
    STORES = {
        'metro_ca': {
            'name': 'Metro',
            'color': '#e41e26',
            'icon': '🏪'
        },
        'walmart_ca': {
            'name': 'Walmart',
            'color': '#004c91',
            'icon': '🛒'
        },
        'freshco_com': {
            'name': 'FreshCo',
            'color': '#00a651',
            'icon': '🍎'
        }
    }
    
    # Optimization strategies
    OPTIMIZATION_STRATEGIES = {
        'cost_only': {
            'name': 'Cost Only',
            'description': 'Minimize total cost regardless of convenience',
            'icon': '💰'
        },
        'convenience': {
            'name': 'Convenience',
            'description': 'Minimize travel time and store visits',
            'icon': '⏰'
        },
        'balanced': {
            'name': 'Balanced',
            'description': 'Balance cost, convenience, and quality',
            'icon': '⚖️'
        },
        'quality_first': {
            'name': 'Quality First',
            'description': 'Prioritize product quality and freshness',
            'icon': '⭐'
        },
        'time_efficient': {
            'name': 'Time Efficient',
            'description': 'Minimize total shopping time',
            'icon': '🚀'
        },
        'adaptive': {
            'name': 'Adaptive',
            'description': 'AI-powered strategy selection',
            'icon': '🤖'
        }
    }

def init_session_state():
    """Initialize session state variables."""
    
    # Dashboard state
    if 'dashboard_state' not in st.session_state:
        from dashboard.streamlit_app import DashboardState
        st.session_state.dashboard_state = DashboardState()
    
    # Execution parameters
    if 'execution_params' not in st.session_state:
        st.session_state.execution_params = {}
    
    # Execution state
    if 'execution_state' not in st.session_state:
        st.session_state.execution_state = {
            'running': False,
            'current_stage': None,
            'progress': 0,
            'metrics': {},
            'results': None,
            'logs': []
        }
    
    # Theme
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

def format_currency(amount: float) -> str:
    """Format currency with proper symbol and decimals."""
    return f"${amount:.2f}"

def format_percentage(value: float) -> str:
    """Format percentage with proper symbol."""
    return f"{value:.1%}"

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence score."""
    if confidence >= 0.9:
        return "🟢"
    elif confidence >= 0.8:
        return "🟡"
    elif confidence >= 0.7:
        return "🟠"
    else:
        return "🔴"

def export_results_to_json(results: Dict[str, Any]) -> str:
    """Export results to JSON format."""
    return json.dumps(results, indent=2, default=str)

def save_execution_history(execution_data: Dict[str, Any]):
    """Save execution to history."""
    history_file = "dashboard/execution_history.json"
    
    # Load existing history
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except:
            history = []
    
    # Add new execution
    execution_data['timestamp'] = datetime.now().isoformat()
    history.append(execution_data)
    
    # Keep only last 100 executions
    history = history[-100:]
    
    # Save updated history
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, default=str)