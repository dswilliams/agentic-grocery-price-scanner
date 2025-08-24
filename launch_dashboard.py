#!/usr/bin/env python3
"""
Launch script for the Agentic Grocery Scanner Streamlit Dashboard.
"""

import sys
import os
import subprocess

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Dashboard path
    dashboard_path = os.path.join(current_dir, "dashboard", "streamlit_app.py")
    
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard not found at: {dashboard_path}")
        return 1
    
    print("🚀 Launching Agentic Grocery Scanner Dashboard...")
    print(f"📁 Dashboard path: {dashboard_path}")
    print("🌐 Opening in browser...")
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Dashboard shutdown requested")
        return 0
    
    return 0

if __name__ == "__main__":
    sys.exit(launch_dashboard())