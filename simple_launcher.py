#!/usr/bin/env python3
"""
Simple Stock Predictor Launcher
Bypasses Streamlit's initial setup prompts
"""

import subprocess
import sys
import os
import time

def main():
    print("🚀 AI Stock Market Predictor - Simple Launcher")
    print("=" * 50)
    print("🔧 Starting server...")
    print("📱 App will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "stock_predictor_app.py")
    
    try:
        # Set environment variable to skip Streamlit's first-run experience
        env = os.environ.copy()
        env['STREAMLIT_SERVER_HEADLESS'] = 'true'
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        # Run streamlit with headless mode
        cmd = [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.runOnSave", "true"
        ]
        
        print("⚡ Server starting...")
        time.sleep(1)
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped!")
        print("💡 Run this script again to restart")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check if port 8501 is available")
        print("3. Try running: streamlit run stock_predictor_app.py")

if __name__ == "__main__":
    main()
