#!/usr/bin/env python3
"""
Stock Predictor App Launcher
Simple script to launch the Streamlit web application
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting AI Stock Market Predictor Web App...")
    print("=" * 50)
    print("📱 The app will open in your default web browser")
    print("🔗 Default URL: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "stock_predictor_app.py")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--theme.base", "light"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using AI Stock Market Predictor!")
        print("💡 Run this script again anytime to restart the app")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("💡 Make sure you have streamlit installed: pip install streamlit")

if __name__ == "__main__":
    main()
