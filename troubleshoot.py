#!/usr/bin/env python3
"""
Stock Predictor Troubleshooter
Diagnoses and fixes common issues
"""

import subprocess
import sys
import os
import socket
import importlib

def check_port(port):
    """Check if a port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def check_package(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def main():
    print("🔧 Stock Market Predictor - Troubleshooter")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    else:
        print("✅ Python version is compatible")
    
    print()
    
    # Check required packages
    print("📦 Checking Required Packages:")
    packages = {
        'streamlit': 'Streamlit web framework',
        'plotly': 'Interactive charts',
        'yfinance': 'Stock data',
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization'
    }
    
    missing_packages = []
    
    for package, description in packages.items():
        if check_package(package):
            print(f"✅ {package:<12} - {description}")
        else:
            print(f"❌ {package:<12} - {description} (MISSING)")
            missing_packages.append(package)
    
    print()
    
    # Check port availability
    print("🌐 Checking Ports:")
    ports_to_check = [8501, 8502, 8503]
    
    for port in ports_to_check:
        if check_port(port):
            print(f"🔴 Port {port} - In use")
        else:
            print(f"✅ Port {port} - Available")
    
    print()
    
    # Check if files exist
    print("📁 Checking Project Files:")
    required_files = [
        'stock_predictor_app.py',
        'simple_launcher.py',
        'requirements.txt',
        'README.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (MISSING)")
    
    print()
    print("🛠️  Solutions:")
    print("=" * 20)
    
    if missing_packages:
        print("📦 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        print()
    
    if check_port(8501):
        print("🌐 Port 8501 is busy. Try these:")
        print("   1. Kill existing Streamlit: pkill -f streamlit")
        print("   2. Use different port: streamlit run stock_predictor_app.py --server.port 8502")
        print()
    
    print("🚀 Launch Commands:")
    print("   Option 1: python3 simple_launcher.py")
    print("   Option 2: streamlit run stock_predictor_app.py --server.headless true")
    print("   Option 3: streamlit run stock_predictor_app.py --server.port 8502")
    
    print()
    print("🌐 After launching, open in browser:")
    print("   http://localhost:8501  (or 8502 if using different port)")

if __name__ == "__main__":
    main()
