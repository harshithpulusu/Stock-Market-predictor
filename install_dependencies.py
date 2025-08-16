#!/usr/bin/env python3
"""
Dependency Installer for Stock Market Predictor
Run this if you need to install all required packages
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔧 Stock Market Predictor - Dependency Installer")
    print("=" * 50)
    
    # Required packages
    packages = [
        "streamlit>=1.48.0",
        "plotly>=6.3.0", 
        "yfinance>=0.2.18",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    print("📦 Installing required packages...")
    
    failed_packages = []
    
    for package in packages:
        package_name = package.split(">=")[0]
        print(f"   Installing {package_name}...")
        
        if install_package(package):
            print(f"   ✅ {package_name} installed successfully")
        else:
            print(f"   ❌ Failed to install {package_name}")
            failed_packages.append(package_name)
    
    print("\n" + "=" * 50)
    
    if failed_packages:
        print("❌ Some packages failed to install:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("\n💡 Try installing them manually:")
        print(f"   pip install {' '.join(failed_packages)}")
    else:
        print("🎉 All packages installed successfully!")
        print("\n🚀 You can now run the app:")
        print("   python3 launch_app.py")
        print("   or")
        print("   streamlit run stock_predictor_app.py")

if __name__ == "__main__":
    main()
