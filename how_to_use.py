#!/usr/bin/env python3
"""
🚀 STOCK MARKET PREDICTOR - COMPLETE USAGE GUIDE
================================================================

This is your complete guide to running the AI Stock Market Predictor web app.
"""

import os
import subprocess
import sys

def print_banner():
    print("🚀 AI STOCK MARKET PREDICTOR")
    print("=" * 60)
    print("📈 Advanced AI-powered stock analysis and prediction")
    print("🤖 Machine learning ensemble models")
    print("📊 Interactive web interface with real-time charts")
    print("=" * 60)

def print_status():
    print("\n📋 CURRENT STATUS:")
    print("=" * 20)
    
    # Check if files exist
    required_files = [
        'stock_predictor_app.py',
        'simple_launcher.py', 
        'test_app.py',
        'requirements.txt'
    ]
    
    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (MISSING)")
            all_files_exist = False
    
    if all_files_exist:
        print("🎉 All required files are present!")
    else:
        print("⚠️  Some files are missing!")
        
    return all_files_exist

def print_launch_options():
    print("\n🚀 HOW TO START THE APP:")
    print("=" * 25)
    print()
    print("OPTION 1 - Simple Launcher (Recommended):")
    print("   python3 simple_launcher.py")
    print()
    print("OPTION 2 - Direct Streamlit:")
    print("   streamlit run stock_predictor_app.py --server.headless true")
    print()
    print("OPTION 3 - Custom Port:")
    print("   streamlit run stock_predictor_app.py --server.port 8502")
    print()

def print_app_features():
    print("✨ APP FEATURES:")
    print("=" * 15)
    print("📊 Real-time stock data fetching")
    print("🤖 AI ensemble predictions (Random Forest, Gradient Boosting, Linear Regression)")
    print("📈 Interactive Plotly charts with zoom/pan/hover")
    print("📉 Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)")
    print("💰 Investment signals (BUY/SELL/HOLD) with confidence levels")
    print("📅 Weekly price targets and predictions")
    print("💾 CSV export functionality")
    print("📱 Mobile-responsive design")
    print("🔍 Feature importance analysis")
    print("📊 Model performance metrics")

def print_usage_instructions():
    print("\n📱 HOW TO USE THE APP:")
    print("=" * 20)
    print("1. 🌐 Open http://localhost:8501 in your web browser")
    print("2. 💼 Enter a stock symbol (e.g., AAPL, TSLA, MSFT, GOOGL)")
    print("3. ⚙️  Choose analysis settings:")
    print("   - Data period (6 months to 5 years)")
    print("   - Prediction timeframe (7-60 days)")
    print("4. 🔮 Click 'Analyze Stock' button")
    print("5. 📊 View interactive results:")
    print("   - Price predictions and charts")
    print("   - Technical analysis indicators")
    print("   - AI model performance metrics")
    print("   - Investment recommendations")
    print("6. 💾 Download results as CSV if needed")

def print_troubleshooting():
    print("\n🛠️  TROUBLESHOOTING:")
    print("=" * 17)
    print("❌ If localhost refuses to connect:")
    print("   1. Make sure the app is running (check terminal output)")
    print("   2. Try: pkill -f streamlit (to stop existing processes)")
    print("   3. Restart with: python3 simple_launcher.py")
    print("   4. Check different port: http://localhost:8502")
    print()
    print("❌ If you get import errors:")
    print("   pip install streamlit plotly yfinance pandas numpy scikit-learn matplotlib seaborn")
    print()
    print("❌ If the app crashes:")
    print("   1. Check terminal for error messages")
    print("   2. Run: python3 test_app.py (to test core functionality)")
    print("   3. Restart the app with: python3 simple_launcher.py")

def print_examples():
    print("\n💡 EXAMPLE STOCKS TO TRY:")
    print("=" * 24)
    print("🍎 AAPL  - Apple Inc.")
    print("🚗 TSLA  - Tesla Inc.")
    print("💻 MSFT  - Microsoft Corporation")
    print("🌐 GOOGL - Alphabet Inc. (Google)")
    print("🛒 AMZN  - Amazon.com Inc.")
    print("🎯 META  - Meta Platforms Inc.")
    print("💰 BRK-B - Berkshire Hathaway")
    print("💳 V     - Visa Inc.")

def main():
    print_banner()
    
    files_ok = print_status()
    
    if files_ok:
        print_launch_options()
        print_app_features()
        print_usage_instructions()
        print_examples()
        print_troubleshooting()
        
        print("\n🎯 QUICK START:")
        print("=" * 13)
        print("1. Run: python3 simple_launcher.py")
        print("2. Open: http://localhost:8501")
        print("3. Enter stock symbol and click 'Analyze Stock'")
        print("4. Enjoy your AI-powered stock analysis! 📈")
        
    else:
        print("\n❌ Setup incomplete. Please ensure all files are present.")
    
    print("\n🎉 Happy Trading! Remember to always do your own research! 💼")

if __name__ == "__main__":
    main()
