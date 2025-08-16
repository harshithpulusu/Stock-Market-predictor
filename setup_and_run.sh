#!/bin/bash
# Stock Market Predictor App Setup & Launch Guide

echo "🚀 AI Stock Market Predictor - Setup Guide"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "stock_predictor_app.py" ]; then
    echo "❌ Error: stock_predictor_app.py not found!"
    echo "📁 Please navigate to the Stock-Market-predictor directory first"
    echo "   cd '/Users/harshithpulusu/Documents/Hobby Projects/Stock-Market-predictor'"
    exit 1
fi

echo "✅ Found stock_predictor_app.py"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "📥 Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python 3 is installed: $(python3 --version)"
echo ""

# Check if required packages are installed
echo "🔍 Checking required packages..."

required_packages=("streamlit" "plotly" "yfinance" "pandas" "numpy" "scikit-learn" "matplotlib" "seaborn")

for package in "${required_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✅ $package is installed"
    else
        echo "❌ $package is missing"
        echo "📦 Installing $package..."
        pip3 install $package
    fi
done

echo ""
echo "🎉 All dependencies are ready!"
echo ""

# Launch options
echo "🚀 Launch Options:"
echo "=================="
echo ""
echo "Option 1 - Quick Launch (Recommended):"
echo "   python3 launch_app.py"
echo ""
echo "Option 2 - Direct Streamlit:"
echo "   streamlit run stock_predictor_app.py"
echo ""
echo "Option 3 - With Custom Settings:"
echo "   streamlit run stock_predictor_app.py --server.port 8502"
echo ""

# Launch the app
echo "🔥 Launching the app now..."
echo "📱 The app will open at http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Run the app
python3 -m streamlit run stock_predictor_app.py
