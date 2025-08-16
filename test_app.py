#!/usr/bin/env python3
"""
Quick Test for Stock Predictor App
Tests core functionality without the web interface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from stock_predictor_app import StockPredictorApp
    import yfinance as yf
    import pandas as pd
    import numpy as np
    
    def test_basic_functionality():
        print("🧪 Testing Stock Predictor Core Functionality")
        print("=" * 50)
        
        # Initialize predictor
        predictor = StockPredictorApp()
        
        # Test data fetching
        print("📥 Testing data fetching...")
        data, stock_info, success = predictor.fetch_data("AAPL", "1y")
        
        if not success:
            print("❌ Data fetching failed")
            return False
            
        print(f"✅ Fetched {len(data)} days of data for AAPL")
        
        # Test feature creation
        print("🔬 Testing feature engineering...")
        enhanced_data = predictor.create_technical_features(data)
        
        if 'Returns' not in enhanced_data.columns:
            print("❌ Returns column not created")
            return False
            
        print(f"✅ Created {len(predictor.feature_columns)} features")
        
        # Test model training
        print("🤖 Testing model training...")
        try:
            model_results, best_name, best_score = predictor.train_model(enhanced_data)
            print(f"✅ Best model: {best_name} (R² = {best_score:.4f})")
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            return False
        
        # Test predictions
        print("🔮 Testing predictions...")
        try:
            predictions = predictor.make_predictions(enhanced_data, 7)
            print(f"✅ Generated {len(predictions)} predictions")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return False
        
        print("\n🎉 All tests passed! The app should work correctly.")
        return True
    
    def main():
        if test_basic_functionality():
            print("\n🚀 Ready to use the web app!")
            print("📱 Open: http://localhost:8501")
            print("💡 Run: python3 simple_launcher.py")
        else:
            print("\n❌ There are issues that need to be fixed.")
            
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all dependencies are installed:")
    print("   pip install streamlit plotly yfinance pandas numpy scikit-learn matplotlib seaborn")

if __name__ == "__main__":
    main()
