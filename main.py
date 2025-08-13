from data_fetcher import StockDataFetcher
from stock_predictor import StockMarketPredictor
from utils import plot_stock_prediction
import pandas as pd
from datetime import datetime

def main():
    # Get stock symbol from user
    symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
    
    # Initialize data fetcher and get data
    print(f"\nFetching data for {symbol}...")
    fetcher = StockDataFetcher()
    data = fetcher.fetch_stock_data(symbol, period='2y')
    
    if data is None:
        print("Failed to fetch stock data. Please try again.")
        return
        
    # Get company information
    company_info = fetcher.get_company_info(symbol)
    if company_info:
        print("\nCompany Information:")
        for key, value in company_info.items():
            print(f"{key}: {value}")
    
    # Initialize predictor
    predictor = StockMarketPredictor()
    
    # Prepare features
    print("\nPreparing features...")
    X, y = predictor.prepare_features(data)
    
    # Train the model
    print("\nTraining model...")
    X_test, y_test, y_pred = predictor.train(X, y)
    
    # Plot results
    print("\nGenerating prediction plot...")
    plot_stock_prediction(y_test, y_pred, data.iloc[-len(y_test):]['Date'],
                         f"{symbol} Stock Price Prediction")
    print("Plot saved as 'prediction_plot.png'")
    
    # Make prediction for next day
    last_data = X.iloc[-1:].copy()
    next_day_pred = predictor.predict(last_data)
    current_price = data['Close'].iloc[-1]
    
    print(f"\nCurrent stock price: ${current_price:.2f}")
    print(f"Predicted next day price: ${next_day_pred[0]:.2f}")
    print(f"Predicted change: {((next_day_pred[0] - current_price) / current_price * 100):.2f}%")

if __name__ == "__main__":
    main()
