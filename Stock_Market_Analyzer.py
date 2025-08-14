import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class StockMarketAnalyzer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def get_stock_data(self, symbol, period='2y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            print(f"Successfully downloaded {symbol} stock data")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
            
    def prepare_features(self, data):
        """Prepare features for prediction"""
        df = data.copy()
        
        # Technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=5).std()
        
        # Trading ranges
        df['High_Low_Range'] = df['High'] - df['Low']
        df['Close_Open_Range'] = df['Close'] - df['Open']
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Features for prediction
        features = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'MA50',
                   'Price_Change', 'Price_Change_5', 'Volume_Change',
                   'Volatility', 'High_Low_Range', 'Close_Open_Range']
        
        X = df[features]
        y = df['Close']
        
        return X, y, df
        
    def train_model(self, X, y):
        """Train the prediction model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                           test_size=0.2,
                                                           shuffle=False)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_score = r2_score(y_train, train_pred)
        test_score = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        print(f"\nModel Performance:")
        print(f"Training R² Score: {train_score:.4f}")
        print(f"Testing R² Score: {test_score:.4f}")
        print(f"Test RMSE: ${test_rmse:.2f}")
        
        return X_test_scaled, y_test, test_pred
    
    def plot_predictions(self, y_true, y_pred, title="Stock Price Predictions"):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true.values, label='Actual', linewidth=2)
        plt.plot(y_true.index, y_pred, label='Predicted', linewidth=2, linestyle='--')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def predict_next_days(self, latest_data, days=5):
        """Predict stock prices for the next few days"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale the input data
        scaled_data = self.scaler.transform(latest_data)
        
        # Make prediction
        prediction = self.model.predict(scaled_data)
        return prediction[0]

def main():
    # Create analyzer instance
    analyzer = StockMarketAnalyzer()
    
    # Set stock symbol (Apple)
    symbol = 'AAPL'
    
    # Get stock data
    print(f"\nFetching {symbol} stock data...")
    data = analyzer.get_stock_data(symbol)
    
    if data is not None:
        # Prepare features
        print("\nPreparing features...")
        X, y, processed_data = analyzer.prepare_features(data)
        
        # Train model
        print("\nTraining model...")
        X_test_scaled, y_test, y_pred = analyzer.train_model(X, y)
        
        # Plot results
        analyzer.plot_predictions(y_test, y_pred, f"{symbol} Stock Price Predictions")
        
        # Predict next day's price
        latest_data = X.iloc[-1:].copy()
        next_day_price = analyzer.predict_next_days(latest_data)
        current_price = y.iloc[-1]
        
        print(f"\nPrediction Summary for {symbol}:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Next Price: ${next_day_price:.2f}")
        print(f"Predicted Change: {((next_day_price - current_price) / current_price * 100):.2f}%")

if __name__ == "__main__":
    main()
