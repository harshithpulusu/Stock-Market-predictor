import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class SimpleStockPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def get_stock_data(self, symbol, period='1y'):
        """Download stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            print(f"Downloaded {len(df)} days of {symbol} data")
            return df
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

    def prepare_features(self, data):
        """Create basic technical indicators for prediction"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # Price Changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Previous days' prices
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_Volume'] = df['Volume'].shift(1)
        
        # Drop any rows with missing data
        df = df.dropna()
        
        # Features for prediction
        features = ['Open', 'High', 'Low', 'Volume', 
                   'SMA_5', 'SMA_20', 'Price_Change', 
                   'Volume_Change', 'Prev_Close', 'Prev_Volume']
        
        X = df[features]
        y = df['Close']
        
        return X, y

    def train_model(self, X, y):
        """Train the prediction model"""
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model on all data
        self.model.fit(X_scaled, y)
        
        # Calculate accuracy on training data
        predictions = self.model.predict(X_scaled)
        accuracy = np.mean(np.abs(predictions - y) / y) * 100
        print(f"Average prediction error: {accuracy:.2f}%")

    def predict_future(self, X, days=30):
        """Predict future stock prices"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        
        # Get the last known price
        last_price = X_scaled[-1:]
        
        # Make predictions
        predictions = []
        dates = []
        
        current_date = datetime.now()
        for i in range(days):
            # Skip weekends
            while current_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
                current_date += timedelta(days=1)
            
            # Predict the next day
            pred = self.model.predict(last_price)[0]
            predictions.append(pred)
            dates.append(current_date)
            
            current_date += timedelta(days=1)
        
        return dates, predictions

    def plot_predictions(self, dates, predictions, symbol, current_price):
        """Plot the predicted stock prices"""
        plt.figure(figsize=(12, 6))
        
        # Plot predictions
        plt.plot(dates, predictions, 'r--', label='Predicted Price')
        
        # Add current price point
        plt.scatter([dates[0]], [current_price], color='blue', label='Current Price')
        
        plt.title(f'{symbol} Stock Price Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    # Get stock symbol from user
    symbol = input("Enter stock symbol (e.g., AAPL for Apple): ").upper()
    
    # Create predictor instance
    predictor = SimpleStockPredictor()
    
    # Get historical data
    print(f"\nDownloading {symbol} stock data...")
    data = predictor.get_stock_data(symbol)
    
    if data is not None:
        # Prepare features and train model
        print("\nPreparing data and training model...")
        X, y = predictor.prepare_features(data)
        predictor.train_model(X, y)
        
        # Get predictions for next month
        current_price = data['Close'][-1]
        print(f"\nCurrent {symbol} price: ${current_price:.2f}")
        
        # Predict next week and month
        dates_week, predictions_week = predictor.predict_future(X, days=7)
        dates_month, predictions_month = predictor.predict_future(X, days=30)
        
        # Print predictions
        print("\nPredicted prices for next week:")
        for date, price in zip(dates_week, predictions_week):
            print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
        
        print("\nPredicted prices for next month (showing weekly intervals):")
        for date, price in zip(dates_month[::5], predictions_month[::5]):
            print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
        
        # Plot the predictions
        predictor.plot_predictions(dates_month, predictions_month, symbol, current_price)

if __name__ == "__main__":
    main()
