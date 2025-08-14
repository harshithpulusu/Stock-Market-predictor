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

    def predict_future(self, X, y, days=30):
        """Predict future stock prices"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get the last known data point
        last_data = X.iloc[-1].copy()
        last_price = y.iloc[-1]
        
        # Make predictions
        predictions = [last_price]  # Start with the last known price
        dates = [datetime.now()]
        current_date = datetime.now()
        
        for i in range(days):
            # Skip weekends
            current_date += timedelta(days=1)
            while current_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
                current_date += timedelta(days=1)
            
            # Update the features for the next prediction
            last_data['Prev_Close'] = predictions[-1]
            last_data['Price_Change'] = (predictions[-1] - predictions[-2]) / predictions[-2] if len(predictions) > 1 else 0
            
            # Update moving averages
            if i < 5:
                last_data['SMA_5'] = np.mean(predictions[-5:] + [last_price] * (5 - len(predictions)))
            else:
                last_data['SMA_5'] = np.mean(predictions[-5:])
            
            if i < 20:
                last_data['SMA_20'] = np.mean(predictions[-20:] + [last_price] * (20 - len(predictions)))
            else:
                last_data['SMA_20'] = np.mean(predictions[-20:])
            
            # Scale and predict
            scaled_data = self.scaler.transform(last_data.values.reshape(1, -1))
            next_price = self.model.predict(scaled_data)[0]
            
            # Add some realistic volatility
            volatility = np.std(y.pct_change().dropna()) * np.sqrt(252)  # Annualized volatility
            daily_volatility = volatility / np.sqrt(252)
            random_factor = np.random.normal(0, daily_volatility)
            next_price *= (1 + random_factor)
            
            predictions.append(next_price)
            dates.append(current_date)
        
        # Remove the first element (current price) as it's just used for calculations
        return dates[1:], predictions[1:]

    def plot_predictions(self, dates, predictions, symbol, current_price):
        """Plot the predicted stock prices with confidence interval"""
        plt.figure(figsize=(12, 6))
        
        # Calculate confidence interval (±2 standard deviations)
        std_dev = np.std(predictions)
        upper_bound = np.array(predictions) + 2 * std_dev
        lower_bound = np.array(predictions) - 2 * std_dev
        
        # Plot confidence interval
        plt.fill_between(dates, lower_bound, upper_bound, 
                        alpha=0.2, color='gray', label='95% Confidence Interval')
        
        # Plot predictions
        plt.plot(dates, predictions, 'r-', label='Predicted Price', linewidth=2)
        
        # Add current price point
        plt.scatter([dates[0]], [current_price], color='blue', s=100,
                   label='Current Price', zorder=5)
        
        # Formatting
        plt.title(f'{symbol} Stock Price Predictions\nNext {len(dates)} Trading Days',
                 fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        
        # Add price labels
        plt.text(dates[0], current_price, f'${current_price:.2f}', 
                horizontalalignment='right', verticalalignment='bottom')
        plt.text(dates[-1], predictions[-1], f'${predictions[-1]:.2f}', 
                horizontalalignment='left', verticalalignment='bottom')
        
        # Calculate and display predicted change
        pct_change = ((predictions[-1] - current_price) / current_price) * 100
        color = 'green' if pct_change >= 0 else 'red'
        plt.figtext(0.02, 0.02, f'Predicted Change: {pct_change:+.2f}%',
                   color=color, fontsize=10)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
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
        current_price = data['Close'].iloc[-1]
        print(f"\nCurrent {symbol} price: ${current_price:.2f}")
        
        # Predict next week and month
        dates_week, predictions_week = predictor.predict_future(X, y, days=7)
        dates_month, predictions_month = predictor.predict_future(X, y, days=30)
        
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
