import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

class AIStockPredictor:
    def __init__(self):
        self.lstm_model = None
        self.scaler = MinMaxScaler()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # You would need to sign up for a free API key at newsapi.org
        self.newsapi = NewsApiClient(api_key='YOUR_NEWS_API_KEY')  # Replace with your key
        self.sequence_length = 60  # Number of time steps to look back
        
    def get_stock_data(self, symbol, period='2y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            print(f"Successfully downloaded {len(df)} days of {symbol} data")
            return df
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

    def get_news_sentiment(self, symbol, days=30):
        """Get news sentiment for the stock"""
        try:
            # Get news articles
            news = self.newsapi.get_everything(
                q=symbol,
                language='en',
                from_param=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                to=datetime.now().strftime('%Y-%m-%d'),
                sort_by='relevancy'
            )
            
            sentiments = []
            for article in news['articles']:
                # Analyze sentiment of headline and description
                headline_sentiment = self.sentiment_analyzer.polarity_scores(article['title'])
                desc_sentiment = self.sentiment_analyzer.polarity_scores(article['description'] or '')
                
                # Combine sentiments (giving more weight to headlines)
                combined_sentiment = (headline_sentiment['compound'] * 0.7 +
                                   desc_sentiment['compound'] * 0.3)
                sentiments.append(combined_sentiment)
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            sentiment_std = np.std(sentiments) if sentiments else 0
            
            return avg_sentiment, sentiment_std
            
        except Exception as e:
            print(f"Error getting news sentiment: {e}")
            return 0, 0

    def prepare_data(self, data):
        """Prepare data for LSTM model"""
        df = data.copy()
        
        # Technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Drop missing values
        df = df.dropna()
        
        # Select features
        features = ['Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD',
                   'Volume_MA5', 'Price_Change', 'Volume_Change']
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(df[features])
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict next day's closing price
            
        return np.array(X), np.array(y), df

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd

    def create_lstm_model(self, input_shape):
        """Create and compile LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        
        return model

    def train_model(self, X, y):
        """Train the LSTM model"""
        # Split data into training and validation sets
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        # Create and train the model
        self.lstm_model = self.create_lstm_model(X_train[0].shape)
        
        print("\nTraining LSTM model...")
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        # Calculate and print performance metrics
        train_loss = self.lstm_model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.lstm_model.evaluate(X_val, y_val, verbose=0)
        
        print(f"Training MAE: ${train_loss[1]:.2f}")
        print(f"Validation MAE: ${val_loss[1]:.2f}")
        
        return X_val, y_val

    def predict_future(self, X, symbol, days=30):
        """Predict future stock prices using LSTM and sentiment analysis"""
        if self.lstm_model is None:
            raise ValueError("Model not trained yet!")
        
        # Get the last sequence
        last_sequence = X[-1:]
        
        # Get sentiment analysis
        sentiment, sentiment_std = self.get_news_sentiment(symbol)
        print(f"\nNews Sentiment Analysis:")
        print(f"Average Sentiment: {sentiment:.2f} (-1 to 1, where 1 is most positive)")
        print(f"Sentiment Volatility: {sentiment_std:.2f}")
        
        # Make predictions
        predictions = []
        prediction_dates = []
        current_sequence = last_sequence[0]
        
        current_date = datetime.now()
        for i in range(days):
            # Skip weekends
            while current_date.weekday() > 4:
                current_date += timedelta(days=1)
            
            # Predict next day
            current_sequence_reshaped = current_sequence.reshape(1, self.sequence_length, -1)
            next_pred = self.lstm_model.predict(current_sequence_reshaped, verbose=0)[0]
            
            # Adjust prediction based on sentiment
            sentiment_adjustment = 1 + (sentiment * 0.01)  # Adjust up to ±1%
            next_pred = next_pred * sentiment_adjustment
            
            # Add some randomness based on historical volatility
            volatility = np.std(X[-30:]) * np.sqrt(252)  # Annualized volatility
            random_factor = np.random.normal(0, volatility * 0.1)
            next_pred = next_pred * (1 + random_factor)
            
            predictions.append(next_pred)
            prediction_dates.append(current_date)
            
            # Update the sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_pred
            
            current_date += timedelta(days=1)
        
        # Inverse transform predictions to get actual prices
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(
            np.hstack([predictions, np.zeros((len(predictions), X.shape[2]-1))])
        )[:, 0]
        
        return prediction_dates, predictions

    def plot_predictions(self, dates, predictions, symbol, current_price):
        """Plot the predicted stock prices with confidence intervals"""
        plt.figure(figsize=(15, 8))
        
        # Calculate confidence intervals
        volatility = np.std(predictions) * np.sqrt(252)
        upper_bound = predictions * (1 + volatility)
        lower_bound = predictions * (1 - volatility)
        
        # Plot confidence intervals
        plt.fill_between(dates, lower_bound, upper_bound,
                        alpha=0.2, color='gray',
                        label='Confidence Interval (±1σ)')
        
        # Plot predictions
        plt.plot(dates, predictions, 'r-',
                label='AI Predicted Price', linewidth=2)
        
        # Add current price point
        plt.scatter([dates[0]], [current_price], color='blue',
                   s=100, label='Current Price', zorder=5)
        
        # Formatting
        plt.title(f'{symbol} Stock Price Predictions with AI\n'
                 f'Next {len(dates)} Trading Days',
                 fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        
        # Add price labels
        plt.text(dates[0], current_price,
                f'${current_price:.2f}',
                horizontalalignment='right',
                verticalalignment='bottom')
        plt.text(dates[-1], predictions[-1],
                f'${predictions[-1]:.2f}',
                horizontalalignment='left',
                verticalalignment='bottom')
        
        # Calculate and display predicted change
        pct_change = ((predictions[-1] - current_price) / current_price) * 100
        color = 'green' if pct_change >= 0 else 'red'
        plt.figtext(0.02, 0.02,
                   f'Predicted Change: {pct_change:+.2f}%',
                   color=color, fontsize=10)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    # Create predictor instance
    predictor = AIStockPredictor()
    
    # Get stock symbol from user
    symbol = input("Enter stock symbol (e.g., AAPL for Apple): ").upper()
    
    # Get stock data
    print(f"\nDownloading {symbol} stock data...")
    data = predictor.get_stock_data(symbol)
    
    if data is not None:
        # Prepare data and train model
        print("\nPreparing data and training AI model...")
        X, y, processed_data = predictor.prepare_data(data)
        X_val, y_val = predictor.train_model(X, y)
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        print(f"\nCurrent {symbol} price: ${current_price:.2f}")
        
        # Make predictions
        dates, predictions = predictor.predict_future(X, symbol)
        
        # Print predictions
        print("\nAI-Powered Predictions:")
        print("\nNext Week:")
        for i in range(5):  # Show 5 trading days
            print(f"{dates[i].strftime('%Y-%m-%d')}: ${predictions[i]:.2f}")
        
        print("\nMonth-End Prediction:")
        print(f"{dates[-1].strftime('%Y-%m-%d')}: ${predictions[-1]:.2f}")
        
        # Plot results
        predictor.plot_predictions(dates, predictions, symbol, current_price)

if __name__ == "__main__":
    main()
