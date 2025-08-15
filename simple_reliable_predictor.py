import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SimpleReliablePredictor:
    """A simple, reliable stock predictor focusing on proven techniques"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def get_data(self, symbol, period='1y'):
        """Get clean stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            data = data.dropna()
            print(f"📊 Downloaded {len(data)} days of {symbol} data")
            return data
        except Exception as e:
            print(f"❌ Error getting data: {e}")
            return None
    
    def create_features(self, data):
        """Create simple, robust features"""
        df = data.copy()
        
        # Simple moving averages
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        
        # Price ratios
        df['Price_MA5_Ratio'] = df['Close'] / df['MA5']
        df['Price_MA20_Ratio'] = df['Close'] / df['MA20']
        
        # Simple momentum
        df['Return_1d'] = df['Close'].pct_change(1)
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_20d'] = df['Close'].pct_change(20)
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Volatility
        df['Volatility'] = df['Return_1d'].rolling(20).std()
        
        # High-Low ratios
        df['HL_Ratio'] = df['High'] / df['Low']
        df['Close_HL_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return df
    
    def prepare_data(self, data):
        """Prepare training data"""
        df = self.create_features(data)
        df = df.dropna()
        
        features = [
            'MA5', 'MA20', 'Price_MA5_Ratio', 'Price_MA20_Ratio',
            'Return_1d', 'Return_5d', 'Return_20d',
            'Volume_Ratio', 'Volatility', 'HL_Ratio', 'Close_HL_Position'
        ]
        
        # Target: next day's price change percentage
        df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
        
        X = df[features][:-1]  # Remove last row (no target)
        y = df['Target'][:-1]  # Remove last row
        
        # Remove any remaining NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        return X, y, df
    
    def train(self, X, y):
        """Train the model with time series validation"""
        print("🤖 Training model...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(self.model, X, y, cv=tscv, scoring='r2')
        
        print(f"📊 Cross-validation R² score: {scores.mean():.4f} ± {scores.std():.4f}")
        
        # Train on full data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🔍 Top 5 important features:")
        for _, row in importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        return scores.mean()
    
    def predict_future(self, data, days=20):
        """Make future predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        current_price = data['Close'].iloc[-1]
        df = self.create_features(data)
        
        # Get last features
        features = [
            'MA5', 'MA20', 'Price_MA5_Ratio', 'Price_MA20_Ratio',
            'Return_1d', 'Return_5d', 'Return_20d',
            'Volume_Ratio', 'Volatility', 'HL_Ratio', 'Close_HL_Position'
        ]
        
        last_features = df[features].iloc[-1:].fillna(method='ffill')
        
        predictions = []
        prices = [current_price]
        
        for i in range(days):
            # Scale and predict
            X_scaled = self.scaler.transform(last_features)
            change = self.model.predict(X_scaled)[0]
            
            # Apply prediction with some bounds
            change = np.clip(change, -0.1, 0.1)  # Limit to ±10% per day
            
            next_price = prices[-1] * (1 + change)
            predictions.append(change)
            prices.append(next_price)
        
        return prices[1:], predictions  # Remove initial price
    
    def create_simple_chart(self, historical_data, predicted_prices, symbol):
        """Create a simple, clear chart"""
        # Get recent historical prices
        recent_prices = historical_data['Close'].tail(30)
        recent_dates = recent_prices.index
        
        # Create future dates (business days)
        last_date = recent_dates[-1]
        future_dates = []
        current_date = last_date
        
        for _ in range(len(predicted_prices)):
            current_date += timedelta(days=1)
            while current_date.weekday() > 4:  # Skip weekends
                current_date += timedelta(days=1)
            future_dates.append(current_date)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Main price chart
        ax1.plot(recent_dates, recent_prices, 'b-', linewidth=2, label='Historical', marker='o')
        ax1.plot(future_dates, predicted_prices, 'r-', linewidth=2, label='Predicted', marker='s')
        ax1.axvline(x=recent_dates[-1], color='gray', linestyle='--', alpha=0.7, label='Today')
        
        ax1.set_title(f'{symbol} Stock Price Prediction', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate and show key statistics
        current_price = recent_prices.iloc[-1]
        final_price = predicted_prices[-1]
        total_return = (final_price - current_price) / current_price * 100
        
        ax1.text(0.02, 0.98, f'Current: ${current_price:.2f}\nTarget: ${final_price:.2f}\nReturn: {total_return:+.1f}%',
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Daily returns chart
        daily_returns = [(predicted_prices[i] - (recent_prices.iloc[-1] if i == 0 else predicted_prices[i-1])) / 
                        (recent_prices.iloc[-1] if i == 0 else predicted_prices[i-1]) * 100 
                        for i in range(len(predicted_prices))]
        
        colors = ['green' if ret >= 0 else 'red' for ret in daily_returns]
        ax2.bar(range(len(daily_returns)), daily_returns, color=colors, alpha=0.7)
        ax2.set_title('Daily Predicted Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trading Days Ahead', fontsize=12)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        try:
            plt.show()
            print("✅ Chart displayed successfully!")
        except:
            filename = f"{symbol}_prediction_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 Chart saved as: {filename}")
        
        return future_dates
    
    def print_summary(self, symbol, current_price, predicted_prices, future_dates):
        """Print a clear summary"""
        print(f"\n📈 {symbol} Prediction Summary")
        print("=" * 50)
        
        final_price = predicted_prices[-1]
        total_return = (final_price - current_price) / current_price * 100
        
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"🎯 {len(predicted_prices)}-Day Target: ${final_price:.2f}")
        print(f"📊 Expected Return: {total_return:+.2f}%")
        print(f"📅 Target Date: {future_dates[-1].strftime('%Y-%m-%d')}")
        
        # Weekly targets
        print(f"\n📅 Weekly Targets:")
        for week in [4, 9, 14, 19]:  # Roughly weekly intervals
            if week < len(predicted_prices):
                week_price = predicted_prices[week]
                week_return = (week_price - current_price) / current_price * 100
                print(f"   {future_dates[week].strftime('%m/%d')}: ${week_price:.2f} ({week_return:+.1f}%)")
        
        # Risk assessment
        volatility = np.std(predicted_prices) / np.mean(predicted_prices) * 100
        if volatility < 2:
            risk = "🟢 LOW"
        elif volatility < 5:
            risk = "🟡 MEDIUM"
        else:
            risk = "🔴 HIGH"
        
        print(f"\n⚠️ Volatility Risk: {risk} ({volatility:.1f}%)")
        
        # Simple recommendation
        if total_return > 5:
            rec = "🚀 STRONG BUY"
        elif total_return > 2:
            rec = "📈 BUY"
        elif total_return > -2:
            rec = "➡️ HOLD"
        elif total_return > -5:
            rec = "📉 SELL"
        else:
            rec = "⚠️ STRONG SELL"
        
        print(f"🎯 AI Recommendation: {rec}")

def main():
    print("🤖 Simple & Reliable Stock Predictor")
    print("=" * 40)
    
    symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
    
    try:
        # Initialize
        predictor = SimpleReliablePredictor()
        
        # Get data
        data = predictor.get_data(symbol)
        if data is None:
            return
        
        # Prepare and train
        X, y, processed_data = predictor.prepare_data(data)
        score = predictor.train(X, y)
        
        if score < -0.5:
            print("⚠️ Warning: Model shows low predictive power. Use results cautiously.")
        
        # Make predictions
        current_price = data['Close'].iloc[-1]
        predicted_prices, _ = predictor.predict_future(data, days=20)
        
        # Create visualization
        future_dates = predictor.create_simple_chart(data, predicted_prices, symbol)
        
        # Print summary
        predictor.print_summary(symbol, current_price, predicted_prices, future_dates)
        
        print(f"\n✅ Analysis complete!")
        print("📝 Remember: This is for educational purposes. Always do your own research!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
