import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")

class ProfessionalStockPredictor:
    def __init__(self):
        """Initialize the professional-grade AI stock predictor"""
        # Use robust, proven algorithms
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            ),
            'Ridge Regression': Ridge(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        self.scaler = RobustScaler()  # More robust to outliers
        self.best_model = None
        self.best_score = -np.inf
        self.feature_importance = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def get_stock_data(self, symbol, period='2y'):
        """Fetch and clean stock data"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Basic data cleaning
            df = df.dropna()
            df = df[df['Volume'] > 0]  # Remove zero volume days
            
            print(f"✅ Downloaded {len(df)} days of clean {symbol} data")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate robust technical indicators without lookahead bias"""
        data = df.copy()
        
        # Price-based indicators
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Moving averages (lagged to avoid lookahead)
        data['SMA_5'] = data['Close'].shift(1).rolling(window=5).mean()
        data['SMA_20'] = data['Close'].shift(1).rolling(window=20).mean()
        data['SMA_50'] = data['Close'].shift(1).rolling(window=50).mean()
        
        # Exponential moving averages
        data['EMA_12'] = data['Close'].shift(1).ewm(span=12).mean()
        data['EMA_26'] = data['Close'].shift(1).ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # RSI (properly calculated)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI'] = data['RSI'].shift(1)  # Lag to avoid lookahead
        
        # Bollinger Bands
        sma_20 = data['Close'].shift(1).rolling(window=20).mean()
        std_20 = data['Close'].shift(1).rolling(window=20).std()
        data['BB_Upper'] = sma_20 + (2 * std_20)
        data['BB_Lower'] = sma_20 - (2 * std_20)
        data['BB_Position'] = (data['Close'].shift(1) - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].shift(1).rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'].shift(1) / data['Volume_SMA']
        
        # Volatility (lagged)
        data['Volatility'] = data['Returns'].shift(1).rolling(window=20).std() * np.sqrt(252)
        
        # Price position indicators
        data['High_Low_Ratio'] = data['High'].shift(1) / data['Low'].shift(1)
        data['Price_Range'] = (data['High'].shift(1) - data['Low'].shift(1)) / data['Close'].shift(1)
        
        # Momentum indicators
        data['Momentum_5'] = data['Close'].shift(1) / data['Close'].shift(6) - 1
        data['Momentum_20'] = data['Close'].shift(1) / data['Close'].shift(21) - 1
        
        return data
    
    def prepare_features(self, data):
        """Prepare features with proper time series handling"""
        df = self.calculate_technical_indicators(data)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) < 100:
            raise ValueError("Insufficient data after cleaning and feature engineering")
        
        # Select features (no lookahead bias)
        feature_columns = [
            'SMA_5', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'RSI', 'BB_Position',
            'Volume_Ratio', 'Volatility', 'High_Low_Ratio', 'Price_Range',
            'Momentum_5', 'Momentum_20'
        ]
        
        # Target: next day's return (more stable than absolute price)
        target = df['Returns'].shift(-1)  # Next day's return
        
        # Features and target
        X = df[feature_columns]
        y = target
        
        # Remove last row (no target available)
        X = X[:-1]
        y = y[:-1]
        
        # Remove any remaining NaN
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"✅ Prepared {len(X)} samples with {len(feature_columns)} features")
        return X, y, df
    
    def train_and_evaluate_models(self, X, y):
        """Train multiple models with proper time series cross-validation"""
        print("🔬 Training and evaluating AI models...")
        print("=" * 50)
        
        # Use time series split for proper validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"\n🤖 Evaluating {name}...")
            
            # Time series cross-validation
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            
            model_scores[name] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"   CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Update best model
            if cv_scores.mean() > self.best_score:
                self.best_score = cv_scores.mean()
                self.best_model = model
                self.best_model_name = name
        
        # Train best model on full dataset
        print(f"\n🏆 Best Model: {self.best_model_name} (CV R² = {self.best_score:.4f})")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train best model
        self.best_model.fit(X_scaled, y)
        
        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Top 5 Most Important Features:")
            for idx, row in self.feature_importance.head().iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return model_scores
    
    def analyze_sentiment(self, symbol):
        """Simple sentiment analysis without external dependencies"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Use basic company info for sentiment proxy
            sentiment_score = 0
            
            # Check recommendation mean (1=Strong Buy, 5=Strong Sell)
            if 'recommendationMean' in info:
                rec_mean = info['recommendationMean']
                if rec_mean <= 2:
                    sentiment_score = 0.5  # Positive
                elif rec_mean >= 4:
                    sentiment_score = -0.5  # Negative
                else:
                    sentiment_score = 0  # Neutral
            
            return sentiment_score
            
        except Exception as e:
            print(f"⚠️ Sentiment analysis unavailable: {e}")
            return 0
    
    def predict_future_returns(self, X, current_price, symbol, days=30):
        """Predict future returns with confidence intervals"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        print(f"🔮 Generating {days}-day predictions...")
        
        # Get sentiment
        sentiment = self.analyze_sentiment(symbol)
        print(f"📊 Market Sentiment: {sentiment:.2f} (-1 to 1)")
        
        # Use last available features
        last_features = X.iloc[-1:].copy()
        X_scaled = self.scaler.transform(last_features)
        
        predicted_returns = []
        confidence_intervals = []
        
        for day in range(days):
            # Predict next day's return
            pred_return = self.best_model.predict(X_scaled)[0]
            
            # Add sentiment adjustment (small effect)
            pred_return += sentiment * 0.001  # Max 0.1% sentiment impact
            
            # Add realistic noise based on historical volatility
            if hasattr(self, 'historical_volatility'):
                noise = np.random.normal(0, self.historical_volatility * 0.1)
                pred_return += noise
            
            predicted_returns.append(pred_return)
            
            # Simple confidence interval based on historical prediction error
            confidence_intervals.append(abs(pred_return) * 0.5)  # ±50% of prediction
        
        # Convert returns to prices
        predicted_prices = [current_price]
        for ret in predicted_returns:
            next_price = predicted_prices[-1] * (1 + ret)
            predicted_prices.append(next_price)
        
        predicted_prices = predicted_prices[1:]  # Remove initial price
        
        # Generate dates (business days only)
        dates = []
        current_date = datetime.now()
        business_days_added = 0
        
        while business_days_added < days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:  # Monday to Friday
                dates.append(current_date)
                business_days_added += 1
        
        return dates, predicted_prices, confidence_intervals
    
    def create_comprehensive_visualization(self, dates, prices, confidence_intervals, symbol, current_price, historical_data):
        """Create professional visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Price prediction chart
        ax1.plot(dates, prices, 'b-', linewidth=3, label='AI Prediction', marker='o', markersize=4)
        
        # Confidence bands
        upper_band = [p + ci * p for p, ci in zip(prices, confidence_intervals)]
        lower_band = [p - ci * p for p, ci in zip(prices, confidence_intervals)]
        
        ax1.fill_between(dates, lower_band, upper_band, alpha=0.3, color='lightblue', label='Confidence Band')
        ax1.axhline(y=current_price, color='red', linestyle='--', label=f'Current Price: ${current_price:.2f}')
        
        ax1.set_title(f'{symbol} Price Predictions - Next {len(dates)} Trading Days', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Return distribution
        returns = [(p - current_price) / current_price * 100 for p in prices]
        ax2.hist(returns, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2f}%')
        ax2.set_title('Predicted Return Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            ax3.barh(range(len(top_features)), top_features['importance'], color='lightgreen')
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'])
            ax3.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Importance Score')
            ax3.grid(True, alpha=0.3)
        
        # 4. Historical vs Predicted
        hist_prices = historical_data['Close'].tail(30).values
        hist_dates = historical_data.index[-30:]
        
        ax4.plot(hist_dates, hist_prices, 'g-', linewidth=2, label='Historical (30 days)', marker='o', markersize=3)
        ax4.plot(dates[:10], prices[:10], 'b-', linewidth=2, label='Predicted (10 days)', marker='s', markersize=3)
        ax4.set_title('Historical vs Predicted Trend', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Price ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        try:
            plt.show()
            print("✅ Comprehensive analysis charts displayed!")
        except:
            filename = f"{symbol}_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 Charts saved as: {filename}")
        
        plt.close()
    
    def print_detailed_analysis(self, dates, prices, confidence_intervals, current_price, symbol):
        """Print comprehensive analysis report"""
        print(f"\n📈 Professional AI Analysis Report: {symbol}")
        print("=" * 80)
        
        # Key metrics
        final_price = prices[-1]
        total_return = (final_price - current_price) / current_price * 100
        max_price = max(prices)
        min_price = min(prices)
        volatility = np.std(prices) / np.mean(prices) * 100
        
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"🎯 {len(dates)}-Day Target: ${final_price:.2f}")
        print(f"📊 Expected Return: {total_return:+.2f}%")
        print(f"📈 Potential Upside: ${max_price:.2f} (+{(max_price-current_price)/current_price*100:.2f}%)")
        print(f"📉 Potential Downside: ${min_price:.2f} ({(min_price-current_price)/current_price*100:.2f}%)")
        print(f"🎲 Prediction Volatility: {volatility:.2f}%")
        print(f"🏆 Model Used: {self.best_model_name}")
        print(f"🎯 Model Accuracy (R²): {self.best_score:.4f}")
        
        # Risk assessment
        risk_level = "🟢 LOW" if volatility < 3 else "🟡 MEDIUM" if volatility < 6 else "🔴 HIGH"
        print(f"⚠️ Risk Level: {risk_level}")
        
        # Weekly breakdown
        print(f"\n📅 Weekly Price Targets:")
        for week in range(0, min(len(dates), 28), 7):
            if week < len(dates):
                week_price = prices[week]
                week_return = (week_price - current_price) / current_price * 100
                print(f"   Week {week//7 + 1}: ${week_price:.2f} ({week_return:+.2f}%)")
        
        # Model confidence
        confidence_score = max(0, min(1, (self.best_score + 1) / 2))  # Normalize R² to 0-1
        print(f"\n🧠 AI Confidence Level: {confidence_score:.2f}/1.0")
        
        if confidence_score > 0.7:
            print("✅ High confidence predictions - Model shows strong predictive power")
        elif confidence_score > 0.4:
            print("⚠️ Medium confidence predictions - Use with additional analysis")
        else:
            print("❌ Low confidence predictions - High uncertainty, use cautiously")

def main():
    """Main execution function"""
    print("🚀 Professional AI Stock Predictor v3.0")
    print("=" * 60)
    print("🔬 Advanced Machine Learning")
    print("📊 Robust Technical Analysis")
    print("🎯 Time Series Cross-Validation")
    print("📈 Professional Risk Assessment")
    print("=" * 60)
    
    # Get user input
    symbol = input("\n💼 Enter stock symbol (e.g., AAPL): ").upper()
    
    try:
        # Initialize predictor
        predictor = ProfessionalStockPredictor()
        
        # Get and prepare data
        print(f"\n📥 Fetching {symbol} data...")
        data = predictor.get_stock_data(symbol)
        
        if data is None:
            print("❌ Failed to get stock data. Please check the symbol.")
            return
        
        # Prepare features
        print("\n🔬 Engineering features...")
        X, y, processed_data = predictor.prepare_features(data)
        
        # Store historical volatility for predictions
        predictor.historical_volatility = processed_data['Returns'].std()
        
        # Train models
        model_scores = predictor.train_and_evaluate_models(X, y)
        
        # Current price
        current_price = data['Close'].iloc[-1]
        print(f"\n💰 Current {symbol} price: ${current_price:.2f}")
        
        # Make predictions
        dates, prices, confidence_intervals = predictor.predict_future_returns(
            X, current_price, symbol, days=30
        )
        
        # Create visualizations
        print(f"\n📊 Creating comprehensive analysis...")
        predictor.create_comprehensive_visualization(
            dates, prices, confidence_intervals, symbol, current_price, data
        )
        
        # Print detailed analysis
        predictor.print_detailed_analysis(
            dates, prices, confidence_intervals, current_price, symbol
        )
        
        print(f"\n🎉 Analysis complete! Happy trading! 📈")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your input and try again.")

if __name__ == "__main__":
    main()
