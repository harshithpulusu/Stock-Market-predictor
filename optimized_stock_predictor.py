#!/usr/bin/env python3
"""
Optimized Stock Market Predictor v5.0
Focus on proven techniques and realistic expectations
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class OptimizedStockPredictor:
    """
    An optimized stock predictor focusing on proven techniques
    and realistic performance expectations
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.data = None
        self.predictions = None
        self.stock_info = None
        
    def fetch_data(self, symbol, period="2y"):
        """Fetch stock data with error handling"""
        try:
            stock = yf.Ticker(symbol)
            self.data = stock.history(period=period)
            self.stock_info = stock.info
            
            if len(self.data) < 50:
                raise ValueError("Insufficient data")
                
            print(f"✅ Fetched {len(self.data)} days of data for {symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False
    
    def create_technical_features(self):
        """Create meaningful technical indicators"""
        df = self.data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Price_MA_5'] = df['Close'].rolling(5).mean()
        df['Price_MA_20'] = df['Close'].rolling(20).mean()
        df['Price_MA_50'] = df['Close'].rolling(50).mean()
        
        # Moving average ratios (normalized features)
        df['MA_Ratio_5_20'] = df['Price_MA_5'] / df['Price_MA_20']
        df['MA_Ratio_20_50'] = df['Price_MA_20'] / df['Price_MA_50']
        df['Price_to_MA20'] = df['Close'] / df['Price_MA_20']
        
        # Volatility features
        df['Volatility_5'] = df['Returns'].rolling(5).std()
        df['Volatility_20'] = df['Returns'].rolling(20).std()
        
        # Volume features
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Normalized'] = (df['RSI'] - 50) / 50  # Normalize to [-1, 1]
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = bb_middle + (bb_std * 2)
        df['BB_Lower'] = bb_middle - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Lagged features (to prevent data leakage)
        for lag in [1, 2, 3, 5]:
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
            df[f'Volume_Ratio_Lag_{lag}'] = df['Volume_Ratio'].shift(lag)
        
        # Target variable (next day's return)
        df['Target'] = df['Returns'].shift(-1)
        
        # Select final features (avoid future-looking data)
        self.feature_columns = [
            'MA_Ratio_5_20', 'MA_Ratio_20_50', 'Price_to_MA20',
            'Volatility_5', 'Volatility_20', 'Volume_Ratio',
            'RSI_Normalized', 'MACD', 'MACD_Histogram', 'BB_Position',
            'Returns_Lag_1', 'Returns_Lag_2', 'Returns_Lag_3', 'Returns_Lag_5',
            'Volume_Ratio_Lag_1', 'Volume_Ratio_Lag_2', 'Volume_Ratio_Lag_3'
        ]
        
        self.data = df
        print(f"✅ Created {len(self.feature_columns)} technical features")
        
    def prepare_data(self):
        """Prepare data for modeling"""
        # Remove rows with NaN values
        feature_data = self.data[self.feature_columns + ['Target']].dropna()
        
        # Separate features and target
        X = feature_data[self.feature_columns]
        y = feature_data['Target']
        
        print(f"✅ Prepared dataset: {len(X)} samples, {len(self.feature_columns)} features")
        return X, y
    
    def train_model(self, X, y):
        """Train and evaluate models using time series cross-validation"""
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'Linear Regression': LinearRegression()
        }
        
        best_score = -np.inf
        best_model = None
        best_name = ""
        
        print("\n🤖 Training and evaluating models...")
        print("="*50)
        
        for name, model in models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            print(f"📊 {name}:")
            print(f"   CV R² Score: {mean_score:.4f} (±{std_score:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        
        # Train the best model on full dataset
        print(f"\n🏆 Best Model: {best_name} (R² = {best_score:.4f})")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        best_model.fit(X_scaled, y)
        
        self.model = best_model
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Top 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return best_score
    
    def make_predictions(self, days=30):
        """Generate future predictions"""
        # Get the last complete row of features
        last_features = self.data[self.feature_columns].iloc[-1:].values
        
        # Scale features
        last_features_scaled = self.scaler.transform(last_features)
        
        # Predict next day's return
        predicted_return = self.model.predict(last_features_scaled)[0]
        
        # Convert return to price prediction
        current_price = self.data['Close'].iloc[-1]
        predicted_price = current_price * (1 + predicted_return)
        
        # Generate multi-day predictions (simplified approach)
        predictions = []
        price = current_price
        
        for i in range(days):
            # Use historical volatility to simulate price paths
            daily_volatility = self.data['Returns'].std()
            
            # Simple random walk with drift (predicted return)
            if i == 0:
                return_pred = predicted_return
            else:
                # Decay the prediction strength over time
                decay_factor = 0.95 ** i
                return_pred = predicted_return * decay_factor + np.random.normal(0, daily_volatility * 0.5)
            
            price = price * (1 + return_pred)
            predictions.append(price)
        
        # Create prediction dates
        last_date = self.data.index[-1]
        prediction_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        self.predictions = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Price': predictions
        })
        
        print(f"🔮 Generated {days}-day predictions")
        return self.predictions
    
    def create_visualizations(self, symbol):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 12))
        
        # Price and predictions plot
        ax1 = plt.subplot(3, 2, 1)
        recent_data = self.data.tail(100)
        plt.plot(recent_data.index, recent_data['Close'], label='Historical Price', linewidth=2)
        plt.plot(recent_data.index, recent_data['Price_MA_20'], label='20-day MA', alpha=0.7)
        
        if self.predictions is not None:
            plt.plot(self.predictions['Date'], self.predictions['Predicted_Price'], 
                    label='Predictions', color='red', linewidth=2, linestyle='--')
        
        plt.title(f'{symbol} - Price Analysis & Predictions', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Technical indicators
        ax2 = plt.subplot(3, 2, 2)
        plt.plot(recent_data.index, recent_data['RSI'], label='RSI', color='purple')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        plt.title('RSI Indicator', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Volume analysis
        ax3 = plt.subplot(3, 2, 3)
        plt.bar(recent_data.index, recent_data['Volume'], alpha=0.6, label='Volume')
        plt.plot(recent_data.index, recent_data['Volume_MA_20'], color='red', label='20-day MA')
        plt.title('Volume Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Returns distribution
        ax4 = plt.subplot(3, 2, 4)
        returns = self.data['Returns'].dropna()
        plt.hist(returns, bins=50, alpha=0.7, density=True, label='Daily Returns')
        plt.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        plt.title('Returns Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MACD
        ax5 = plt.subplot(3, 2, 5)
        plt.plot(recent_data.index, recent_data['MACD'], label='MACD', linewidth=2)
        plt.plot(recent_data.index, recent_data['MACD_Signal'], label='Signal', linewidth=2)
        plt.bar(recent_data.index, recent_data['MACD_Histogram'], alpha=0.6, label='Histogram')
        plt.title('MACD Indicator', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Bollinger Bands
        ax6 = plt.subplot(3, 2, 6)
        plt.plot(recent_data.index, recent_data['Close'], label='Price', linewidth=2)
        plt.plot(recent_data.index, recent_data['BB_Upper'], label='Upper Band', alpha=0.7)
        plt.plot(recent_data.index, recent_data['BB_Lower'], label='Lower Band', alpha=0.7)
        plt.fill_between(recent_data.index, recent_data['BB_Upper'], recent_data['BB_Lower'], alpha=0.1)
        plt.title('Bollinger Bands', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Comprehensive visualizations displayed!")
    
    def generate_report(self, symbol, model_score):
        """Generate comprehensive analysis report"""
        if self.predictions is None:
            print("❌ No predictions available")
            return
        
        current_price = self.data['Close'].iloc[-1]
        predicted_price_1d = self.predictions['Predicted_Price'].iloc[0]
        predicted_price_30d = self.predictions['Predicted_Price'].iloc[-1]
        
        # Calculate statistics
        expected_return_1d = (predicted_price_1d - current_price) / current_price * 100
        expected_return_30d = (predicted_price_30d - current_price) / current_price * 100
        
        # Risk metrics
        volatility = self.data['Returns'].std() * np.sqrt(252) * 100  # Annualized
        max_predicted = self.predictions['Predicted_Price'].max()
        min_predicted = self.predictions['Predicted_Price'].min()
        
        # Recent performance
        recent_return_1w = (current_price - self.data['Close'].iloc[-8]) / self.data['Close'].iloc[-8] * 100
        recent_return_1m = (current_price - self.data['Close'].iloc[-21]) / self.data['Close'].iloc[-21] * 100
        
        # Generate recommendation
        confidence = max(0, min(1, (model_score + 1) / 2))  # Convert R² to 0-1 scale
        
        if expected_return_30d > 5 and confidence > 0.6:
            recommendation = "🟢 BUY"
            risk_level = "🟡 MEDIUM"
        elif expected_return_30d < -5 and confidence > 0.6:
            recommendation = "🔴 SELL"
            risk_level = "🟠 HIGH"
        else:
            recommendation = "🟡 HOLD"
            risk_level = "🟢 LOW"
        
        # Display comprehensive report
        print(f"\n📈 Optimized Stock Analysis Report: {symbol}")
        print("="*70)
        
        # Current status
        company_name = self.stock_info.get('longName', 'N/A') if self.stock_info else 'N/A'
        sector = self.stock_info.get('sector', 'N/A') if self.stock_info else 'N/A'
        
        print(f"🏢 Company: {company_name}")
        print(f"🏭 Sector: {sector}")
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"📊 Annualized Volatility: {volatility:.1f}%")
        
        # Recent performance
        print(f"\n📈 Recent Performance:")
        print(f"   1 Week: {recent_return_1w:+.1f}%")
        print(f"   1 Month: {recent_return_1m:+.1f}%")
        
        # Predictions
        print(f"\n🔮 AI Predictions:")
        print(f"   1-Day Target: ${predicted_price_1d:.2f} ({expected_return_1d:+.1f}%)")
        print(f"   30-Day Target: ${predicted_price_30d:.2f} ({expected_return_30d:+.1f}%)")
        print(f"   30-Day Range: ${min_predicted:.2f} - ${max_predicted:.2f}")
        
        # Model performance
        print(f"\n🤖 Model Performance:")
        print(f"   R² Score: {model_score:.4f}")
        print(f"   Confidence Level: {confidence:.2f}/1.0")
        print(f"   Risk Assessment: {risk_level}")
        
        # Investment recommendation
        print(f"\n🎯 Investment Recommendation:")
        print(f"   Signal: {recommendation}")
        print(f"   Confidence: {confidence:.1%}")
        
        # Weekly targets
        print(f"\n📅 Weekly Price Targets:")
        weekly_predictions = self.predictions.iloc[::7]  # Every 7 days
        for i, row in weekly_predictions.head(5).iterrows():
            week_num = (i // 7) + 1
            date_str = row['Date'].strftime('%Y-%m-%d')
            price = row['Predicted_Price']
            change = (price - current_price) / current_price * 100
            print(f"   Week {week_num} ({date_str}): ${price:.2f} ({change:+.1f}%)")
        
        # Disclaimer
        print(f"\n⚠️  IMPORTANT DISCLAIMER:")
        print(f"   📚 This analysis is for educational purposes only.")
        print(f"   📊 Model R² score indicates prediction reliability.")
        print(f"   🔍 Always conduct additional research before investing.")
        print(f"   💼 Consider consulting with a financial advisor.")
        
        print(f"\n🎉 Analysis complete! Trade wisely! 📈")

def main():
    print("🚀 Optimized Stock Market Predictor v5.0")
    print("="*60)
    print("🎯 Realistic Predictions")
    print("📊 Proven Technical Analysis")
    print("🤖 Optimized Machine Learning")
    print("⚡ Enhanced Performance")
    print("="*60)
    
    # Initialize predictor
    predictor = OptimizedStockPredictor()
    
    # Get user input
    symbol = input("\n💼 Enter stock symbol (e.g., AAPL, TSLA, MSFT): ").upper().strip()
    
    if not symbol:
        symbol = "AAPL"  # Default
    
    print(f"\n📥 Fetching data for {symbol}...")
    
    # Fetch and process data
    if not predictor.fetch_data(symbol):
        return
    
    print("🔬 Creating technical features...")
    predictor.create_technical_features()
    
    print("📊 Preparing dataset for machine learning...")
    X, y = predictor.prepare_data()
    
    # Train model
    model_score = predictor.train_model(X, y)
    
    # Make predictions
    predictions = predictor.make_predictions(30)
    
    # Create visualizations
    print("\n📊 Creating comprehensive analysis charts...")
    predictor.create_visualizations(symbol)
    
    # Generate report
    predictor.generate_report(symbol, model_score)

if __name__ == "__main__":
    main()
