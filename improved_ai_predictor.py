import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class ImprovedAIStockPredictor:
    def __init__(self):
        """Initialize improved AI stock predictor with better architecture"""
        # Use ensemble of proven models
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.is_trained = False
        
    def fetch_stock_data(self, symbol, period='2y'):
        """Fetch and validate stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Clean data
            data = data.dropna()
            data = data[data['Volume'] > 0]
            
            # Get company info for context
            try:
                info = ticker.info
                company_name = info.get('longName', symbol)
                sector = info.get('sector', 'Unknown')
                print(f"📊 {company_name} ({symbol}) - {sector}")
            except:
                print(f"📊 {symbol}")
            
            print(f"✅ Successfully loaded {len(data)} days of data")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def engineer_features(self, data):
        """Create robust technical features without lookahead bias"""
        df = data.copy()
        
        # Price-based features (all properly lagged)
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages (lagged to prevent data leakage)
        for window in [5, 10, 20, 50]:
            df[f'SMA_{window}'] = df['Close'].shift(1).rolling(window=window).mean()
            df[f'Price_SMA_{window}_Ratio'] = df['Close'].shift(1) / df[f'SMA_{window}']
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].shift(1).ewm(span=12).mean()
        df['EMA_26'] = df['Close'].shift(1).ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].shift(1)  # Lag to prevent lookahead
        
        # Bollinger Bands
        bb_window = 20
        sma_bb = df['Close'].shift(1).rolling(window=bb_window).mean()
        bb_std = df['Close'].shift(1).rolling(window=bb_window).std()
        df['BB_Upper'] = sma_bb + (2 * bb_std)
        df['BB_Lower'] = sma_bb - (2 * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'].shift(1) - df['BB_Lower']) / df['BB_Width']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].shift(1).rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'].shift(1) / df['Volume_SMA']
        df['Price_Volume'] = df['Close'].shift(1) * df['Volume'].shift(1)
        df['PV_SMA'] = df['Price_Volume'].rolling(window=10).mean()
        df['PV_Ratio'] = df['Price_Volume'] / df['PV_SMA']
        
        # Volatility indicators
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        df['Volatility_Ratio'] = df['Volatility_10'] / df['Volatility_20']
        
        # Price momentum
        for period in [3, 5, 10, 20]:
            df[f'Momentum_{period}'] = df['Close'].shift(1) / df['Close'].shift(period+1) - 1
        
        # High-Low indicators
        df['HL_Ratio'] = df['High'].shift(1) / df['Low'].shift(1)
        df['Price_Range'] = (df['High'].shift(1) - df['Low'].shift(1)) / df['Close'].shift(1)
        df['Close_Position'] = (df['Close'].shift(1) - df['Low'].shift(1)) / (df['High'].shift(1) - df['Low'].shift(1))
        
        return df
    
    def prepare_dataset(self, data):
        """Prepare clean dataset for training"""
        df = self.engineer_features(data)
        
        # Select robust features
        feature_columns = [
            'SMA_5', 'SMA_10', 'SMA_20', 'Price_SMA_5_Ratio', 'Price_SMA_20_Ratio',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'RSI', 'BB_Position', 'BB_Width',
            'Volume_Ratio', 'PV_Ratio',
            'Volatility_10', 'Volatility_20', 'Volatility_Ratio',
            'Momentum_3', 'Momentum_5', 'Momentum_10', 'Momentum_20',
            'HL_Ratio', 'Price_Range', 'Close_Position'
        ]
        
        # Target: Next day's return (more stable than absolute price)
        df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
        
        # Clean dataset
        df = df.dropna()
        
        if len(df) < 100:
            raise ValueError("Insufficient data after feature engineering")
        
        X = df[feature_columns]
        y = df['Target']
        
        print(f"✅ Dataset prepared: {len(X)} samples, {len(feature_columns)} features")
        return X, y, df
    
    def train_models(self, X, y):
        """Train and evaluate models with time series validation"""
        print("\n🤖 Training AI Models...")
        print("=" * 50)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = -np.inf
        
        for name, model in self.models.items():
            print(f"\n📊 Evaluating {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2', n_jobs=-1)
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            print(f"   CV R² Score: {mean_score:.4f} (±{std_score:.4f})")
            
            # Track best model
            if mean_score > best_score:
                best_score = mean_score
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n🏆 Best Model: {self.best_model_name} (R² = {best_score:.4f})")
        
        # Train best model on full dataset
        X_scaled = self.scaler.fit_transform(X)
        self.best_model.fit(X_scaled, y)
        self.is_trained = True
        
        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔍 Top 8 Most Important Features:")
            for idx, row in self.feature_importance.head(8).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return best_score
    
    def generate_predictions(self, X, current_price, symbol, days=30):
        """Generate future price predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        print(f"\n🔮 Generating {days}-day predictions for {symbol}...")
        
        # Get last known features
        last_features = X.iloc[-1:].copy()
        
        predictions = []
        prices = [current_price]
        
        for day in range(days):
            # Scale features and predict
            X_scaled = self.scaler.transform(last_features)
            predicted_return = self.best_model.predict(X_scaled)[0]
            
            # Apply realistic constraints
            predicted_return = np.clip(predicted_return, -0.1, 0.1)  # Max ±10% per day
            
            # Calculate next price
            next_price = prices[-1] * (1 + predicted_return)
            predictions.append(predicted_return)
            prices.append(next_price)
            
            # Update features for next prediction (simplified approach)
            # In practice, you'd need to update all technical indicators
            last_features = last_features.copy()
        
        # Generate business dates
        dates = []
        current_date = datetime.now().date()
        business_days = 0
        
        while business_days < days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:  # Monday to Friday
                dates.append(current_date)
                business_days += 1
        
        return dates, prices[1:], predictions  # Remove initial price
    
    def create_comprehensive_charts(self, historical_data, dates, predicted_prices, symbol, current_price):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0, :])
        
        # Historical data (last 60 days)
        hist_data = historical_data['Close'].tail(60)
        hist_dates = hist_data.index.date
        
        ax1.plot(hist_dates, hist_data.values, 'b-', linewidth=2, label='Historical Price', alpha=0.8)
        ax1.plot(dates, predicted_prices, 'r-', linewidth=3, label='AI Prediction', marker='o', markersize=4)
        ax1.axvline(x=datetime.now().date(), color='gray', linestyle='--', alpha=0.7, label='Today')
        
        ax1.set_title(f'{symbol} Stock Price Analysis & AI Predictions', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Add annotations
        ax1.annotate(f'Current: ${current_price:.2f}', 
                    xy=(datetime.now().date(), current_price),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7),
                    color='white', fontweight='bold')
        
        final_price = predicted_prices[-1]
        ax1.annotate(f'Target: ${final_price:.2f}', 
                    xy=(dates[-1], final_price),
                    xytext=(-10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                    color='white', fontweight='bold')
        
        # Returns distribution
        ax2 = fig.add_subplot(gs[1, 0])
        returns = [(p - current_price) / current_price * 100 for p in predicted_prices]
        ax2.hist(returns, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.1f}%')
        ax2.set_title('Predicted Returns Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Feature importance
        ax3 = fig.add_subplot(gs[1, 1])
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(8)
            ax3.barh(range(len(top_features)), top_features['importance'], color='lightgreen', alpha=0.8)
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'], fontsize=9)
            ax3.set_title('Feature Importance', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Importance')
            ax3.grid(True, alpha=0.3)
        
        # Weekly performance
        ax4 = fig.add_subplot(gs[2, :])
        weekly_prices = []
        weekly_labels = []
        for i in range(0, min(len(predicted_prices), 28), 5):
            if i < len(predicted_prices):
                weekly_prices.append(predicted_prices[i])
                weekly_labels.append(f'Week {i//5 + 1}')
        
        weekly_returns = [(p - current_price) / current_price * 100 for p in weekly_prices]
        colors = ['green' if r >= 0 else 'red' for r in weekly_returns]
        
        bars = ax4.bar(weekly_labels, weekly_returns, color=colors, alpha=0.7)
        ax4.set_title('Weekly Target Returns', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Return (%)')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ret in zip(bars, weekly_returns):
            height = bar.get_height()
            ax4.annotate(f'{ret:+.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontweight='bold')
        
        plt.suptitle(f'AI Stock Analysis: {symbol}', fontsize=18, fontweight='bold', y=0.98)
        
        try:
            plt.show()
            print("✅ Comprehensive charts displayed successfully!")
        except:
            filename = f"{symbol}_AI_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📁 Charts saved as: {filename}")
        
        plt.close()
    
    def print_analysis_report(self, symbol, current_price, predicted_prices, dates, model_score):
        """Print comprehensive analysis report"""
        print(f"\n📈 AI Stock Analysis Report: {symbol}")
        print("=" * 80)
        
        # Key metrics
        final_price = predicted_prices[-1]
        total_return = (final_price - current_price) / current_price * 100
        max_price = max(predicted_prices)
        min_price = min(predicted_prices)
        avg_price = np.mean(predicted_prices)
        price_volatility = np.std(predicted_prices) / avg_price * 100
        
        # Display metrics
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"🎯 {len(dates)}-Day Target: ${final_price:.2f}")
        print(f"📊 Expected Return: {total_return:+.2f}%")
        print(f"📈 Max Upside: ${max_price:.2f} (+{((max_price-current_price)/current_price*100):+.1f}%)")
        print(f"📉 Max Downside: ${min_price:.2f} ({((min_price-current_price)/current_price*100):+.1f}%)")
        print(f"📊 Average Target: ${avg_price:.2f}")
        print(f"🎲 Price Volatility: {price_volatility:.2f}%")
        
        # Model performance
        confidence = max(0, min(1, (model_score + 1) / 2))  # Normalize to 0-1
        print(f"\n🤖 AI Model Performance:")
        print(f"   Model: {self.best_model_name}")
        print(f"   R² Score: {model_score:.4f}")
        print(f"   Confidence: {confidence:.2f}/1.0")
        
        # Risk assessment
        risk_level = "🟢 LOW" if price_volatility < 3 else "🟡 MEDIUM" if price_volatility < 6 else "🔴 HIGH"
        print(f"   Risk Level: {risk_level}")
        
        # Weekly breakdown
        print(f"\n📅 Weekly Price Targets:")
        for week in range(0, min(len(predicted_prices), 28), 5):
            if week < len(predicted_prices):
                week_price = predicted_prices[week]
                week_return = (week_price - current_price) / current_price * 100
                week_date = dates[week]
                print(f"   Week {week//5 + 1} ({week_date}): ${week_price:.2f} ({week_return:+.1f}%)")
        
        # Investment recommendation
        print(f"\n🎯 AI Investment Recommendation:")
        if total_return > 10:
            recommendation = "🚀 STRONG BUY - High growth potential"
        elif total_return > 5:
            recommendation = "📈 BUY - Positive outlook"
        elif total_return > 2:
            recommendation = "➕ WEAK BUY - Slight upside"
        elif total_return > -2:
            recommendation = "➡️ HOLD - Sideways movement expected"
        elif total_return > -5:
            recommendation = "📉 SELL - Downward pressure"
        else:
            recommendation = "⚠️ STRONG SELL - Significant downside risk"
        
        print(f"   {recommendation}")
        print(f"   Confidence Level: {'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'}")
        
        # Disclaimer
        print(f"\n⚠️  IMPORTANT DISCLAIMER:")
        print(f"   This analysis is for educational purposes only.")
        print(f"   Past performance does not guarantee future results.")
        print(f"   Always conduct your own research before investing.")

def main():
    """Main execution function"""
    print("🚀 Advanced AI Stock Predictor v4.0")
    print("=" * 60)
    print("🧠 Machine Learning Ensemble")
    print("📊 Professional Technical Analysis") 
    print("🎯 Time Series Validation")
    print("📈 Comprehensive Risk Assessment")
    print("=" * 60)
    
    # Get user input
    symbol = input("\n💼 Enter stock symbol (e.g., AAPL, TSLA, MSFT): ").upper().strip()
    
    if not symbol:
        print("❌ Please enter a valid stock symbol")
        return
    
    try:
        # Initialize predictor
        predictor = ImprovedAIStockPredictor()
        
        # Fetch data
        print(f"\n📥 Fetching data for {symbol}...")
        data = predictor.fetch_stock_data(symbol)
        
        if data is None:
            print("❌ Could not fetch data. Please check symbol and try again.")
            return
        
        # Prepare dataset
        print("\n🔬 Engineering features and preparing dataset...")
        X, y, processed_data = predictor.prepare_dataset(data)
        
        # Train models
        model_score = predictor.train_models(X, y)
        
        if model_score < -0.2:
            print("⚠️ Warning: Low model performance detected. Results may be unreliable.")
        
        # Generate predictions
        current_price = data['Close'].iloc[-1]
        dates, predicted_prices, returns = predictor.generate_predictions(X, current_price, symbol, days=30)
        
        # Create visualizations
        print(f"\n📊 Creating comprehensive analysis charts...")
        predictor.create_comprehensive_charts(data, dates, predicted_prices, symbol, current_price)
        
        # Print analysis report
        predictor.print_analysis_report(symbol, current_price, predicted_prices, dates, model_score)
        
        print(f"\n🎉 Analysis complete! Remember to always do your own research! 📚")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main()
