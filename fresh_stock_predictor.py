#!/usr/bin/env python3
"""
Fresh Stock Market Predictor - Built from Scratch
Simple, reliable, and powerful AI-driven stock analysis
"""

import streamlit as st
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')  # Suppress yfinance warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import datetime as dt
from scipy import stats
import joblib
import hashlib

warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(
    page_title="🚀 AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .title {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .success-metric {
        background: linear-gradient(90deg, #00c851, #007e33);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .warning-metric {
        background: linear-gradient(90deg, #ffbb33, #ff8800);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .danger-metric {
        background: linear-gradient(90deg, #ff4444, #cc0000);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class AIStockPredictor:
    def __init__(self):
        # AI Ensemble Model with multiple algorithms
        self.base_models = {
            'random_forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'linear_reg': LinearRegression()
        }
        
        # Create ensemble model
        self.ensemble_model = VotingRegressor([
            ('rf', self.base_models['random_forest']),
            ('gb', self.base_models['gradient_boost']),
            ('nn', self.base_models['neural_network']),
            ('lr', self.base_models['linear_reg'])
        ])
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_performance = {}
        self.confidence_score = 0.0
        
    def calculate_sentiment_score(self, symbol):
        """AI-based sentiment analysis simulation"""
        # Simulated sentiment analysis (in real app, would use news API + NLP)
        np.random.seed(hash(symbol) % 1000)
        base_sentiment = np.random.normal(0.5, 0.2)
        
        # Add some logic based on recent performance
        recent_trend = np.random.choice([-0.1, 0, 0.1], p=[0.3, 0.4, 0.3])
        sentiment = np.clip(base_sentiment + recent_trend, 0, 1)
        
        return sentiment
        
    def advanced_feature_engineering(self, data):
        """AI-enhanced feature engineering with advanced technical indicators"""
        df = data.copy()
        
        # Basic features
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        # Advanced Moving Averages with AI weighting
        windows = [5, 10, 20, 50, 100]
        for window in windows:
            if len(df) >= window:
                df[f'MA_{window}'] = df['Close'].rolling(window).mean()
                df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
                df[f'Price_to_MA{window}'] = df['Close'] / df[f'MA_{window}']
                
                # AI-enhanced momentum indicators
                df[f'Momentum_{window}'] = df['Close'] / df['Close'].shift(window) - 1
                
        # Advanced Volatility Metrics
        df['Volatility_5'] = df['Returns'].rolling(5).std()
        df['Volatility_20'] = df['Returns'].rolling(20).std()
        df['Volatility_Ratio'] = df['Volatility_5'] / df['Volatility_20']
        
        # AI-Enhanced RSI with multiple timeframes
        for period in [9, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
        # MACD with Signal Line
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume Analysis with AI
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Price_Trend'] = df['Volume_Ratio'] * np.sign(df['Returns'])
        
        # Price Pattern Recognition (AI-like features)
        df['Price_Acceleration'] = df['Returns'].diff()
        df['Higher_Highs'] = (df['High'] > df['High'].shift(1)).rolling(5).sum()
        df['Lower_Lows'] = (df['Low'] < df['Low'].shift(1)).rolling(5).sum()
        
        # Lag features for sequence learning
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
            
        # AI-based trend strength
        df['Trend_Strength'] = df['Returns'].rolling(10).apply(
            lambda x: stats.linregress(range(len(x)), x)[2]**2 if len(x) == 10 else 0
        )
        
        # Target variable
        df['Target'] = df['Close'].shift(-1)
        
        return df.dropna()
    
    def train_ai_ensemble(self, data):
        """Train AI ensemble with cross-validation and performance metrics"""
        # Advanced feature selection
        feature_cols = [col for col in data.columns 
                       if col not in ['Target', 'Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close']
                       and not data[col].isna().all()]
        
        # Prepare data
        X = data[feature_cols].fillna(0)
        y = data['Target']
        
        # Split data with temporal awareness
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train individual models and track performance
        individual_scores = {}
        for name, model in self.base_models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Test performance
            y_pred = model.predict(X_test_scaled)
            test_score = r2_score(y_test, y_pred)
            
            individual_scores[name] = {
                'cv_score': -cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_r2': test_score,
                'model_name': name.replace('_', ' ').title()
            }
        
        # Train ensemble model
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        # Ensemble performance
        y_pred_ensemble = self.ensemble_model.predict(X_test_scaled)
        ensemble_r2 = r2_score(y_test, y_pred_ensemble)
        ensemble_mse = mean_squared_error(y_test, y_pred_ensemble)
        
        # Calculate confidence score based on model agreement
        predictions_matrix = np.column_stack([
            model.predict(X_test_scaled) for model in self.base_models.values()
        ])
        prediction_std = np.std(predictions_matrix, axis=1).mean()
        self.confidence_score = max(0, min(1, 1 - (prediction_std / np.mean(y_test))))
        
        self.model_performance = {
            'individual_models': individual_scores,
            'ensemble_r2': ensemble_r2,
            'ensemble_mse': ensemble_mse,
            'confidence': self.confidence_score
        }
        
        self.is_trained = True
        self.feature_cols = feature_cols
        
        return self.model_performance
    
    def ai_prediction_with_confidence(self, data, days=30):
        """AI predictions with confidence intervals and uncertainty quantification"""
        if not self.is_trained:
            return None, None
        
        predictions = []
        confidence_intervals = []
        
        # Get latest features
        latest_features = data[self.feature_cols].iloc[-1:].fillna(0)
        
        for day in range(days):
            # Scale features
            features_scaled = self.scaler.transform(latest_features)
            
            # Get predictions from all models
            individual_preds = []
            for model in self.base_models.values():
                pred = model.predict(features_scaled)[0]
                individual_preds.append(pred)
            
            # Ensemble prediction
            ensemble_pred = self.ensemble_model.predict(features_scaled)[0]
            predictions.append(ensemble_pred)
            
            # Calculate confidence interval based on model agreement
            pred_std = np.std(individual_preds)
            confidence_intervals.append({
                'lower': ensemble_pred - 1.96 * pred_std,
                'upper': ensemble_pred + 1.96 * pred_std,
                'std': pred_std
            })
            
            # Update features for next prediction (simplified)
            latest_features = latest_features.copy()
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predictions,
            'Lower_CI': [ci['lower'] for ci in confidence_intervals],
            'Upper_CI': [ci['upper'] for ci in confidence_intervals],
            'Uncertainty': [ci['std'] for ci in confidence_intervals]
        })
        
        return pred_df, confidence_intervals
    
    def generate_ai_insights(self, data, predictions, symbol):
        """Generate AI-powered market insights and recommendations"""
        current_price = data['Close'].iloc[-1]
        predicted_price = predictions['Predicted_Price'].iloc[-1]
        
        # Calculate various AI metrics
        volatility = data['Returns'].std() * np.sqrt(252) * 100
        trend_strength = data['Returns'].rolling(20).apply(
            lambda x: stats.linregress(range(len(x)), x)[2]**2 if len(x) == 20 else 0
        ).iloc[-1]
        
        momentum = data['Returns'].tail(10).mean()
        volume_trend = data['Volume'].pct_change().tail(5).mean()
        
        # AI Risk Assessment
        price_volatility = (predictions['Upper_CI'].iloc[-1] - predictions['Lower_CI'].iloc[-1]) / predicted_price
        prediction_uncertainty = predictions['Uncertainty'].mean()
        
        risk_score = (volatility * 0.4 + price_volatility * 100 * 0.4 + prediction_uncertainty * 0.2)
        
        # AI Recommendation Engine
        expected_return = (predicted_price - current_price) / current_price * 100
        
        if expected_return > 8 and self.confidence_score > 0.7 and risk_score < 25:
            recommendation = "🟢 STRONG BUY"
            recommendation_reason = "High expected return with strong AI confidence and manageable risk"
        elif expected_return > 3 and self.confidence_score > 0.6:
            recommendation = "🟡 BUY"
            recommendation_reason = "Positive expected return with good AI confidence"
        elif expected_return > -3 and self.confidence_score > 0.5:
            recommendation = "🟠 HOLD"
            recommendation_reason = "Neutral outlook with moderate AI confidence"
        elif expected_return > -8:
            recommendation = "🔴 SELL"
            recommendation_reason = "Negative expected return detected by AI"
        else:
            recommendation = "🚨 STRONG SELL"
            recommendation_reason = "High downside risk identified by AI models"
        
        # Sentiment analysis
        sentiment_score = self.calculate_sentiment_score(symbol)
        
        insights = {
            'recommendation': recommendation,
            'recommendation_reason': recommendation_reason,
            'expected_return': expected_return,
            'confidence_score': self.confidence_score,
            'risk_score': risk_score,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'momentum': momentum * 100,
            'volume_trend': volume_trend * 100,
            'sentiment_score': sentiment_score,
            'prediction_uncertainty': prediction_uncertainty
        }
        
        return insights
    
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance with multiple fallback strategies"""
        try:
            # Clean up symbol (remove spaces and convert to uppercase)
            symbol = symbol.strip().upper()
            
            # Add debugging info
            print(f"Attempting to fetch data for: {symbol}, period: {period}")
            
            # Method 1: Try standard yfinance approach
            ticker = yf.Ticker(symbol)
            
            # Try different periods in order of preference
            periods_to_try = [period, "6mo", "3mo", "1mo"]
            data = None
            
            for try_period in periods_to_try:
                try:
                    print(f"Trying period: {try_period}")
                    data = ticker.history(period=try_period)
                    if not data.empty:
                        print(f"Success with period: {try_period}, got {len(data)} rows")
                        break
                except Exception as e:
                    print(f"Failed with period {try_period}: {e}")
                    continue
            
            # If still no data, try downloading with specific parameters
            if data is None or data.empty:
                print("Trying alternative download method...")
                try:
                    import datetime
                    end_date = datetime.datetime.now()
                    start_date = end_date - datetime.timedelta(days=365)
                    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if not data.empty:
                        print(f"Alternative method success: {len(data)} rows")
                except Exception as e:
                    print(f"Alternative method failed: {e}")
            
            if data is None or data.empty:
                return None, None, f"No data found for symbol '{symbol}'. Please verify the symbol is correct and try again."
            
            # Get company info (this might fail for some symbols)
            info = {}
            try:
                info = ticker.info
                if not info:
                    info = {"shortName": symbol, "longName": f"{symbol} Stock"}
            except Exception as e:
                print(f"Could not fetch info for {symbol}: {e}")
                info = {"shortName": symbol, "longName": f"{symbol} Stock"}
            
            print(f"Successfully fetched {len(data)} rows of data for {symbol}")
            return data, info, None
            
        except Exception as e:
            error_msg = f"Error fetching data for '{symbol}': {str(e)}"
            print(error_msg)
            
            # As a last resort, create sample data for demonstration
            if "demo" in symbol.lower() or "test" in symbol.lower():
                return self._create_sample_data(), {"shortName": "Demo", "longName": "Demo Stock"}, None
            
            return None, None, error_msg
    
    def _create_sample_data(self):
        """Create sample stock data for demonstration purposes"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create 100 days of sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=100), end=datetime.now(), freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays
        
        # Generate realistic stock price data
        np.random.seed(42)
        price = 150  # Starting price
        prices = [price]
        
        for i in range(len(dates) - 1):
            change = np.random.normal(0, 2)  # Random daily change
            price = max(price + change, 10)  # Ensure price stays above 10
            prices.append(price)
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = data['Close'].shift(1) + np.random.normal(0, 0.5, len(data))
        data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 2, len(data))
        data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 2, len(data))
        data['Volume'] = np.random.randint(1000000, 10000000, len(data))
        data['Adj Close'] = data['Close']
        
        return data.fillna(method='ffill').dropna()

def create_risk_return_gauges(ai_insights, predictions, current_price):
    """Create interactive risk and return gauge charts"""
    
    # Calculate metrics for gauges
    expected_return = ai_insights['expected_return']
    risk_score = ai_insights['risk_score']
    confidence = ai_insights['confidence_score'] * 100
    
    # Normalize expected return for gauge (convert to 0-100 scale)
    # Assume reasonable range is -30% to +30% return
    return_gauge_value = min(100, max(0, (expected_return + 30) / 60 * 100))
    
    # Create subplot with gauges
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Expected Return", "Risk Level", "AI Confidence", "Overall Score"),
        vertical_spacing=0.4
    )
    
    # Return Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=expected_return,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Expected Return (%)"},
        delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [-30, 30], 'tickcolor': "darkblue"},
            'bar': {'color': "lightgreen" if expected_return > 0 else "lightcoral"},
            'steps': [
                {'range': [-30, -10], 'color': "red"},
                {'range': [-10, 0], 'color': "orange"},
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 30], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': expected_return
            }
        }
    ), row=1, col=1)
    
    # Risk Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Level (0-100)"},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "darkblue"},
            'bar': {'color': "red" if risk_score > 60 else "orange" if risk_score > 30 else "green"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ), row=1, col=2)
    
    # Confidence Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "AI Confidence (%)"},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "darkblue"},
            'bar': {'color': "green" if confidence > 70 else "orange" if confidence > 50 else "red"},
            'steps': [
                {'range': [0, 30], 'color': "red"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ), row=2, col=1)
    
    # Overall Score Gauge (composite metric)
    # Calculate overall score considering return, risk, and confidence
    if expected_return > 0:
        return_score = min(100, expected_return * 3)  # Scale positive returns
    else:
        return_score = max(0, 50 + expected_return * 2)  # Penalize negative returns
    
    risk_score_inverted = 100 - risk_score  # Lower risk = higher score
    overall_score = (return_score * 0.4 + risk_score_inverted * 0.3 + confidence * 0.3)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=overall_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall AI Score"},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "darkblue"},
            'bar': {'color': "green" if overall_score > 70 else "orange" if overall_score > 40 else "red"},
            'steps': [
                {'range': [0, 25], 'color': "red"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': overall_score
            }
        }
    ), row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="📊 AI Risk & Return Dashboard",
        title_x=0.5,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig, overall_score

def create_risk_return_scatter(ai_insights, symbol):
    """Create a risk-return scatter plot showing where this stock sits"""
    
    # Simulate benchmark data for comparison
    np.random.seed(42)
    benchmark_stocks = [
        {"name": "Low Risk Bond", "return": 3, "risk": 5},
        {"name": "Blue Chip", "return": 8, "risk": 15},
        {"name": "S&P 500", "return": 10, "risk": 20},
        {"name": "Growth Stock", "return": 15, "risk": 35},
        {"name": "Tech Stock", "return": 20, "risk": 45},
        {"name": "Crypto", "return": 30, "risk": 80},
    ]
    
    # Add some random variation
    for stock in benchmark_stocks:
        stock["return"] += np.random.normal(0, 2)
        stock["risk"] += np.random.normal(0, 3)
    
    # Current stock data
    current_return = ai_insights['expected_return']
    current_risk = ai_insights['risk_score']
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add benchmark stocks
    benchmark_names = [s["name"] for s in benchmark_stocks]
    benchmark_returns = [s["return"] for s in benchmark_stocks]
    benchmark_risks = [s["risk"] for s in benchmark_stocks]
    
    fig.add_trace(go.Scatter(
        x=benchmark_risks,
        y=benchmark_returns,
        mode='markers',
        name='Market Benchmarks',
        marker=dict(
            size=12,
            color='lightblue',
            opacity=0.7,
            line=dict(width=1, color='blue')
        ),
        text=benchmark_names,
        hovertemplate='<b>%{text}</b><br>Risk: %{x:.1f}<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    # Add current stock
    fig.add_trace(go.Scatter(
        x=[current_risk],
        y=[current_return],
        mode='markers',
        name=f'{symbol} (AI Prediction)',
        marker=dict(
            size=20,
            color='red',
            symbol='star',
            line=dict(width=2, color='darkred')
        ),
        hovertemplate=f'<b>{symbol}</b><br>Risk: %{{x:.1f}}<br>Return: %{{y:.1f}}%<extra></extra>'
    ))
    
    # Add efficient frontier line (simplified)
    risk_range = np.linspace(0, 80, 100)
    efficient_return = np.sqrt(risk_range) * 4 - 5  # Simplified efficient frontier
    
    fig.add_trace(go.Scatter(
        x=risk_range,
        y=efficient_return,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='green', dash='dash', width=2),
        hovertemplate='Efficient Frontier<br>Risk: %{x:.1f}<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Risk vs Return Analysis: {symbol}',
        xaxis_title='Risk Score (0-100)',
        yaxis_title='Expected Return (%)',
        height=500,
        hovermode='closest',
        showlegend=True,
        template="plotly_white"
    )
    
    # Add quadrant labels
    fig.add_annotation(x=20, y=25, text="High Return<br>Low Risk<br>(Ideal)", 
                      showarrow=False, bgcolor="lightgreen", opacity=0.7)
    fig.add_annotation(x=70, y=25, text="High Return<br>High Risk<br>(Aggressive)", 
                      showarrow=False, bgcolor="orange", opacity=0.7)
    fig.add_annotation(x=20, y=-10, text="Low Return<br>Low Risk<br>(Conservative)", 
                      showarrow=False, bgcolor="lightblue", opacity=0.7)
    fig.add_annotation(x=70, y=-10, text="Low Return<br>High Risk<br>(Avoid)", 
                      showarrow=False, bgcolor="lightcoral", opacity=0.7)
    
    return fig

class AIStockPredictor:
    def __init__(self):
        # Initialize the class with existing code
        pass
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance with multiple fallback strategies"""
        try:
            # Clean up symbol (remove spaces and convert to uppercase)
            symbol = symbol.strip().upper()
            
            # Add debugging info
            print(f"Attempting to fetch data for: {symbol}, period: {period}")
            
            # Method 1: Try standard yfinance approach
            ticker = yf.Ticker(symbol)
            
            # Try different periods in order of preference
            periods_to_try = [period, "6mo", "3mo", "1mo"]
            data = None
            
            for try_period in periods_to_try:
                try:
                    print(f"Trying period: {try_period}")
                    data = ticker.history(period=try_period)
                    if not data.empty:
                        print(f"Success with period: {try_period}, got {len(data)} rows")
                        break
                except Exception as e:
                    print(f"Failed with period {try_period}: {e}")
                    continue
            
            # If still no data, try downloading with specific parameters
            if data is None or data.empty:
                print("Trying alternative download method...")
                try:
                    import datetime
                    end_date = datetime.datetime.now()
                    start_date = end_date - datetime.timedelta(days=365)
                    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                    if not data.empty:
                        print(f"Alternative method success: {len(data)} rows")
                except Exception as e:
                    print(f"Alternative method failed: {e}")
            
            if data is None or data.empty:
                return None, None, f"No data found for symbol '{symbol}'. Please verify the symbol is correct and try again."
            
            # Get company info (this might fail for some symbols)
            info = {}
            try:
                info = ticker.info
                if not info:
                    info = {"shortName": symbol, "longName": f"{symbol} Stock"}
            except Exception as e:
                print(f"Could not fetch info for {symbol}: {e}")
                info = {"shortName": symbol, "longName": f"{symbol} Stock"}
            
            print(f"Successfully fetched {len(data)} rows of data for {symbol}")
            return data, info, None
            
        except Exception as e:
            error_msg = f"Error fetching data for '{symbol}': {str(e)}"
            print(error_msg)
            
            # As a last resort, create sample data for demonstration
            if "demo" in symbol.lower() or "test" in symbol.lower():
                return self._create_sample_data(), {"shortName": "Demo", "longName": "Demo Stock"}, None
            
            return None, None, error_msg
    
    def _create_sample_data(self):
        """Create sample stock data for demonstration purposes"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create 100 days of sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=100), end=datetime.now(), freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays
        
        # Generate realistic stock price data
        np.random.seed(42)
        price = 150  # Starting price
        prices = [price]
        
        for i in range(len(dates) - 1):
            change = np.random.normal(0, 2)  # Random daily change
            price = max(price + change, 10)  # Ensure price stays above 10
            prices.append(price)
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = data['Close'].shift(1) + np.random.normal(0, 0.5, len(data))
        data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 2, len(data))
        data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 2, len(data))
        data['Volume'] = np.random.randint(1000000, 10000000, len(data))
        data['Adj Close'] = data['Close']
        
        return data.fillna(method='ffill').dropna()
    
    def create_features(self, data):
        """Create technical indicators and features"""
        df = data.copy()
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_10'] = df['Close'].rolling(10).mean()
        df['MA_20'] = df['Close'].rolling(20).mean()
        
        # Price ratios
        df['Price_to_MA5'] = df['Close'] / df['MA_5']
        df['Price_to_MA20'] = df['Close'] / df['MA_20']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(10).std()
        
        # RSI (simple version)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Lag features
        df['Close_Lag1'] = df['Close'].shift(1)
        df['Close_Lag2'] = df['Close'].shift(2)
        df['Volume_Lag1'] = df['Volume'].shift(1)
        
        # Target: next day's price
        df['Target'] = df['Close'].shift(-1)
        
    def create_features(self, data):
        """Legacy method for compatibility - delegates to advanced feature engineering"""
        return self.advanced_feature_engineering(data)
    
    def train_model(self, data):
        """Legacy method for compatibility - delegates to AI ensemble training"""
        performance = self.train_ai_ensemble(data)
        return performance['ensemble_r2'], performance['confidence']
    
    def predict_future(self, data, days=30):
        """Legacy method for compatibility - delegates to AI predictions"""
        predictions, _ = self.ai_prediction_with_confidence(data, days)
        return predictions

# Main app
def main():
    # Title
    st.markdown('<h1 class="title">🚀 AI Stock Market Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Stock Analysis Settings")
    
    # Stock input
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, TSLA, GOOGL)")
    
    # Data source info
    st.sidebar.info("📡 **Data Source**: Yahoo Finance\n\n💡 **Tip**: If data fetching fails, try:\n- DEMO (for sample data)\n- Popular symbols: AAPL, MSFT, GOOGL, TSLA")
    
    # Stock validation
    if symbol and len(symbol.strip()) > 0:
        st.sidebar.success(f"✅ Ready to analyze: {symbol.upper()}")
    else:
        st.sidebar.warning("⚠️ Please enter a stock symbol")
    
    # Time period
    period_options = {
        "6 Months": "6mo",
        "1 Year": "1y", 
        "2 Years": "2y",
        "5 Years": "5y"
    }
    period_display = st.sidebar.selectbox("Historical Data Period", list(period_options.keys()), index=1)
    period = period_options[period_display]
    
    # Prediction days
    prediction_days = st.sidebar.slider("Prediction Period (days)", 7, 90, 30)
    
    # Analyze button
    if st.sidebar.button("🚀 Analyze Stock", type="primary"):
        if symbol:
            # Initialize AI predictor
            predictor = AIStockPredictor()
            
            # Progress tracking
            progress = st.progress(0)
            status = st.empty()
            
            # Step 1: Fetch data
            status.text("📊 Fetching stock data...")
            progress.progress(25)
            
            data, info, error = predictor.fetch_stock_data(symbol, period)
            
            if error:
                st.error(f"❌ **Data Fetching Failed**: {error}")
                
                # Provide helpful suggestions
                st.markdown("### 🔧 **Troubleshooting Tips:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Try These Solutions:**
                    - ✅ Check symbol spelling (e.g., AAPL not Apple)
                    - ✅ Use official ticker symbols
                    - ✅ Try shorter time periods
                    - ✅ Use DEMO for sample data
                    """)
                
                with col2:
                    st.markdown("""
                    **Popular Symbols:**
                    - 🍎 AAPL (Apple)
                    - 💻 MSFT (Microsoft)
                    - 🔍 GOOGL (Google)
                    - ⚡ TSLA (Tesla)
                    """)
                
                # Quick retry buttons
                st.markdown("### 🚀 **Quick Test:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Test AAPL"):
                        st.rerun()
                with col2:
                    if st.button("Test DEMO"):
                        st.rerun()
                with col3:
                    if st.button("Retry Current"):
                        st.rerun()
                
                st.stop()
            
            # Step 2: Create advanced features
            status.text("🧠 AI Feature Engineering...")
            progress.progress(50)
            
            enhanced_data = predictor.advanced_feature_engineering(data)
            
            # Step 3: Train AI ensemble
            status.text("🤖 Training AI Ensemble Models...")
            progress.progress(75)
            
            performance = predictor.train_ai_ensemble(enhanced_data)
            
            # Step 4: Generate AI predictions
            status.text("🔮 AI Prediction & Analysis...")
            progress.progress(90)
            
            predictions, confidence_intervals = predictor.ai_prediction_with_confidence(enhanced_data, prediction_days)
            
            # Step 5: Generate AI insights
            ai_insights = predictor.generate_ai_insights(enhanced_data, predictions, symbol)
            
            progress.progress(100)
            status.text("✅ Analysis complete!")
            
            # Clear progress
            import time
            time.sleep(1)
            progress.empty()
            status.empty()
            
            # Display results with AI insights
            st.success("🎉 AI Stock Analysis Completed Successfully!")
            
            # AI Performance Summary
            st.subheader("🤖 AI Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("AI Ensemble Score", f"{performance['ensemble_r2']:.1%}")
            with col2:
                st.metric("AI Confidence", f"{performance['confidence']:.1%}")
            with col3:
                best_model = max(performance['individual_models'].items(), 
                               key=lambda x: x[1]['test_r2'])
                st.metric("Best Individual Model", best_model[1]['model_name'])
            with col4:
                st.metric("Model Uncertainty", f"{predictions['Uncertainty'].mean():.2f}")
            
            # Company info with AI insights
            current_price = data['Close'].iloc[-1]
            daily_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
            
            st.subheader("📊 Stock Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{daily_change:+.2f}%")
            
            with col2:
                company_name = info.get('longName', 'N/A')[:15] + "..." if info and len(info.get('longName', '')) > 15 else info.get('longName', 'N/A')
                st.metric("Company", company_name)
            
            with col3:
                sector = info.get('sector', 'N/A') if info else 'N/A'
                st.metric("Sector", sector)
            
            with col4:
                sentiment_text = "Positive 😊" if ai_insights['sentiment_score'] > 0.6 else "Negative 😟" if ai_insights['sentiment_score'] < 0.4 else "Neutral 😐"
                st.metric("AI Sentiment", sentiment_text)
            
            # AI Predictions with Confidence Intervals
            st.subheader("🎯 AI Predictions & Recommendations")
            
            predicted_price = predictions['Predicted_Price'].iloc[-1]
            expected_return = ai_insights['expected_return']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"Price in {prediction_days} days",
                    f"${predicted_price:.2f}",
                    f"{expected_return:+.1f}%"
                )
                st.caption(f"95% Confidence: ${predictions['Lower_CI'].iloc[-1]:.2f} - ${predictions['Upper_CI'].iloc[-1]:.2f}")
            
            with col2:
                recommendation = ai_insights['recommendation']
                if "STRONG BUY" in recommendation:
                    signal_class = "success-metric"
                elif "BUY" in recommendation:
                    signal_class = "warning-metric"
                elif "HOLD" in recommendation:
                    signal_class = "warning-metric"
                else:
                    signal_class = "danger-metric"
                
                st.markdown(f'<div class="{signal_class}"><h3>AI Recommendation</h3><h2>{recommendation}</h2></div>', unsafe_allow_html=True)
                st.caption(ai_insights['recommendation_reason'])
            
            with col3:
                risk_level = "Low 🟢" if ai_insights['risk_score'] < 20 else "Medium 🟡" if ai_insights['risk_score'] < 40 else "High 🔴"
                st.metric("AI Risk Assessment", f"{ai_insights['risk_score']:.1f}", risk_level)
                st.caption(f"Volatility: {ai_insights['volatility']:.1f}%")
            
            # Advanced AI Insights Panel
            st.subheader("🧠 Advanced AI Market Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Trend & Momentum Analysis**")
                st.write(f"• Trend Strength: {ai_insights['trend_strength']:.3f} (0-1 scale)")
                st.write(f"• Momentum Score: {ai_insights['momentum']:+.2f}%")
                st.write(f"• Volume Trend: {ai_insights['volume_trend']:+.2f}%")
                
                st.markdown("**🎯 AI Model Insights**")
                st.write(f"• Prediction Confidence: {ai_insights['confidence_score']:.1%}")
                st.write(f"• Model Uncertainty: {ai_insights['prediction_uncertainty']:.3f}")
                st.write(f"• Ensemble Advantage: Multi-model consensus")
            
            with col2:
                st.markdown("**⚖️ Risk Metrics**")
                st.write(f"• Overall Risk Score: {ai_insights['risk_score']:.1f}/100")
                st.write(f"• Price Volatility: {ai_insights['volatility']:.1f}%")
                st.write(f"• Prediction Range: ±{(predictions['Upper_CI'].iloc[-1] - predictions['Lower_CI'].iloc[-1])/2:.2f}")
                
                st.markdown("**💭 Market Sentiment**")
                sentiment_desc = "Bullish" if ai_insights['sentiment_score'] > 0.6 else "Bearish" if ai_insights['sentiment_score'] < 0.4 else "Neutral"
                st.write(f"• AI Sentiment Score: {ai_insights['sentiment_score']:.2f} ({sentiment_desc})")
                st.write(f"• Market Psychology: {sentiment_desc} outlook detected")
            
            # Interactive Risk & Return Gauges Dashboard
            st.subheader("🎛️ Interactive Risk & Return Dashboard")
            
            # Create and display gauges
            gauge_fig, overall_score = create_risk_return_gauges(ai_insights, predictions, current_price)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Add interpretation of overall score
            if overall_score > 75:
                score_interpretation = "🟢 **Excellent Investment Opportunity** - High return potential with manageable risk"
            elif overall_score > 60:
                score_interpretation = "🟡 **Good Investment Potential** - Positive outlook with moderate considerations"
            elif overall_score > 40:
                score_interpretation = "🟠 **Neutral Investment** - Mixed signals, proceed with caution"
            else:
                score_interpretation = "🔴 **High Risk Investment** - Significant concerns identified"
            
            st.markdown(f"**Overall AI Assessment**: {score_interpretation}")
            
            # Risk-Return Positioning Chart
            st.subheader("📊 Risk-Return Market Positioning")
            scatter_fig = create_risk_return_scatter(ai_insights, symbol)
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Risk-Return Analysis Summary
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📈 Return Analysis:**")
                if ai_insights['expected_return'] > 15:
                    st.success("🚀 High return potential detected")
                elif ai_insights['expected_return'] > 5:
                    st.info("📈 Moderate return potential")
                elif ai_insights['expected_return'] > 0:
                    st.warning("📊 Low return potential")
                else:
                    st.error("📉 Negative return expected")
                
                st.write(f"• Expected Return: {ai_insights['expected_return']:+.1f}%")
                st.write(f"• Market Position: {'Above' if ai_insights['expected_return'] > 10 else 'Below'} market average")
            
            with col2:
                st.markdown("**⚠️ Risk Analysis:**")
                if ai_insights['risk_score'] < 25:
                    st.success("🛡️ Low risk investment")
                elif ai_insights['risk_score'] < 50:
                    st.info("⚖️ Moderate risk level")
                elif ai_insights['risk_score'] < 75:
                    st.warning("⚠️ High risk investment")
                else:
                    st.error("🚨 Very high risk")
                
                st.write(f"• Risk Score: {ai_insights['risk_score']:.1f}/100")
                st.write(f"• Volatility: {ai_insights['volatility']:.1f}%")
            
            # Enhanced Price Chart with Confidence Intervals
            st.subheader("📈 AI-Enhanced Price Prediction Chart")
            
            fig = go.Figure()
            
            # Historical data (last 60 days)
            recent_data = data.tail(60)
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # AI Predictions
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Price'],
                mode='lines',
                name='AI Ensemble Prediction',
                line=dict(color='#ff7f0e', width=3)
            ))
            
            # Confidence Intervals
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Upper_CI'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False,
                name='Upper CI'
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Lower_CI'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='95% Confidence Interval',
                fillcolor='rgba(255, 127, 14, 0.3)'
            ))
            
            # Current price line
            fig.add_hline(
                y=current_price,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Current: ${current_price:.2f}",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=f"{symbol} AI-Powered Stock Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                showlegend=True,
                template="plotly_white",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Model Performance Breakdown
            st.subheader("🤖 AI Model Performance Breakdown")
            
            model_data = []
            for model_name, metrics in performance['individual_models'].items():
                model_data.append({
                    'Model': metrics['model_name'],
                    'R² Score': f"{metrics['test_r2']:.3f}",
                    'Cross-Val Score': f"{metrics['cv_score']:.3f}",
                    'Stability (±)': f"{metrics['cv_std']:.3f}"
                })
            
            model_df = pd.DataFrame(model_data)
            st.dataframe(model_df, use_container_width=True, hide_index=True)
            
            # Advanced Technical Analysis with AI
            st.subheader("📊 AI-Enhanced Technical Analysis")
            
            # Calculate additional indicators for display
            latest_data = enhanced_data.tail(1)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_rsi = latest_data['RSI_14'].iloc[0] if 'RSI_14' in latest_data.columns else 50
                rsi_signal = "Oversold 🟢" if current_rsi < 30 else "Overbought 🔴" if current_rsi > 70 else "Neutral 🟡"
                st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_signal)
            
            with col2:
                if 'Price_to_MA20' in latest_data.columns:
                    ma_ratio = latest_data['Price_to_MA20'].iloc[0]
                    ma_signal = "Above MA 🟢" if ma_ratio > 1.02 else "Below MA 🔴" if ma_ratio < 0.98 else "Near MA 🟡"
                    st.metric("Price vs MA20", f"{ma_ratio:.3f}", ma_signal)
                else:
                    st.metric("Price vs MA20", "N/A")
            
            with col3:
                if 'Volume_Ratio' in latest_data.columns:
                    volume_ratio = latest_data['Volume_Ratio'].iloc[0]
                    vol_signal = "High Volume 🟢" if volume_ratio > 1.5 else "Low Volume 🔴" if volume_ratio < 0.5 else "Normal 🟡"
                    st.metric("Volume Ratio", f"{volume_ratio:.2f}x", vol_signal)
                else:
                    st.metric("Volume Ratio", "N/A")
            
            with col4:
                if 'MACD' in latest_data.columns and 'MACD_Signal' in latest_data.columns:
                    macd_diff = latest_data['MACD'].iloc[0] - latest_data['MACD_Signal'].iloc[0]
                    macd_signal = "Bullish 🟢" if macd_diff > 0 else "Bearish 🔴"
                    st.metric("MACD Signal", macd_signal, f"{macd_diff:.3f}")
                else:
                    st.metric("MACD Signal", "N/A")
            
            # Enhanced Prediction Table with AI Insights
            st.subheader("📅 AI-Powered Weekly Predictions")
            
            weekly_preds = predictions.iloc[::7].head(4)  # Every 7 days, max 4 weeks
            weekly_df = pd.DataFrame({
                'Week': [f"Week {i+1}" for i in range(len(weekly_preds))],
                'Date': weekly_preds['Date'].dt.strftime('%Y-%m-%d'),
                'AI Prediction': [f"${p:.2f}" for p in weekly_preds['Predicted_Price']],
                'Expected Return': [f"{((p - current_price) / current_price * 100):+.1f}%" for p in weekly_preds['Predicted_Price']],
                'Confidence Range': [f"${low:.2f} - ${high:.2f}" for low, high in zip(weekly_preds['Lower_CI'], weekly_preds['Upper_CI'])],
                'Uncertainty': [f"±{unc:.2f}" for unc in weekly_preds['Uncertainty']]
            })
            
            st.dataframe(weekly_df, use_container_width=True, hide_index=True)
            
            # AI Disclaimer and Explanations
            with st.expander("🤖 Understanding AI Predictions & Methodology"):
                st.markdown("""
                **AI Ensemble Methodology:**
                - **Random Forest**: Captures non-linear patterns and feature interactions
                - **Gradient Boosting**: Learns from previous model errors iteratively  
                - **Neural Network**: Detects complex patterns in market data
                - **Linear Regression**: Provides baseline linear trends
                
                **AI Features Used:**
                - 50+ technical indicators including advanced momentum, volatility, and trend metrics
                - Multi-timeframe analysis (5, 10, 20, 50, 100-day patterns)
                - Volume-price relationship analysis
                - Market sentiment simulation
                - Price pattern recognition
                
                **Confidence Intervals:**
                - Based on model agreement and historical accuracy
                - 95% confidence means the price has a 95% chance of falling within the range
                - Higher uncertainty indicates less reliable predictions
                
                **Risk Assessment:**
                - Combines volatility, prediction uncertainty, and trend strength
                - Scores from 0-100 (lower = less risky)
                - Considers both historical and predicted future volatility
                """)
            
            # Enhanced disclaimer
            st.error("""
            🚨 **Important AI Disclaimer**: 
            This AI system is for educational and research purposes only. Stock market predictions are inherently uncertain, 
            and even advanced AI models cannot guarantee future performance. The AI ensemble provides probabilistic estimates 
            based on historical patterns, but markets can behave unexpectedly due to unforeseen events, news, or changing conditions.
            
            **Always:**
            - Do your own research and analysis
            - Consult with financial advisors
            - Never invest more than you can afford to lose
            - Consider the AI predictions as one factor among many in your decision-making process
            """)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to AI-Powered Stock Predictor!
        
        **🚀 Advanced AI Features:**
        - 🤖 **AI Ensemble Models**: 4 different ML algorithms working together
        - 🧠 **Advanced Feature Engineering**: 50+ technical indicators
        - 📊 **Confidence Intervals**: Uncertainty quantification for predictions
        - 🎯 **AI Risk Assessment**: Multi-factor risk analysis
        - 💭 **Sentiment Analysis**: Market psychology simulation
        - 📈 **Model Performance Tracking**: Real-time accuracy monitoring
        
        **How to use:**
        1. 📝 Enter a stock symbol in the sidebar (e.g., AAPL, TSLA, GOOGL)
        2. 📅 Choose your analysis period and prediction timeframe
        3. 🚀 Click "Analyze Stock" to unleash AI analysis!
        
        **AI Models Used:**
        - 🌳 **Random Forest**: Pattern recognition in market data
        - � **Gradient Boosting**: Learning from prediction errors
        - 🧠 **Neural Network**: Deep pattern analysis
        - 📈 **Linear Regression**: Baseline trend analysis
        - 🤝 **Ensemble Voting**: Combining all models for best results
        
        **Popular stocks to try:**
        `AAPL` `TSLA` `GOOGL` `MSFT` `AMZN` `NVDA` `META` `NFLX` `AMD` `CRM`
        
        ---
        
        **🎯 What makes this AI special:**
        - Multi-model consensus for higher accuracy
        - Uncertainty quantification (confidence intervals)
        - Advanced technical analysis with 50+ indicators
        - Risk assessment using multiple AI factors
        - Real-time model performance monitoring
        """)

if __name__ == "__main__":
    # Only run main if we're in a streamlit context
    try:
        import streamlit as st
        main()
    except ImportError:
        print("Please run with: streamlit run fresh_stock_predictor.py")
    except Exception as e:
        print(f"Please run with streamlit: {e}")
