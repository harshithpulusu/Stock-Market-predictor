#!/usr/bin/env python3
"""
Stock Market Predictor Web App
A comprehensive stock analysis application built with Streamlit
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import datetime as dt
from io import StringIO
import time
import hashlib
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import functools

warnings.filterwarnings('ignore')

# Create cache directory if it doesn't exist
CACHE_DIR = ".streamlit_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Page configuration with performance optimizations
st.set_page_config(
    page_title="AI Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    /* Performance optimization: reduce animation overhead */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #1E88E5, #42A5F5);
    }
</style>
""", unsafe_allow_html=True)

# Efficiency optimization functions
def create_cache_key(*args):
    """Create a unique cache key from arguments"""
    key_str = str(args)
    return hashlib.md5(key_str.encode()).hexdigest()

def load_from_cache(cache_key):
    """Load data from disk cache"""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_to_cache(cache_key, data):
    """Save data to disk cache"""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except:
        pass

@st.cache_data(ttl=7200, max_entries=50)  # Cache for 2 hours, max 50 entries
def get_cached_stock_data(symbol, period):
    """Enhanced caching for stock data with longer TTL"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        stock_info = stock.info
        
        if len(data) < 50:
            raise ValueError("Insufficient data")
            
        return data, stock_info, True
        
    except Exception as e:
        return None, None, False

@st.cache_data(ttl=3600, max_entries=20)
def get_cached_technical_features(data_hash):
    """Cache technical feature calculations"""
    # This will be called by create_technical_features
    return None

class StockPredictorApp:
    """Optimized application class for stock prediction with enhanced caching"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.data = None
        self.predictions = None
        self.stock_info = None
        self._model_cache = {}
        
    def fetch_data(self, symbol, period="2y"):
        """Fetch stock data with enhanced caching"""
        # Use the cached function
        return get_cached_stock_data(symbol, period)
    
    @functools.lru_cache(maxsize=10)
    def _calculate_indicators_cached(self, data_tuple, length):
        """Cache expensive indicator calculations"""
        # Convert tuple back to series for calculations
        closes = pd.Series(data_tuple)
        
        # RSI calculation
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def create_technical_features(self, data):
        """Optimized technical indicators with caching"""
        # Create cache key for this dataset
        data_hash = create_cache_key(data['Close'].values.tobytes(), len(data))
        
        # Try to load from cache first
        cached_result = load_from_cache(f"features_{data_hash}")
        if cached_result is not None:
            return cached_result
        
        df = data.copy()
        
        # Vectorized operations for better performance
        closes = df['Close']
        volumes = df['Volume']
        
        # Price-based features (vectorized)
        df['Returns'] = closes.pct_change()
        df['Price_MA_5'] = closes.rolling(5, min_periods=1).mean()
        df['Price_MA_20'] = closes.rolling(20, min_periods=1).mean()
        df['Price_MA_50'] = closes.rolling(50, min_periods=1).mean()
        
        # Moving average ratios
        df['MA_Ratio_5_20'] = df['Price_MA_5'] / df['Price_MA_20']
        df['MA_Ratio_20_50'] = df['Price_MA_20'] / df['Price_MA_50']
        df['Price_to_MA20'] = closes / df['Price_MA_20']
        
        # Vectorized volatility features
        returns = df['Returns']
        df['Volatility_5'] = returns.rolling(5, min_periods=1).std()
        df['Volatility_20'] = returns.rolling(20, min_periods=1).std()
        
        # Volume features
        df['Volume_MA_20'] = volumes.rolling(20, min_periods=1).mean()
        df['Volume_Ratio'] = volumes / df['Volume_MA_20']
        
        # Optimized RSI calculation
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Normalized'] = (df['RSI'] - 50) / 50
        
        # Optimized MACD
        ema_12 = closes.ewm(span=12, min_periods=1).mean()
        ema_26 = closes.ewm(span=26, min_periods=1).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Optimized Bollinger Bands
        bb_middle = closes.rolling(20, min_periods=1).mean()
        bb_std = closes.rolling(20, min_periods=1).std()
        df['BB_Upper'] = bb_middle + (bb_std * 2)
        df['BB_Lower'] = bb_middle - (bb_std * 2)
        df['BB_Position'] = (closes - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Vectorized lagged features
        for lag in [1, 2, 3, 5]:
            df[f'Returns_Lag_{lag}'] = returns.shift(lag)
            df[f'Volume_Ratio_Lag_{lag}'] = df['Volume_Ratio'].shift(lag)
        
        # Target variable
        df['Target'] = returns.shift(-1)
        
        # Feature selection
        self.feature_columns = [
            'MA_Ratio_5_20', 'MA_Ratio_20_50', 'Price_to_MA20',
            'Volatility_5', 'Volatility_20', 'Volume_Ratio',
            'RSI_Normalized', 'MACD', 'MACD_Histogram', 'BB_Position',
            'Returns_Lag_1', 'Returns_Lag_2', 'Returns_Lag_3', 'Returns_Lag_5',
            'Volume_Ratio_Lag_1', 'Volume_Ratio_Lag_2', 'Volume_Ratio_Lag_3'
        ]
        
        # Save to cache
        save_to_cache(f"features_{data_hash}", df)
        
        return df
    
    def train_model(self, data):
        """Optimized machine learning model training with parallel processing"""
        # Create cache key for model training
        features_hash = create_cache_key(
            [data[col].values.tobytes() for col in self.feature_columns if col in data.columns],
            len(data)
        )
        
        # Check model cache first
        cached_model = load_from_cache(f"model_{features_hash}")
        if cached_model is not None:
            self.model, self.scaler, model_results, best_name, best_score = cached_model
            return model_results, best_name, best_score
        
        # Prepare data
        feature_data = data[self.feature_columns + ['Target']].dropna()
        X = feature_data[self.feature_columns]
        y = feature_data['Target']
        
        # Optimized models with reduced complexity for faster training
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=50, max_depth=8, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1  # Use all CPU cores
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=50, max_depth=5, learning_rate=0.15, random_state=42
            ),
            'Linear Regression': LinearRegression(n_jobs=-1)
        }
        
        # Reduced cross-validation for faster execution
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced from 5 to 3
        best_score = -np.inf
        best_model = None
        best_name = ""
        
        model_results = {}
        
        # Parallel model evaluation using ThreadPoolExecutor
        def evaluate_model(name_model_pair):
            name, model = name_model_pair
            try:
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2', n_jobs=1)
                return name, cv_scores.mean(), cv_scores.std(), model
            except Exception as e:
                return name, -1, 1, model
        
        # Use parallel processing for model evaluation
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(evaluate_model, models.items()))
        
        for name, mean_score, std_score, model in results:
            model_results[name] = {
                'mean_score': mean_score,
                'std_score': std_score,
                'model': model
            }
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_name = name
        
        # Train best model
        X_scaled = self.scaler.fit_transform(X)
        best_model.fit(X_scaled, y)
        
        self.model = best_model
        
        # Cache the trained model
        model_cache_data = (best_model, self.scaler, model_results, best_name, best_score)
        save_to_cache(f"model_{features_hash}", model_cache_data)
        
        return model_results, best_name, best_score
    
    def make_predictions(self, data, days=30):
        """Generate predictions"""
        last_features = data[self.feature_columns].iloc[-1:].values
        last_features_scaled = self.scaler.transform(last_features)
        predicted_return = self.model.predict(last_features_scaled)[0]
        
        current_price = data['Close'].iloc[-1]
        predictions = []
        price = current_price
        
        for i in range(days):
            daily_volatility = data['Returns'].std()
            
            if i == 0:
                return_pred = predicted_return
            else:
                decay_factor = 0.95 ** i
                return_pred = predicted_return * decay_factor + np.random.normal(0, daily_volatility * 0.5)
            
            price = price * (1 + return_pred)
            predictions.append(price)
        
        last_date = data.index[-1]
        prediction_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        return pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Price': predictions
        })

@st.cache_data(ttl=1800, max_entries=20)  # Cache charts for 30 minutes
def create_interactive_charts(data_hash, recent_data_dict, predictions_dict, symbol):
    """Create enhanced interactive Plotly charts with caching"""
    
    # Convert dictionaries back to DataFrames
    recent_data = pd.DataFrame(recent_data_dict)
    recent_data.index = pd.to_datetime(recent_data.index)
    
    predictions = pd.DataFrame(predictions_dict) if predictions_dict else None
    if predictions is not None:
        predictions['Date'] = pd.to_datetime(predictions['Date'])
    
    # Main price chart with predictions
    fig_main = go.Figure()
    
    # Historical prices with optimized rendering
    fig_main.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2),  # Reduced line width for performance
        hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
    ))
    
    # Moving averages with optimized styling
    fig_main.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['Price_MA_20'],
        mode='lines',
        name='20-day MA',
        line=dict(color='orange', width=1, dash='dash'),  # Thinner lines
        hovertemplate='<b>20-day MA</b>: $%{y:.2f}<extra></extra>'
    ))
    
    if 'Price_MA_50' in recent_data.columns:
        fig_main.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['Price_MA_50'],
            mode='lines',
            name='50-day MA',
            line=dict(color='purple', width=1, dash='dot'),
            hovertemplate='<b>50-day MA</b>: $%{y:.2f}<extra></extra>'
        ))
    
    # Enhanced predictions with confidence bands
    if predictions is not None:
        current_price = recent_data['Close'].iloc[-1]
        
        # Simplified prediction line for performance
        fig_main.add_trace(go.Scatter(
            x=predictions['Date'],
            y=predictions['Predicted_Price'],
            mode='lines+markers',
            name='🔮 AI Predictions',
            line=dict(color='red', width=2),
            marker=dict(size=4, color='red'),  # Smaller markers
            hovertemplate='<b>Predicted Date</b>: %{x}<br><b>Predicted Price</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Add current price line
        fig_main.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="bottom right"
        )
    
    # Optimized layout
    fig_main.update_layout(
        title=f'📈 {symbol} - Price Analysis & AI Predictions',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=500,  # Reduced height for performance
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig_main

def create_prediction_timeline_chart(predictions, current_price, symbol):
    """Create a detailed prediction timeline chart"""
    if predictions is None:
        return None
    
    fig = go.Figure()
    
    # Calculate percentage changes
    pct_changes = [(price - current_price) / current_price * 100 for price in predictions['Predicted_Price']]
    
    # Color code based on performance
    colors = ['green' if change > 0 else 'red' for change in pct_changes]
    
    # Main prediction line
    fig.add_trace(go.Scatter(
        x=predictions['Date'],
        y=pct_changes,
        mode='lines+markers',
        name='Expected Return %',
        line=dict(color='blue', width=3),
        marker=dict(size=8, color=colors),
        hovertemplate='<b>Date</b>: %{x}<br><b>Expected Return</b>: %{y:.2f}%<br><b>Price Target</b>: $%{customdata:.2f}<extra></extra>',
        customdata=predictions['Predicted_Price']
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
    
    # Add target zones
    fig.add_hrect(y0=5, y1=20, fillcolor="green", opacity=0.1, annotation_text="Strong Buy Zone", annotation_position="top right")
    fig.add_hrect(y0=-20, y1=-5, fillcolor="red", opacity=0.1, annotation_text="Strong Sell Zone", annotation_position="bottom right")
    
    fig.update_layout(
        title=f'🎯 {symbol} - Prediction Timeline (% Change from Current Price)',
        xaxis_title='Date',
        yaxis_title='Expected Return (%)',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_weekly_targets_chart(predictions, current_price, symbol):
    """Create weekly price targets chart"""
    if predictions is None:
        return None
    
    # Get weekly data points
    weekly_data = predictions.iloc[::7].head(8)  # Every 7 days, max 8 weeks
    weeks = [f"Week {i+1}" for i in range(len(weekly_data))]
    
    fig = go.Figure()
    
    # Calculate changes from current price
    changes = [(price - current_price) / current_price * 100 for price in weekly_data['Predicted_Price']]
    
    # Color bars based on positive/negative
    colors = ['rgba(0,128,0,0.7)' if change > 0 else 'rgba(255,0,0,0.7)' for change in changes]
    
    fig.add_trace(go.Bar(
        x=weeks,
        y=weekly_data['Predicted_Price'],
        name='Price Target',
        marker_color=colors,
        hovertemplate='<b>%{x}</b><br><b>Price Target</b>: $%{y:.2f}<br><b>Change</b>: %{customdata:.1f}%<extra></extra>',
        customdata=changes
    ))
    
    # Add current price line
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Current: ${current_price:.2f}"
    )
    
    fig.update_layout(
        title=f'📅 {symbol} - Weekly Price Targets',
        xaxis_title='Time Period',
        yaxis_title='Price Target ($)',
        height=400,
        template='plotly_white'
    )
    
    return fig

def calculate_advanced_risk_metrics(data, predictions, current_price):
    """Calculate comprehensive risk metrics for enhanced analysis"""
    metrics = {}
    
    if data is not None and 'Returns' in data.columns:
        returns = data['Returns'].dropna()
        
        # Historical metrics
        metrics['historical_volatility'] = returns.std() * np.sqrt(252) * 100
        metrics['avg_annual_return'] = returns.mean() * 252 * 100
        
        # Downside deviation (volatility of negative returns only)
        negative_returns = returns[returns < 0]
        metrics['downside_deviation'] = negative_returns.std() * np.sqrt(252) * 100 if len(negative_returns) > 0 else 0
        
        # Maximum drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        metrics['max_drawdown'] = abs(drawdown.min()) * 100
        
        # Sortino ratio (return per unit of downside risk)
        if metrics['downside_deviation'] > 0:
            metrics['sortino_ratio'] = (metrics['avg_annual_return'] - 2) / metrics['downside_deviation']
        else:
            metrics['sortino_ratio'] = 0
        
        # Calmar ratio (annual return / max drawdown)
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = metrics['avg_annual_return'] / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = 0
        
        # VaR calculations (95% and 99% confidence)
        metrics['var_95'] = np.percentile(returns, 5) * 100
        metrics['var_99'] = np.percentile(returns, 1) * 100
        
        # Expected shortfall (average of worst 5% returns)
        worst_5_percent = returns[returns <= np.percentile(returns, 5)]
        metrics['expected_shortfall'] = worst_5_percent.mean() * 100 if len(worst_5_percent) > 0 else 0
        
    # Prediction-based metrics
    if predictions is not None:
        predicted_returns = (predictions['Predicted_Price'] / current_price - 1) * 100
        metrics['prediction_volatility'] = predicted_returns.std()
        metrics['prediction_skewness'] = predicted_returns.skew()
        metrics['prediction_kurtosis'] = predicted_returns.kurtosis()
        
        # Probability metrics
        metrics['prob_gain'] = (predicted_returns > 0).mean() * 100
        metrics['prob_loss_5'] = (predicted_returns < -5).mean() * 100
        metrics['prob_gain_10'] = (predicted_returns > 10).mean() * 100
        
        # Expected values
        metrics['expected_gain'] = predicted_returns[predicted_returns > 0].mean() if (predicted_returns > 0).any() else 0
        metrics['expected_loss'] = predicted_returns[predicted_returns < 0].mean() if (predicted_returns < 0).any() else 0
    
    return metrics

def create_risk_return_gauge(predictions, current_price, volatility, data=None, prediction_days=30):
    """Create enhanced risk-return gauge chart with more accurate metrics"""
    if predictions is None:
        return None
    
    # Calculate more accurate return metrics
    final_return = (predictions['Predicted_Price'].iloc[-1] - current_price) / current_price * 100
    max_return = (predictions['Predicted_Price'].max() - current_price) / current_price * 100
    min_return = (predictions['Predicted_Price'].min() - current_price) / current_price * 100
    
    # Calculate probability of positive return
    positive_predictions = (predictions['Predicted_Price'] > current_price).sum()
    probability_of_gain = (positive_predictions / len(predictions)) * 100
    
    # Calculate Value at Risk (VaR) - 5% worst case scenario
    returns_distribution = predictions['Predicted_Price'].pct_change().dropna()
    if len(returns_distribution) > 0:
        var_5_percent = np.percentile(returns_distribution, 5) * 100
    else:
        var_5_percent = min_return
    
    # Enhanced risk calculation using multiple factors
    if data is not None:
        # Historical volatility (annualized)
        historical_vol = data['Returns'].std() * np.sqrt(252) * 100 if 'Returns' in data.columns else volatility
        
        # Maximum drawdown calculation
        rolling_max = data['Close'].expanding().max()
        drawdown = (data['Close'] - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdown.min())
        
        # Sharpe ratio approximation (using 2% risk-free rate)
        avg_return = data['Returns'].mean() * 252 * 100 if 'Returns' in data.columns else 0
        sharpe_ratio = (avg_return - 2) / historical_vol if historical_vol > 0 else 0
        
        # Combined risk score (0-100)
        volatility_score = min(50, historical_vol * 1.5)  # Cap at 50
        drawdown_score = min(30, max_drawdown * 1.2)      # Cap at 30
        prediction_uncertainty = min(20, abs(max_return - min_return) * 0.5)  # Cap at 20
        
        risk_score = min(100, volatility_score + drawdown_score + prediction_uncertainty)
    else:
        risk_score = min(100, volatility * 5)
        sharpe_ratio = 0
        max_drawdown = 0
    
    # Create enhanced subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=['Expected Return', 'Risk Level', 'Success Probability', 'Risk-Adjusted Return']
    )
    
    # Expected Return gauge (dynamic range based on predictions)
    return_range = max(50, abs(max_return) * 1.5, abs(min_return) * 1.5)
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=final_return,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{prediction_days}-Day Return (%)"},
        delta={'reference': 0},
        gauge={'axis': {'range': [-return_range, return_range]},
               'bar': {'color': "green" if final_return > 0 else "red"},
               'steps': [
                   {'range': [-return_range, -5], 'color': "lightcoral"},
                   {'range': [-5, 5], 'color': "lightgray"},
                   {'range': [5, return_range], 'color': "lightgreen"}],
               'threshold': {'line': {'color': "blue", 'width': 3},
                           'thickness': 0.75, 'value': 0}}), row=1, col=1)
    
    # Enhanced Risk gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Comprehensive Risk Score"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if risk_score > 70 else "orange" if risk_score > 40 else "green"},
               'steps': [
                   {'range': [0, 25], 'color': "lightgreen"},
                   {'range': [25, 50], 'color': "yellow"},
                   {'range': [50, 75], 'color': "orange"},
                   {'range': [75, 100], 'color': "lightcoral"}],
               'threshold': {'line': {'color': "darkred", 'width': 4},
                           'thickness': 0.75, 'value': 80}}), row=1, col=2)
    
    # Success Probability gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=probability_of_gain,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probability of Profit (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "green" if probability_of_gain > 60 else "orange" if probability_of_gain > 40 else "red"},
               'steps': [
                   {'range': [0, 30], 'color': "lightcoral"},
                   {'range': [30, 50], 'color': "yellow"},
                   {'range': [50, 70], 'color': "lightgreen"},
                   {'range': [70, 100], 'color': "green"}],
               'threshold': {'line': {'color': "blue", 'width': 3},
                           'thickness': 0.75, 'value': 50}}), row=2, col=1)
    
    # Risk-Adjusted Return (Sharpe Ratio visualization)
    sharpe_display = max(-3, min(3, sharpe_ratio))  # Clamp between -3 and 3
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sharpe_display,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sharpe Ratio (Risk-Adjusted)"},
        gauge={'axis': {'range': [-3, 3]},
               'bar': {'color': "green" if sharpe_display > 1 else "orange" if sharpe_display > 0 else "red"},
               'steps': [
                   {'range': [-3, 0], 'color': "lightcoral"},
                   {'range': [0, 1], 'color': "yellow"},
                   {'range': [1, 2], 'color': "lightgreen"},
                   {'range': [2, 3], 'color': "green"}],
               'threshold': {'line': {'color': "blue", 'width': 3},
                           'thickness': 0.75, 'value': 1}}), row=2, col=2)
    
    fig.update_layout(height=500, title_text="📊 Advanced Risk & Return Analysis")
    
    return fig, {
        'expected_return': final_return,
        'max_return': max_return,
        'min_return': min_return,
        'risk_score': risk_score,
        'probability_of_gain': probability_of_gain,
        'var_5_percent': var_5_percent,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown if data is not None else 0
    }

def create_technical_indicators_chart(data):
    """Create technical indicators chart"""
    try:
        recent_data = data.tail(100)
        
        # Check if required columns exist
        required_cols = ['RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in recent_data.columns]
        
        if missing_cols:
            st.error(f"Missing technical indicator columns: {missing_cols}")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=('RSI', 'MACD', 'Bollinger Bands', 'Volume'),
            vertical_spacing=0.05,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        # RSI
        if 'RSI' in recent_data.columns and not recent_data['RSI'].isna().all():
            fig.add_trace(go.Scatter(
                x=recent_data.index, y=recent_data['RSI'],
                name='RSI', line=dict(color='purple')
            ), row=1, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        if 'MACD' in recent_data.columns and 'MACD_Signal' in recent_data.columns:
            if not recent_data['MACD'].isna().all():
                fig.add_trace(go.Scatter(
                    x=recent_data.index, y=recent_data['MACD'],
                    name='MACD', line=dict(color='blue')
                ), row=2, col=1)
            if not recent_data['MACD_Signal'].isna().all():
                fig.add_trace(go.Scatter(
                    x=recent_data.index, y=recent_data['MACD_Signal'],
                    name='Signal', line=dict(color='red')
                ), row=2, col=1)
        
        # Bollinger Bands
        if 'Close' in recent_data.columns and not recent_data['Close'].isna().all():
            fig.add_trace(go.Scatter(
                x=recent_data.index, y=recent_data['Close'],
                name='Price', line=dict(color='black')
            ), row=3, col=1)
            
        if 'BB_Upper' in recent_data.columns and not recent_data['BB_Upper'].isna().all():
            fig.add_trace(go.Scatter(
                x=recent_data.index, y=recent_data['BB_Upper'],
                name='Upper Band', line=dict(color='gray', dash='dash')
            ), row=3, col=1)
            
        if 'BB_Lower' in recent_data.columns and not recent_data['BB_Lower'].isna().all():
            fig.add_trace(go.Scatter(
                x=recent_data.index, y=recent_data['BB_Lower'],
                name='Lower Band', line=dict(color='gray', dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
            ), row=3, col=1)
        
        # Volume
        if 'Volume' in recent_data.columns and not recent_data['Volume'].isna().all():
            fig.add_trace(go.Bar(
                x=recent_data.index, y=recent_data['Volume'],
                name='Volume', marker_color='lightblue'
            ), row=4, col=1)
        
        fig.update_layout(height=800, showlegend=False)
        return fig
        
    except Exception as e:
        st.error(f"Error creating technical indicators chart: {str(e)}")
        st.error(f"Available columns: {list(data.columns)}")
        return None

def main():
    # Performance monitoring
    start_time = time.time()
    
    # Simple app header
    st.markdown('<h1 class="main-header">🚀 Smart Stock Analyzer</h1>', unsafe_allow_html=True)
    
    # Add performance info in sidebar
    with st.sidebar:
        if 'last_analysis_time' in st.session_state:
            st.info(f"⚡ Last analysis: {st.session_state.last_analysis_time:.1f}s")
    
    # Sidebar with beginner explanations
    st.sidebar.header("🎯 Get Started Here")
    
    # Add beginner help section (collapsed by default for performance)
    with st.sidebar.expander("❓ New to Stocks? Click Here!"):
        st.markdown("""
        **Stock Symbol**: A short code for a company (e.g., AAPL = Apple Inc.)
        
        **Popular Stock Symbols:**
        - 🍎 AAPL (Apple)
        - 🚗 TSLA (Tesla)  
        - 💻 MSFT (Microsoft)
        - 🛒 AMZN (Amazon)
        - 🔍 GOOGL (Google)
        
        **Data Period**: How far back to look at the stock's history
        
        **Prediction Days**: How many days into the future to predict
        """)
    
    # Stock symbol input with more guidance
    st.sidebar.markdown("### 📈 Choose a Stock to Analyze")
    symbol = st.sidebar.text_input(
        "Stock Symbol (Ticker)",
        value="AAPL",
        help="💡 Try AAPL, TSLA, MSFT, AMZN, or GOOGL for popular stocks",
        placeholder="e.g., AAPL"
    ).upper().strip()
    
    # Time period selection with explanations
    st.sidebar.markdown("### 📅 Analysis Time Period")
    period_options = {
        "6 months (Recent trends)": "6mo",
        "1 year (Full year view)": "1y", 
        "2 years (Medium-term patterns)": "2y",
        "5 years (Long-term trends)": "5y"
    }
    period_display = st.sidebar.selectbox(
        "How much history to analyze?", 
        list(period_options.keys()), 
        index=1,  # Default to 1 year for better performance
        help="📊 More history = better AI predictions, but 1-2 years is usually optimal"
    )
    period = period_options[period_display]
    
    # Prediction days with beginner explanation
    st.sidebar.markdown("### 🔮 Prediction Timeline")
    prediction_days = st.sidebar.slider(
        "Predict how many days ahead?", 
        7, 30, 15,  # Reduced max and default for performance
        help="🎯 Shorter predictions are more accurate and faster to compute!"
    )
    
    # Advanced settings (collapsed by default)
    with st.sidebar.expander("⚙️ Advanced Settings"):
        enable_detailed_charts = st.checkbox("Enable detailed charts", value=True)
        enable_risk_analysis = st.checkbox("Enable risk analysis", value=True)
        enable_technical_indicators = st.checkbox("Enable technical indicators", value=False)  # Disabled by default
    
    # Analysis button
    analyze_button = st.sidebar.button("🚀 Start Analysis", type="primary")
    
    # Add educational sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Quick Tips")
    st.sidebar.info("""
    **Green signals** 🟢 = Good time to consider buying
    
    **Red signals** 🔴 = Might want to avoid or sell
    
    **Yellow signals** 🟡 = Wait and see
    
    💡 **Remember**: This is analysis, not financial advice!
    """)
    
    # Initialize predictor with session state for persistence
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StockPredictorApp()
    predictor = st.session_state.predictor
    
    if analyze_button and symbol:
        analysis_start = time.time()
        
        # Progress bar with friendly messages
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data
        status_text.text("📊 Getting stock data from the market...")
        progress_bar.progress(20)
        
        data, stock_info, success = predictor.fetch_data(symbol, period)
        
        if not success:
            st.error("❌ Oops! We couldn't find that stock symbol. Try checking the spelling or use a popular one like AAPL, TSLA, or MSFT.")
            return
        
        predictor.data = data
        predictor.stock_info = stock_info
        
        # Create features
        status_text.text("🔬 Analyzing price patterns and trends...")
        progress_bar.progress(40)
        
        enhanced_data = predictor.create_technical_features(data)
        
        # Train model
        status_text.text("🤖 AI is learning from historical data...")
        progress_bar.progress(60)
        
        model_results, best_model_name, best_score = predictor.train_model(enhanced_data)
        
        # Make predictions
        status_text.text("🔮 Creating your personalized prediction...")
        progress_bar.progress(80)
        
        predictions = predictor.make_predictions(enhanced_data, prediction_days)
        
        progress_bar.progress(100)
        status_text.text("✅ Your analysis is ready!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Store analysis time
        analysis_time = time.time() - analysis_start
        st.session_state.last_analysis_time = analysis_time
        
        # Display results with celebration
        st.balloons()
        st.success(f"🎉 Analysis complete in {analysis_time:.1f}s! Here's what our AI found...")
        
        # Company information with explanations
        st.markdown("### 🏢 About This Company")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            daily_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)
            st.metric(
                "Today's Stock Price",
                f"${data['Close'].iloc[-1]:.2f}",
                f"{daily_change:+.2f}%",
                help="💡 Current share price and daily change"
            )
        
        with col2:
            company_name = stock_info.get('longName', 'N/A') if stock_info else 'N/A'
            display_name = company_name[:25] + "..." if len(company_name) > 25 else company_name
            st.metric(
                "Company Name", 
                display_name,
                help="🏢 The company you're analyzing"
            )
        
        with col3:
            sector = stock_info.get('sector', 'N/A') if stock_info else 'N/A'
            st.metric(
                "Business Type", 
                sector,
                help="🏭 Industry sector"
            )
        
        # Prediction summary with beginner explanations
        current_price = data['Close'].iloc[-1]
        predicted_price = predictions['Predicted_Price'].iloc[-1]
        expected_return = (predicted_price - current_price) / current_price * 100
        
        st.markdown("### 🔮 AI Prediction Summary")
        
        # Create recommendation box
        if expected_return > 10:
            recommendation = "🟢 **Strong Positive Signal** - AI sees good potential!"
        elif expected_return > 2:
            recommendation = "🟢 **Positive Signal** - Looks promising!"
        elif expected_return > -2:
            recommendation = "🟡 **Neutral Signal** - Wait and see approach"
        elif expected_return > -10:
            recommendation = "🟠 **Caution Signal** - Be careful"
        else:
            recommendation = "🔴 **Negative Signal** - AI suggests avoiding"
        
        st.markdown(f"#### {recommendation}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                f"Predicted Price ({prediction_days} days)",
                f"${predicted_price:.2f}",
                f"{expected_return:+.1f}%",
                help=f"💡 AI prediction for {prediction_days} days ahead"
            )
        
        with col2:
            confidence = max(0, min(1, (best_score + 1) / 2))
            confidence_text = "High 🎯" if confidence > 0.7 else "Medium 🤔" if confidence > 0.5 else "Low ⚠️"
            st.metric(
                "AI Confidence Level", 
                f"{confidence:.0%}",
                confidence_text,
                help="💡 How confident our AI is"
            )
        
        with col3:
            volatility = enhanced_data['Returns'].std() * np.sqrt(252) * 100
            vol_text = "Very Risky 🌋" if volatility > 40 else "Risky 🌊" if volatility > 25 else "Moderate 📊" if volatility > 15 else "Stable 🏔️"
            st.metric(
                "Risk Level", 
                f"{volatility:.0f}%",
                vol_text,
                help="💡 Price volatility measure"
            )
        
        with col4:
            if expected_return > 5:
                signal = "🟢 BUY SIGNAL"
            elif expected_return < -5:
                signal = "🔴 SELL SIGNAL"
            else:
                signal = "🟡 HOLD/WAIT"
            
            st.metric("Action Suggestion", signal)
        
        # Conditional rendering based on user preferences
        if enable_detailed_charts:
            # Main chart with optimized rendering
            st.markdown("### � Interactive Stock Chart")
            
            recent_data = enhanced_data.tail(50)  # Reduced data for performance
            
            # Convert to serializable format for caching
            recent_data_dict = {
                'Close': recent_data['Close'].to_dict(),
                'Price_MA_20': recent_data['Price_MA_20'].to_dict() if 'Price_MA_20' in recent_data.columns else {},
                'Price_MA_50': recent_data['Price_MA_50'].to_dict() if 'Price_MA_50' in recent_data.columns else {}
            }
            recent_data_dict = {str(k): v for k, v in recent_data_dict.items()}
            
            predictions_dict = predictions.to_dict() if predictions is not None else None
            if predictions_dict:
                predictions_dict = {str(k): v for k, v in predictions_dict.items()}
            
            data_hash = create_cache_key(str(recent_data_dict), str(predictions_dict), symbol)
            
            main_chart = create_interactive_charts(data_hash, recent_data_dict, predictions_dict, symbol)
            st.plotly_chart(main_chart, use_container_width=True)
        
        # Conditional risk analysis
        if enable_risk_analysis:
            # Risk-Return Analysis with explanation
            st.markdown("### ⚖️ Advanced Investment Risk Assessment")
            st.markdown("*Comprehensive risk analysis using professional financial metrics*")
            
            volatility = enhanced_data['Returns'].std() * np.sqrt(252) * 100
            risk_return_result = create_risk_return_gauge(predictions, current_price, volatility, enhanced_data, prediction_days)
            
            if risk_return_result:
                risk_return_chart, risk_metrics = risk_return_result
                st.plotly_chart(risk_return_chart, use_container_width=True)
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Expected Return", f"{risk_metrics['expected_return']:.1f}%")
                with col2:
                    st.metric("Success Probability", f"{risk_metrics['probability_of_gain']:.0f}%")
                with col3:
                    st.metric("Worst Case (5% VaR)", f"{risk_metrics['var_5_percent']:.1f}%")
                with col4:
                    st.metric("Risk-Adjusted Score", f"{risk_metrics['sharpe_ratio']:.2f}")
        
        # Performance summary
        total_time = time.time() - start_time
        st.sidebar.success(f"⚡ Total time: {total_time:.1f}s")
        
        # New comprehensive prediction charts with explanations
        st.markdown("### 🎯 Future Predictions Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Day-by-Day Predictions")
            st.markdown("*See exactly what price our AI predicts for each day*")
            timeline_chart = create_prediction_timeline_chart(predictions, current_price, symbol)
            if timeline_chart:
                st.plotly_chart(timeline_chart, use_container_width=True)
            else:
                st.info("📊 Chart will appear here after analysis")
        
        with col2:
            st.markdown("#### 🎯 Weekly Investment Targets")
            st.markdown("*Perfect for planning your investment strategy*")
            weekly_chart = create_weekly_targets_chart(predictions, current_price, symbol)
            if weekly_chart:
                st.plotly_chart(weekly_chart, use_container_width=True)
            else:
                st.info("📊 Chart will appear here after analysis")
        
        # Risk-Return Analysis with explanation
        st.markdown("### ⚖️ Advanced Investment Risk Assessment")
        st.markdown("*Comprehensive analysis of risk vs reward potential using professional financial metrics*")
        
        volatility = enhanced_data['Returns'].std() * np.sqrt(252) * 100
        risk_return_result = create_risk_return_gauge(predictions, current_price, volatility, enhanced_data, prediction_days)
        
        if risk_return_result:
            risk_return_chart, risk_metrics = risk_return_result
            st.plotly_chart(risk_return_chart, use_container_width=True)
            
            # Display key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Expected Return",
                    f"{risk_metrics['expected_return']:.1f}%",
                    help="💡 The predicted return for your investment period"
                )
                
            with col2:
                st.metric(
                    "Success Probability", 
                    f"{risk_metrics['probability_of_gain']:.0f}%",
                    help="💡 Likelihood that the investment will be profitable"
                )
                
            with col3:
                st.metric(
                    "Worst Case (5% VaR)",
                    f"{risk_metrics['var_5_percent']:.1f}%",
                    help="💡 5% chance of losing more than this amount"
                )
                
            with col4:
                st.metric(
                    "Risk-Adjusted Score",
                    f"{risk_metrics['sharpe_ratio']:.2f}",
                    help="💡 Return per unit of risk (higher is better)"
                )
            
            # Enhanced risk explanation based on multiple factors
            risk_score = risk_metrics['risk_score']
            if risk_score > 75:
                risk_explanation = "� **Very High Risk**: Extreme volatility with significant potential for large gains or losses. Only suitable for experienced investors with high risk tolerance."
            elif risk_score > 60:
                risk_explanation = "🔥 **High Risk**: Substantial price swings expected. Potential for good returns but also significant losses. Requires careful monitoring."
            elif risk_score > 40:
                risk_explanation = "🌊 **Medium Risk**: Moderate price fluctuations with balanced risk-reward profile. Suitable for most investors with some experience."
            elif risk_score > 25:
                risk_explanation = "📊 **Low-Medium Risk**: Relatively stable with occasional fluctuations. Good for conservative growth strategies."
            else:
                risk_explanation = "🏔️ **Low Risk**: Very stable price movements with predictable patterns. Ideal for conservative investors."
            
            # Add probability-based recommendation
            prob_gain = risk_metrics['probability_of_gain']
            if prob_gain > 70:
                probability_text = "🎯 **High confidence** in positive outcomes"
            elif prob_gain > 50:
                probability_text = "⚖️ **Moderate confidence** - balanced probability"
            else:
                probability_text = "⚠️ **Low confidence** - higher chance of losses"
                
            st.info(f"💡 **Risk Assessment**: {risk_explanation}")
            st.info(f"🎲 **Success Likelihood**: {probability_text}")
            
            # Add advanced risk metrics explanation
            with st.expander("🎓 Understanding Advanced Risk Metrics"):
                st.markdown(f"""
                **Comprehensive Risk Score ({risk_score:.0f}/100)**: Combines multiple risk factors:
                - Historical volatility patterns
                - Maximum historical losses (drawdown)
                - Prediction uncertainty range
                
                **Success Probability ({prob_gain:.0f}%)**: Based on how many prediction scenarios show profits
                - Above 70% = High confidence 🎯
                - 50-70% = Moderate confidence ⚖️
                - Below 50% = High uncertainty ⚠️
                
                **Value at Risk (VaR)**: There's a 5% chance you could lose more than {risk_metrics['var_5_percent']:.1f}%
                
                **Sharpe Ratio ({risk_metrics['sharpe_ratio']:.2f})**: Risk-adjusted return measure
                - Above 1.0 = Good risk-adjusted returns ✅
                - 0.5-1.0 = Acceptable returns 👍
                - Below 0.5 = Poor risk-adjusted returns ⚠️
                
                **Maximum Drawdown**: Historical worst loss period was {risk_metrics['max_drawdown']:.1f}%
                """)
            
            # Additional comprehensive risk analysis
            st.markdown("#### 📊 Detailed Risk Analysis")
            detailed_metrics = calculate_advanced_risk_metrics(enhanced_data, predictions, current_price)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Return Analysis**")
                if 'avg_annual_return' in detailed_metrics:
                    st.write(f"• Historical Annual Return: {detailed_metrics['avg_annual_return']:.1f}%")
                st.write(f"• Expected Return Range: {risk_metrics['min_return']:.1f}% to {risk_metrics['max_return']:.1f}%")
                if 'expected_gain' in detailed_metrics:
                    st.write(f"• Average Gain (when positive): {detailed_metrics['expected_gain']:.1f}%")
                    st.write(f"• Average Loss (when negative): {detailed_metrics['expected_loss']:.1f}%")
                
                st.markdown("**🎯 Probability Analysis**")
                if 'prob_gain_10' in detailed_metrics:
                    st.write(f"• Chance of >10% gain: {detailed_metrics['prob_gain_10']:.0f}%")
                    st.write(f"• Chance of >5% loss: {detailed_metrics['prob_loss_5']:.0f}%")
                st.write(f"• Overall success rate: {risk_metrics['probability_of_gain']:.0f}%")
            
            with col2:
                st.markdown("**⚠️ Risk Measures**")
                if 'downside_deviation' in detailed_metrics:
                    st.write(f"• Downside Volatility: {detailed_metrics['downside_deviation']:.1f}%")
                    st.write(f"• Sortino Ratio: {detailed_metrics['sortino_ratio']:.2f}")
                if 'var_95' in detailed_metrics:
                    st.write(f"• 1-Day VaR (95%): {detailed_metrics['var_95']:.2f}%")
                    st.write(f"• Expected Shortfall: {detailed_metrics['expected_shortfall']:.2f}%")
                
                st.markdown("**📊 Risk-Adjusted Performance**")
                if 'calmar_ratio' in detailed_metrics:
                    st.write(f"• Calmar Ratio: {detailed_metrics['calmar_ratio']:.2f}")
                st.write(f"• Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
                st.write(f"• Risk Score: {risk_score:.0f}/100")
        else:
            st.info("📊 Advanced risk analysis will appear here after analysis")
        
        # Technical indicators with beginner explanation
        st.markdown("### 📊 Market Signals & Indicators")
        st.markdown("*Professional trading signals that help predict price movements*")
        
        with st.expander("🎓 Learn about these market indicators"):
            st.markdown("""
            **RSI (Relative Strength)**: Shows if a stock is "overbought" (might go down) or "oversold" (might go up)
            - Above 70 = Might be too expensive right now 📈
            - Below 30 = Might be a good buying opportunity 📉
            
            **MACD (Moving Average)**: Compares short and long-term price trends
            - Lines crossing up = Positive momentum 🟢
            - Lines crossing down = Negative momentum 🔴
            
            **Bollinger Bands**: Shows if a price is unusually high or low
            - Price near top band = Might be overpriced 🔴
            - Price near bottom band = Might be underpriced 🟢
            
            **Volume**: How many shares are being traded
            - High volume + price up = Strong buying interest 💪
            - High volume + price down = Strong selling pressure ⚠️
            """)
        
        tech_chart = create_technical_indicators_chart(enhanced_data)
        if tech_chart:
            st.plotly_chart(tech_chart, use_container_width=True)
        else:
            st.info("📊 Technical indicators will appear here once we have enough data to analyze.")
        
        # Model performance with beginner explanation
        st.markdown("### 🤖 How Accurate is Our AI?")
        st.markdown("*See how well our different AI models performed in testing*")
        
        model_df = pd.DataFrame({
            'AI Model': list(model_results.keys()),
            'Accuracy Score': [f"{results['mean_score']:.1%}" for results in model_results.values()],
            'Consistency': [f"±{results['std_score']:.1%}" for results in model_results.values()],
            'Raw Score': [results['mean_score'] for results in model_results.values()]
        }).sort_values('Raw Score', ascending=False)
        
        # Highlight the best model
        best_model_row = model_df.iloc[0]
        st.success(f"🏆 **Best Performing Model**: {best_model_row['AI Model']} with {best_model_row['Accuracy Score']} accuracy")
        
        # Display model comparison
        st.dataframe(model_df[['AI Model', 'Accuracy Score', 'Consistency']], hide_index=True, use_container_width=True)
        
        with st.expander("🧠 What do these accuracy scores mean?"):
            st.markdown("""
            **Accuracy Score**: How often our AI gets the predictions right
            - 90%+ = Excellent! Very reliable predictions 🎯
            - 80-90% = Very Good! Trustworthy for most decisions ✅
            - 70-80% = Good! Useful but consider other factors 👍
            - Below 70% = Okay, but use with caution ⚠️
            
            **Consistency**: How much the accuracy varies
            - Lower numbers = More consistent and reliable
            - Higher numbers = More unpredictable results
            
            💡 **Remember**: Even the best AI can't predict the future perfectly. Always do your own research!
            """)
        
        st.dataframe(model_df, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(predictor.model, 'feature_importances_'):
            st.subheader("🔍 AI Model: What Drives the Predictions?")
            
            # Create a mapping of technical feature names to user-friendly descriptions
            feature_descriptions = {
                'Returns': '📈 Daily Price Changes',
                'Price_MA_5': '📊 5-Day Moving Average',
                'Price_MA_20': '📊 20-Day Moving Average', 
                'Price_MA_50': '📊 50-Day Moving Average',
                'MA_Ratio_5_20': '⚖️ Short vs Medium Term Trend',
                'MA_Ratio_20_50': '⚖️ Medium vs Long Term Trend',
                'Price_to_MA20': '🎯 Price Relative to 20-Day Average',
                'Volatility_5': '📉 Short-term Price Volatility',
                'Volatility_20': '📉 Medium-term Price Volatility',
                'Volume_MA_20': '📊 20-Day Average Trading Volume',
                'Volume_Ratio': '🔊 Current vs Average Volume',
                'RSI': '⚡ Relative Strength Index (Momentum)',
                'RSI_Normalized': '⚡ Normalized Momentum Indicator',
                'MACD': '🌊 MACD Trend Signal',
                'MACD_Signal': '🌊 MACD Signal Line',
                'MACD_Histogram': '🌊 MACD Histogram',
                'BB_Upper': '📏 Bollinger Band Upper Limit',
                'BB_Lower': '📏 Bollinger Band Lower Limit', 
                'BB_Position': '📍 Position within Bollinger Bands',
                'Returns_Lag_1': '📈 Yesterday\'s Price Change',
                'Returns_Lag_2': '📈 2-Day Ago Price Change',
                'Returns_Lag_3': '📈 3-Day Ago Price Change',
                'Returns_Lag_5': '📈 5-Day Ago Price Change',
                'Volume_Ratio_Lag_1': '🔊 Yesterday\'s Volume Pattern',
                'Volume_Ratio_Lag_2': '🔊 2-Day Ago Volume Pattern',
                'Volume_Ratio_Lag_3': '🔊 3-Day Ago Volume Pattern',
                'Volume_Ratio_Lag_5': '🔊 5-Day Ago Volume Pattern'
            }
            
            importance_df = pd.DataFrame({
                'Feature': predictor.feature_columns,
                'Importance': predictor.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            # Add user-friendly descriptions
            importance_df['Description'] = importance_df['Feature'].map(feature_descriptions)
            # Fill any missing descriptions with the original feature name
            importance_df['Description'] = importance_df['Description'].fillna(importance_df['Feature'])
            
            # Convert importance to percentage for better understanding
            importance_df['Importance_Percent'] = (importance_df['Importance'] * 100).round(1)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance_Percent',
                y='Description',
                orientation='h',
                title='🧠 Top 10 Factors Influencing AI Predictions',
                labels={'Importance_Percent': 'Influence on Predictions (%)', 'Description': 'Market Factor'},
                color='Importance_Percent',
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(
                height=500,
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Add explanation text
            st.info("""
            💡 **What does this mean?**
            - The bars show which market factors the AI considers most important when making predictions
            - Higher percentages mean the AI relies more heavily on that factor
            - These insights help you understand what drives the stock's price movements
            """)
            
            # Show a detailed table for more information
            with st.expander("📋 View Detailed Feature Analysis"):
                display_df = importance_df[['Description', 'Importance_Percent', 'Feature']].copy()
                display_df.columns = ['Market Factor', 'Influence (%)', 'Technical Name']
                display_df.index = range(1, len(display_df) + 1)
                st.dataframe(display_df, use_container_width=True)
        
        # Enhanced weekly predictions table with beginner explanations
        st.markdown("### 📅 Your Personal Investment Calendar")
        st.markdown("*Week-by-week predictions to help you plan your investment strategy*")
        
        with st.expander("💡 How to use this calendar"):
            st.markdown("""
            **Week**: Which week we're predicting for
            **Date**: The exact date of the prediction
            **Target Price**: What our AI thinks the stock will cost
            **Price Change**: How much money you could gain/lose per share
            **Percentage**: The percentage return on your investment
            **Signal**: Our recommendation for that time period
            **Days from Now**: How many days until that prediction
            
            💡 **Pro Tip**: Look for patterns! If several weeks show green signals, it might indicate a good trend.
            """)
        
        # Create comprehensive prediction table
        weekly_predictions = predictions.iloc[::7].head(8)  # Every 7 days
        
        # Get current time without timezone for consistent calculation
        current_time = pd.Timestamp.now().tz_localize(None)
        
        # Calculate additional metrics
        weekly_changes = [((price - current_price) / current_price * 100) for price in weekly_predictions['Predicted_Price']]
        weekly_profits = [(price - current_price) for price in weekly_predictions['Predicted_Price']]
        
        # Create signals based on change with beginner-friendly language
        signals = []
        for change in weekly_changes:
            if change > 5:
                signals.append("🟢 Strong Buy")
            elif change > 0:
                signals.append("🟡 Buy")
            elif change > -5:
                signals.append("🟠 Hold")
            else:
                signals.append("🔴 Sell")
        
        weekly_df = pd.DataFrame({
            'Week': [f"Week {i+1}" for i in range(len(weekly_predictions))],
            'Date': weekly_predictions['Date'].dt.strftime('%a, %b %d, %Y'),
            'Target Price': [f"${price:.2f}" for price in weekly_predictions['Predicted_Price']],
            'Price Change': [f"${profit:+.2f}" for profit in weekly_profits],
            'Percentage': [f"{change:+.1f}%" for change in weekly_changes],
            'Signal': signals,
            'Days from Now': [(pd.Timestamp(date).tz_localize(None) - current_time).days for date in weekly_predictions['Date']]
        })
        
        # Style the dataframe
        def style_dataframe(df):
            def color_signals(val):
                if val == "🟢 Strong Buy":
                    return 'background-color: #d4edda; color: #155724'
                elif val == "🟡 Buy":
                    return 'background-color: #fff3cd; color: #856404'
                elif val == "🟠 Hold":
                    return 'background-color: #ffeaa7; color: #856404'
                elif val == "🔴 Sell":
                    return 'background-color: #f8d7da; color: #721c24'
                return ''
            
            return df.style.applymap(color_signals, subset=['Signal'])
        
        st.dataframe(style_dataframe(weekly_df), use_container_width=True)
        
        # Add prediction milestones with explanations
        st.markdown("### 🎯 Key Investment Milestones")
        st.markdown("*Quick overview of important dates and price targets*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📅 Next Week Target**")
            next_week_price = predictions['Predicted_Price'].iloc[6] if len(predictions) > 6 else predictions['Predicted_Price'].iloc[-1]
            next_week_change = (next_week_price - current_price) / current_price * 100
            next_week_date = predictions['Date'].iloc[6] if len(predictions) > 6 else predictions['Date'].iloc[-1]
            
            st.metric(
                label=f"Price by {next_week_date.strftime('%b %d')}",
                value=f"${next_week_price:.2f}",
                delta=f"{next_week_change:+.1f}%",
                help="💡 Short-term prediction - good for quick decisions"
            )
        
        with col2:
            st.markdown("**📅 One Month Outlook**")
            one_month_idx = min(29, len(predictions) - 1)
            one_month_price = predictions['Predicted_Price'].iloc[one_month_idx]
            one_month_change = (one_month_price - current_price) / current_price * 100
            one_month_date = predictions['Date'].iloc[one_month_idx]
            
            st.metric(
                label=f"Price by {one_month_date.strftime('%b %d')}",
                value=f"${one_month_price:.2f}",
                delta=f"{one_month_change:+.1f}%",
                help="💡 Medium-term prediction - good for planning investments"
            )
        
        with col3:
            st.markdown("**📅 Highest Predicted Price**")
            best_price = predictions['Predicted_Price'].max()
            best_idx = predictions['Predicted_Price'].idxmax()
            best_date = predictions['Date'].iloc[best_idx]
            best_change = (best_price - current_price) / current_price * 100
            
            st.metric(
                label=f"Peak on {best_date.strftime('%b %d')}",
                value=f"${best_price:.2f}",
                delta=f"{best_change:+.1f}%",
                help="💡 Best potential outcome in our prediction period"
            )
        
        # Download predictions with user-friendly explanation
        st.markdown("### 💾 Take Your Analysis With You")
        st.markdown("*Download your complete analysis to review later or share with others*")
        
        # Prepare download data
        download_data = predictions.copy()
        download_data['Symbol'] = symbol
        download_data['Current_Price'] = current_price
        download_data['Expected_Return'] = (download_data['Predicted_Price'] - current_price) / current_price * 100
        
        csv_buffer = StringIO()
        download_data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📊 Download Predictions (CSV)",
            data=csv_data,
            file_name=f"{symbol}_predictions_{dt.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Disclaimer with beginner-friendly language
        st.markdown("---")
        st.warning("""
        ⚠️ **Important: Please Read Before Investing** 
        
        🎓 **This is an Educational Tool**: This app is designed to help you learn about stock analysis, not to tell you what to buy or sell.
        
        🤖 **AI Limitations**: Our AI is very smart, but it can't predict the future perfectly. Stock prices can be affected by news, company changes, and many other factors.
        
        💡 **Always Do Your Own Research**: 
        - Read about the company and its business
        - Check recent news and earnings reports
        - Consider talking to a financial advisor
        - Never invest money you can't afford to lose
        
        📚 **Remember**: This tool shows you patterns and possibilities, but the final investment decision is always yours!
        """)
    
    else:
        # Simple instruction message
        st.markdown("""
        <div style="background-color: #ffffff; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1E88E5; margin-bottom: 1rem;">
            <h3 style="color: #1E88E5; margin-bottom: 1rem;">📊 Welcome to Smart Stock Analyzer</h3>
            <p style="color: #333333; font-size: 1.1rem; margin-bottom: 0.5rem;">Select a stock symbol in the sidebar and click <strong>"Start Analysis"</strong> to begin your AI-powered stock analysis.</p>
            <p style="color: #666666; font-size: 1rem;"><strong>Need help getting started?</strong> Check the expandable help section in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show a simple centered call-to-action
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #ffffff; border: 2px solid #1E88E5; border-radius: 15px; margin: 2rem 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="color: #1E88E5; margin-bottom: 1rem;">👈 Choose a stock symbol from the sidebar to get started</h3>
            <p style="color: #333333; font-size: 1.1rem; font-weight: 500;">Popular choices: AAPL, TSLA, MSFT, AMZN, GOOGL</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
