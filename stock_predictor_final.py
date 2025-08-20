#!/usr/bin/env python3
"""
AI-Powered Stock Market Predictor
A complete stock analysis tool with AI ensemble models and interactive visualizations
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import datetime
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime
from datetime import timedelta
from scipy import stats

# Configure Streamlit
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .metric-card h2 {
        margin: 0.5rem 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-card h3 {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .success-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .success-card h2 {
        margin: 0.5rem 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .success-card h3 {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .prediction-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AIStockPredictor:
    """Advanced AI Stock Predictor with ensemble models"""
    
    def __init__(self):
        """Initialize the AI Stock Predictor"""
        self.scaler = StandardScaler()
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'Linear Regression': LinearRegression()
        }
        self.trained_models = {}
        self.feature_columns = []
        self.confidence_score = 0.0
        
    def fetch_stock_data(self, symbol, period="1y"):
        """Fetch stock data with robust error handling"""
        try:
            symbol = symbol.strip().upper()
            st.info(f"🔄 Fetching data for {symbol}...")
            
            ticker = yf.Ticker(symbol)
            
            # Try multiple periods if one fails
            periods_to_try = [period, "6mo", "3mo", "1mo"]
            data = None
            
            for try_period in periods_to_try:
                try:
                    data = ticker.history(period=try_period)
                    if not data.empty:
                        st.success(f"✅ Successfully fetched {len(data)} days of data")
                        break
                except Exception as e:
                    continue
            
            # Alternative method using yf.download
            if data is None or data.empty:
                try:
                    end_date = datetime.datetime.now()
                    start_date = end_date - timedelta(days=365)
                    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                except:
                    pass
            
            # Create demo data if still no data
            if data is None or data.empty:
                if symbol.upper() in ['DEMO', 'TEST', 'SAMPLE']:
                    return self._create_demo_data(), {"shortName": "Demo Stock", "longName": "Demo Stock for Testing"}, None
                else:
                    return None, None, f"❌ No data found for '{symbol}'. Try DEMO for sample data."
            
            # Get company info
            try:
                info = ticker.info
                if not info:
                    info = {"shortName": symbol, "longName": f"{symbol} Stock"}
            except:
                info = {"shortName": symbol, "longName": f"{symbol} Stock"}
            
            return data, info, None
            
        except Exception as e:
            return None, None, f"❌ Error: {str(e)}"
    
    def _create_demo_data(self):
        """Create realistic demo stock data"""
        np.random.seed(42)
        dates = pd.date_range(start=datetime.datetime.now() - timedelta(days=200), 
                             end=datetime.datetime.now(), freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays
        
        # Generate realistic price movement
        price = 150.0
        prices = []
        volumes = []
        
        for i in range(len(dates)):
            # Random walk with slight upward trend
            change = np.random.normal(0.5, 2.0)
            price = max(price + change, 50)  # Keep price reasonable
            prices.append(price)
            
            # Random volume
            base_volume = 1000000
            volume = base_volume + np.random.randint(-500000, 1500000)
            volumes.append(max(volume, 100000))
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = [prices[0]] + prices[:-1]  # Open is previous close
        data['High'] = data[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 3, len(data))
        data['Low'] = data[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 3, len(data))
        data['Volume'] = volumes
        data['Adj Close'] = data['Close']
        
        return data
    
    def create_features(self, data):
        """Create comprehensive technical indicators and features"""
        df = data.copy()
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            if len(df) >= window:
                df[f'MA_{window}'] = df['Close'].rolling(window).mean()
                df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
                df[f'Price_to_MA{window}'] = df['Close'] / df[f'MA_{window}']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(20).std()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Target: next day's price
        df['Target'] = df['Close'].shift(-1)
        
        return df.dropna()
    
    def train_models(self, data):
        """Train ensemble of AI models"""
        # Select features (exclude target and price columns)
        feature_cols = [col for col in data.columns 
                       if col not in ['Target', 'Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close']
                       and not data[col].isna().all()]
        
        self.feature_columns = feature_cols
        X = data[feature_cols].fillna(0)
        y = data['Target']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        model_scores = {}
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                model_scores[name] = max(score, 0)  # Ensure non-negative
                self.trained_models[name] = model
            except Exception as e:
                st.warning(f"⚠️ Model {name} training failed: {e}")
                model_scores[name] = 0
        
        # Calculate ensemble score
        if model_scores:
            self.confidence_score = np.mean(list(model_scores.values()))
        else:
            self.confidence_score = 0
        
        return model_scores
    
    def predict_future(self, data, days=30):
        """Make future predictions using ensemble models"""
        if not self.trained_models:
            return None, None
        
        # Get latest features
        latest_features = data[self.feature_columns].iloc[-1:].fillna(0)
        latest_scaled = self.scaler.transform(latest_features)
        
        # Make predictions with each model
        predictions = {}
        for name, model in self.trained_models.items():
            try:
                pred = model.predict(latest_scaled)[0]
                predictions[name] = pred
            except:
                continue
        
        if not predictions:
            return None, None
        
        # Ensemble prediction (weighted average)
        ensemble_pred = np.mean(list(predictions.values()))
        
        # Generate prediction series
        last_price = data['Close'].iloc[-1]
        pred_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days, freq='D')
        pred_dates = pred_dates[pred_dates.weekday < 5][:days]  # Only weekdays
        
        # Simple trend projection
        daily_change = (ensemble_pred - last_price) / days
        pred_prices = [last_price + daily_change * (i + 1) for i in range(len(pred_dates))]
        
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted_Price': pred_prices
        })
        
        return pred_df, predictions
    
    def calculate_risk_metrics(self, data):
        """Calculate comprehensive risk metrics"""
        returns = data['Returns'].dropna()
        
        # Basic metrics
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # VaR (Value at Risk)
        var_95 = np.percentile(returns, 5) * 100
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'max_drawdown': max_drawdown
        }

def create_enhanced_price_chart(data, predictions=None, confidence_data=None):
    """Create enhanced interactive price chart with confidence bands and advanced features"""
    fig = go.Figure()
    
    # Historical prices (candlestick)
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Stock Price",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Add 20-day moving average
    if 'MA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=2),
            hovertemplate="<b>MA 20</b><br>%{x}<br>$%{y:.2f}<extra></extra>"
        ))
    
    # Add 50-day moving average
    if 'MA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='blue', width=2),
            hovertemplate="<b>MA 50</b><br>%{x}<br>$%{y:.2f}<extra></extra>"
        ))
    
    # Current price reference line
    current_price = data['Close'].iloc[-1]
    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="bottom right"
    )
    
    # Add predictions with confidence bands
    if predictions is not None:
        # Main prediction line
        fig.add_trace(go.Scatter(
            x=predictions['Date'],
            y=predictions['Predicted_Price'],
            mode='lines+markers',
            name='🔮 AI Prediction',
            line=dict(color='purple', width=4, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate="<b>AI Prediction</b><br>%{x}<br>$%{y:.2f}<extra></extra>"
        ))
        
        # Confidence bands (±10% uncertainty)
        upper_bound = predictions['Predicted_Price'] * 1.1
        lower_bound = predictions['Predicted_Price'] * 0.9
        
        fig.add_trace(go.Scatter(
            x=predictions['Date'],
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=predictions['Date'],
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(128, 0, 128, 0.2)',
            name='🎯 Confidence Band',
            showlegend=True,
            hovertemplate="<b>Confidence Range</b><br>%{x}<br>$%{y:.2f}<extra></extra>"
        ))
        
        # Weekly milestone markers
        weekly_dates = predictions['Date'][::7]  # Every 7th prediction
        weekly_prices = predictions['Predicted_Price'][::7]
        
        fig.add_trace(go.Scatter(
            x=weekly_dates,
            y=weekly_prices,
            mode='markers',
            name='⭐ Weekly Milestones',
            marker=dict(
                size=15,
                symbol='star',
                color='gold',
                line=dict(width=2, color='orange')
            ),
            hovertemplate="<b>⭐ Weekly Target</b><br>%{x}<br>$%{y:.2f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="📈 Enhanced Stock Price Analysis with AI Predictions & Confidence Bands",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=700,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )
    
    return fig

def create_prediction_timeline_chart(predictions, current_price):
    """Create prediction timeline showing percentage changes over time"""
    if predictions is None:
        return None
    
    # Calculate percentage changes
    pct_changes = ((predictions['Predicted_Price'] - current_price) / current_price * 100).round(2)
    
    # Create color mapping
    colors = ['#ff4444' if x < 0 else '#00ff88' for x in pct_changes]
    
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=predictions['Date'],
        y=pct_changes,
        mode='lines+markers',
        name='Expected Return %',
        line=dict(color='blue', width=3),
        marker=dict(size=8, color=colors),
        hovertemplate="<b>%{x}</b><br>" +
                      "Expected Return: %{y:.1f}%<br>" +
                      "Target Price: $%{customdata:.2f}<br>" +
                      "<extra></extra>",
        customdata=predictions['Predicted_Price']
    ))
    
    # Add break-even line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Break-Even Line",
        annotation_position="bottom right"
    )
    
    # Add color zones
    fig.add_hrect(
        y0=5, y1=float('inf'),
        fillcolor="green", opacity=0.1,
        annotation_text="🟢 Strong Buy Zone", annotation_position="top left"
    )
    
    fig.add_hrect(
        y0=0, y1=5,
        fillcolor="lightgreen", opacity=0.1,
        annotation_text="🟡 Buy Zone", annotation_position="top left"
    )
    
    fig.add_hrect(
        y0=-5, y1=0,
        fillcolor="yellow", opacity=0.1,
        annotation_text="🟠 Hold Zone", annotation_position="bottom left"
    )
    
    fig.add_hrect(
        y0=float('-inf'), y1=-5,
        fillcolor="red", opacity=0.1,
        annotation_text="🔴 Sell Zone", annotation_position="bottom left"
    )
    
    fig.update_layout(
        title="🎯 Prediction Timeline - Expected Returns Over Time",
        xaxis_title="Date",
        yaxis_title="Expected Return (%)",
        height=500,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def create_weekly_targets_chart(predictions, current_price):
    """Create weekly targets bar chart"""
    if predictions is None:
        return None
    
    # Group by weeks
    weekly_data = predictions.iloc[::7].copy()  # Every 7th day
    weekly_data['Week'] = ['Week ' + str(i+1) for i in range(len(weekly_data))]
    weekly_data['Change'] = weekly_data['Predicted_Price'] - current_price
    weekly_data['Change_Pct'] = (weekly_data['Change'] / current_price * 100).round(1)
    
    # Color mapping
    colors = ['#ff4444' if x < 0 else '#00ff88' for x in weekly_data['Change']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=weekly_data['Week'],
        y=weekly_data['Predicted_Price'],
        name='Weekly Targets',
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>" +
                      "Target Price: $%{y:.2f}<br>" +
                      "Change: $%{customdata[0]:+.2f}<br>" +
                      "Change %: %{customdata[1]:+.1f}%<br>" +
                      "<extra></extra>",
        customdata=list(zip(weekly_data['Change'], weekly_data['Change_Pct']))
    ))
    
    # Add current price reference line
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Current: ${current_price:.2f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="📅 Weekly Price Targets",
        xaxis_title="Week",
        yaxis_title="Target Price ($)",
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def create_risk_return_gauges(risk_metrics, predictions, current_price):
    """Create interactive gauge dashboard for risk and return"""
    if predictions is None:
        return None
    
    # Calculate 30-day return
    future_price = predictions['Predicted_Price'].iloc[min(29, len(predictions)-1)]
    expected_return = ((future_price - current_price) / current_price * 100)
    
    # Risk score based on volatility
    risk_score = min(risk_metrics['volatility'] * 2, 100)  # Scale volatility to 0-100
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=("Expected 30-Day Return", "Risk Score")
    )
    
    # Return gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=expected_return,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Return %"},
        delta={'reference': 5},
        gauge={
            'axis': {'range': [None, 50]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgray"},
                {'range': [10, 25], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        }
    ), row=1, col=1)
    
    # Risk gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Level"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ), row=1, col=2)
    
    fig.update_layout(
        title="📊 Risk & Return Gauge Dashboard",
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_prediction_table(predictions, current_price):
    """Create detailed prediction timeline table"""
    if predictions is None:
        return None
    
    # Prepare weekly data
    weekly_data = predictions.iloc[::7].copy()  # Every 7th day
    weekly_data['Days_From_Now'] = range(7, len(weekly_data) * 7 + 1, 7)
    weekly_data['Price_Change'] = weekly_data['Predicted_Price'] - current_price
    weekly_data['Change_Pct'] = (weekly_data['Price_Change'] / current_price * 100).round(1)
    
    # Generate signals
    def get_signal(pct_change):
        if pct_change > 5:
            return "🟢 Strong Buy"
        elif pct_change > 2:
            return "🟡 Buy"
        elif pct_change > -2:
            return "🟠 Hold"
        else:
            return "🔴 Sell"
    
    weekly_data['Signal'] = weekly_data['Change_Pct'].apply(get_signal)
    
    # Format the data
    table_data = []
    for _, row in weekly_data.iterrows():
        table_data.append({
            'Date': row['Date'].strftime('%a, %b %d, %Y'),
            'Target Price': f"${row['Predicted_Price']:.2f}",
            'Price Change': f"${row['Price_Change']:+.2f}",
            'Change %': f"{row['Change_Pct']:+.1f}%",
            'Signal': row['Signal'],
            'Days From Now': f"{row['Days_From_Now']} days"
        })
    
    return pd.DataFrame(table_data)

def create_key_milestones(predictions, current_price):
    """Create key prediction milestone cards"""
    if predictions is None:
        return None, None, None
    
    # Next week target (7 days)
    next_week_price = predictions['Predicted_Price'].iloc[6] if len(predictions) > 6 else predictions['Predicted_Price'].iloc[-1]
    next_week_date = predictions['Date'].iloc[6] if len(predictions) > 6 else predictions['Date'].iloc[-1]
    next_week_change = ((next_week_price - current_price) / current_price * 100)
    
    # One month target (30 days)
    one_month_price = predictions['Predicted_Price'].iloc[29] if len(predictions) > 29 else predictions['Predicted_Price'].iloc[-1]
    one_month_date = predictions['Date'].iloc[29] if len(predictions) > 29 else predictions['Date'].iloc[-1]
    one_month_change = ((one_month_price - current_price) / current_price * 100)
    
    # Best predicted price
    best_price_idx = predictions['Predicted_Price'].idxmax()
    best_price = predictions['Predicted_Price'].iloc[best_price_idx]
    best_price_date = predictions['Date'].iloc[best_price_idx]
    best_price_change = ((best_price - current_price) / current_price * 100)
    
    return {
        'next_week': {
            'price': next_week_price,
            'date': next_week_date,
            'change': next_week_change
        },
        'one_month': {
            'price': one_month_price,
            'date': one_month_date,
            'change': one_month_change
        },
        'best_price': {
            'price': best_price,
            'date': best_price_date,
            'change': best_price_change
        }
    }

def create_beginner_tutorial():
    """Create an interactive tutorial for beginners"""
    st.markdown("## 🎓 **Stock Market Basics for Beginners**")
    
    with st.expander("📚 **What Are Stocks?** (Click to learn)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🏢 **What is a Stock?**
            A stock represents **ownership** in a company. When you buy a stock, you become a partial owner (shareholder) of that business.
            
            **Example:** If Apple has 1 billion shares and you own 100 shares, you own 0.00001% of Apple!
            
            ### 💰 **How Do You Make Money?**
            1. **Price Appreciation**: Buy low, sell high
            2. **Dividends**: Some companies pay you for owning their stock
            """)
        
        with col2:
            st.markdown("""
            ### 📈 **Stock Price Basics**
            - **Open**: Price when market opens (9:30 AM EST)
            - **High**: Highest price during the day
            - **Low**: Lowest price during the day  
            - **Close**: Price when market closes (4:00 PM EST)
            - **Volume**: How many shares were traded
            """)
    
    with st.expander("📊 **Understanding Chart Patterns** (Click to learn)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🕯️ **Candlestick Charts**
            Each "candle" shows 4 prices for one day:
            - **Green Candle** 🟢: Price went UP (Close > Open)
            - **Red Candle** 🔴: Price went DOWN (Close < Open)
            - **Wicks**: Show the high and low prices
            """)
        
        with col2:
            st.markdown("""
            ### 📏 **Moving Averages**
            - **MA 20**: Average price over last 20 days
            - **MA 50**: Average price over last 50 days
            - When price is ABOVE the line = **Bullish** 📈
            - When price is BELOW the line = **Bearish** 📉
            """)
    
    with st.expander("🎯 **Risk and Return Explained** (Click to learn)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### ⚠️ **What is Risk?**
            Risk = How much the stock price jumps around
            
            - **Low Risk** (0-30): Stable, like big banks
            - **Medium Risk** (30-60): Normal stocks
            - **High Risk** (60+): Very volatile, like crypto stocks
            """)
        
        with col2:
            st.markdown("""
            ### 💰 **What is Return?**
            Return = How much money you make (or lose)
            
            - **+10%** = You made 10% profit! 🎉
            - **-5%** = You lost 5% 😞
            - **0%** = Break even (no gain, no loss)
            """)
    
    with st.expander("🤖 **How AI Predictions Work** (Click to learn)", expanded=False):
        st.markdown("""
        ### 🧠 **Our AI Uses 4 Smart Models:**
        
        1. **🌳 Random Forest**: Like asking 100 experts and taking the average opinion
        2. **⚡ Gradient Boosting**: Learns from previous mistakes to get better
        3. **🧠 Neural Network**: Mimics how the human brain processes information
        4. **📏 Linear Regression**: Finds mathematical patterns in price movements
        
        ### 🎯 **What Our Predictions Mean:**
        - **🟢 Strong Buy**: AI thinks price will go up 5%+ 
        - **🟡 Buy**: AI thinks price will go up 2-5%
        - **🟠 Hold**: AI thinks price will stay about the same
        - **🔴 Sell**: AI thinks price will go down
        
        ### ⚠️ **Important Disclaimer:**
        - AI predictions are **educated guesses**, not guarantees
        - Always do your own research before investing
        - Never invest money you can't afford to lose
        """)

def create_terminology_guide():
    """Create a comprehensive terminology guide"""
    st.markdown("## 📖 **Stock Market Dictionary**")
    
    with st.expander("💰 **Money & Pricing Terms**", expanded=False):
        terms = {
            "Stock Price": "The current cost to buy one share of a company",
            "Market Cap": "Total value of all company shares (Price × Number of Shares)",
            "Dividend": "Money companies pay shareholders (like a bonus for owning stock)",
            "P/E Ratio": "Price-to-Earnings ratio - how expensive a stock is compared to profits",
            "Bull Market": "When stock prices are generally going up 📈",
            "Bear Market": "When stock prices are generally going down 📉"
        }
        
        for term, definition in terms.items():
            st.markdown(f"**{term}**: {definition}")
    
    with st.expander("📊 **Technical Analysis Terms**", expanded=False):
        tech_terms = {
            "RSI": "Relative Strength Index - shows if a stock is overbought (>70) or oversold (<30)",
            "MACD": "Moving Average Convergence Divergence - shows momentum changes",
            "Bollinger Bands": "Shows if price is high, low, or normal compared to recent average",
            "Volume": "How many shares were bought/sold (high volume = lots of interest)",
            "Volatility": "How much the price jumps around (high volatility = risky)",
            "Support": "Price level where stock tends to stop falling",
            "Resistance": "Price level where stock tends to stop rising"
        }
        
        for term, definition in tech_terms.items():
            st.markdown(f"**{term}**: {definition}")

def create_investment_tips():
    """Create practical investment tips for beginners"""
    st.markdown("## 💡 **Smart Investing Tips for Beginners**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✅ **DO These Things:**
        - 📚 **Educate yourself** before investing
        - 🎯 **Start small** with money you can afford to lose
        - 🌍 **Diversify** - don't put all money in one stock
        - ⏰ **Think long-term** (years, not days)
        - 📊 **Use our AI as a guide**, not gospel
        - 💰 **Only invest extra money** (not rent/food money)
        """)
    
    with col2:
        st.markdown("""
        ### ❌ **DON'T Do These Things:**
        - 😰 **Panic sell** when prices drop
        - 🎰 **Gamble** with money you need
        - 📺 **Follow hot tips** from social media
        - 💸 **Try to time the market** perfectly
        - 🏃 **Trade too frequently** (fees add up)
        - 🤔 **Invest in companies** you don't understand
        """)

def add_interactive_explanations():
    """Add interactive explanations throughout the interface"""
    return {
        'price_explanation': """
        💡 **What you're seeing**: This chart shows the stock's price movement over time. 
        Green candles = price went up that day, Red candles = price went down.
        """,
        
        'prediction_explanation': """
        🔮 **How to read predictions**: The purple dashed line shows where our AI thinks 
        the price will go. The shaded area shows uncertainty - wider area = less confident.
        """,
        
        'risk_explanation': """
        ⚠️ **Risk Score Meaning**: 
        • 0-30 = Low risk (like a savings account)
        • 30-60 = Medium risk (normal stocks)  
        • 60+ = High risk (could gain or lose a lot quickly)
        """,
        
        'return_explanation': """
        💰 **Return Percentage**: This shows how much money you'd make or lose.
        +10% means if you invested $100, you'd have $110.
        -5% means if you invested $100, you'd have $95.
        """
    }

def create_example_scenarios():
    """Create example investment scenarios"""
    st.markdown("## 📝 **Example Investment Scenarios**")
    
    with st.expander("💡 **Scenario 1: Conservative Investor (Sarah)**", expanded=False):
        st.markdown("""
        **Sarah's Profile:** New to investing, wants safety
        
        **What Sarah should look for:**
        - 🟢 Risk Score: 0-30 (Low risk)
        - 📈 Steady upward trend in charts
        - 🏢 Large, established companies (Apple, Microsoft)
        - 💰 Small position size (start with $100-500)
        
        **Sarah's Strategy:**
        - Buy and hold for years
        - Don't panic during short-term drops
        - Focus on companies she understands
        """)
    
    with st.expander("⚡ **Scenario 2: Growth Investor (Mike)**", expanded=False):
        st.markdown("""
        **Mike's Profile:** Experienced, wants growth
        
        **What Mike should look for:**
        - 🟡 Risk Score: 30-60 (Medium risk)
        - 📊 Strong AI predictions (Strong Buy signals)
        - 🚀 Companies in growing industries (tech, green energy)
        - 💰 Larger position sizes (but still diversified)
        
        **Mike's Strategy:**
        - Research company fundamentals
        - Use technical analysis
        - Set stop-losses to limit downside
        """)

def create_real_time_explanations(data, predictions, risk_metrics):
    """Create real-time explanations based on current analysis"""
    explanations = {}
    
    if data is not None:
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
        
        # Price movement explanation
        if price_change_pct > 2:
            explanations['price_movement'] = f"""
            📈 **Good News!** This stock went up {price_change_pct:.1f}% yesterday. 
            This means if you owned $100 worth, you'd now have ${100 + price_change_pct:.2f}.
            """
        elif price_change_pct < -2:
            explanations['price_movement'] = f"""
            📉 **Heads up!** This stock went down {abs(price_change_pct):.1f}% yesterday.
            This means if you owned $100 worth, you'd now have ${100 + price_change_pct:.2f}.
            Don't panic - this is normal in stock investing!
            """
        else:
            explanations['price_movement'] = f"""
            📊 **Steady as she goes!** This stock barely moved yesterday ({price_change_pct:+.1f}%).
            Small daily movements like this are totally normal.
            """
    
    if risk_metrics:
        volatility = risk_metrics['volatility']
        if volatility < 20:
            explanations['volatility'] = """
            🛡️ **Low Risk Stock**: This stock doesn't jump around much in price. 
            Good for beginners who want to sleep well at night!
            """
        elif volatility < 40:
            explanations['volatility'] = """
            ⚖️ **Medium Risk Stock**: This stock has normal price swings. 
            You might see +/-5% days occasionally - don't worry, that's normal!
            """
        else:
            explanations['volatility'] = """
            🎢 **High Risk Stock**: This stock can have big price swings! 
            You might see +/-10% days. Only invest what you can afford to lose.
            """
    
    if predictions is not None:
        future_price = predictions['Predicted_Price'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        expected_return = ((future_price - current_price) / current_price) * 100
        
        if expected_return > 10:
            explanations['prediction'] = f"""
            🚀 **Exciting Prediction!** Our AI thinks this stock could go up {expected_return:.1f}%! 
            But remember: predictions aren't guarantees. Always invest carefully.
            """
        elif expected_return > 0:
            explanations['prediction'] = f"""
            📈 **Positive Outlook**: Our AI thinks this stock could go up {expected_return:.1f}%. 
            A modest gain like this is often more realistic than huge jumps.
            """
        else:
            explanations['prediction'] = f"""
            📉 **Cautious Outlook**: Our AI thinks this stock might go down {abs(expected_return):.1f}%. 
            This doesn't mean "sell immediately" - it means "be extra careful and do research."
            """
    
    return explanations

def create_dashboard_metrics(data, risk_metrics, model_scores, beginner_mode=False):
    """Create dashboard with key metrics and beginner explanations"""
    current_price = data['Close'].iloc[-1]
    price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
    price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if beginner_mode:
            st.metric(
                label="💰 Current Price",
                value=f"${current_price:.2f}",
                delta=f"{price_change_pct:+.2f}%",
                help="This is what one share costs right now. The green/red arrow shows if it went up or down from yesterday."
            )
        else:
            st.metric(
                label="💰 Current Price",
                value=f"${current_price:.2f}",
                delta=f"{price_change_pct:+.2f}%"
            )
    
    with col2:
        if beginner_mode:
            st.metric(
                label="📊 Volatility (Risk Level)",
                value=f"{risk_metrics['volatility']:.1f}%",
                delta="Annualized",
                help="How much the price jumps around. Lower = safer, Higher = riskier but potentially more reward."
            )
        else:
            st.metric(
                label="� Volatility",
                value=f"{risk_metrics['volatility']:.1f}%",
                delta="Annualized"
            )
    
    with col3:
        if beginner_mode:
            st.metric(
                label="📈 Sharpe Ratio",
                value=f"{risk_metrics['sharpe_ratio']:.2f}",
                delta="Risk-Adjusted Return",
                help="How good the returns are compared to the risk. Higher is better. Above 1.0 is considered good."
            )
        else:
            st.metric(
                label="� Sharpe Ratio",
                value=f"{risk_metrics['sharpe_ratio']:.2f}",
                delta="Risk-Adjusted Return"
            )
    
    with col4:
        ai_confidence = np.mean(list(model_scores.values())) * 100 if model_scores else 0
        if beginner_mode:
            st.metric(
                label="🤖 AI Confidence",
                value=f"{ai_confidence:.1f}%",
                delta="Model Accuracy",
                help="How confident our AI models are in their predictions. Higher = more reliable predictions."
            )
        else:
            st.metric(
                label="🤖 AI Confidence",
                value=f"{ai_confidence:.1f}%",
                delta="Model Accuracy"
            )

def main():
    """Main Streamlit application with beginner-friendly features"""
    # Header
    st.markdown('<h1 class="main-header">🚀 AI Stock Market Predictor</h1>', unsafe_allow_html=True)
    
    # Beginner Mode Toggle
    beginner_mode = st.checkbox("🎓 **Beginner Mode** (Show explanations and tutorials)", value=True)
    
    if beginner_mode:
        # Educational sections for beginners
        st.markdown("---")
        create_beginner_tutorial()
        
        st.markdown("---")
        create_terminology_guide()
        
        st.markdown("---")
        create_investment_tips()
        
        st.markdown("---")
        create_example_scenarios()
        
        st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 Analysis Settings")
    
    if beginner_mode:
        st.sidebar.info("""
        🎓 **New to stocks?** 
        
        Start with these safe choices:
        • AAPL (Apple)
        • MSFT (Microsoft) 
        • GOOGL (Google)
        • DEMO (Practice data)
        """)
    
    # Stock input
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter stock ticker (e.g., AAPL, TSLA, GOOGL) or DEMO for sample data"
    )
    
    if beginner_mode:
        st.sidebar.markdown("""
        💡 **What's a stock symbol?**
        It's like a nickname for companies:
        • AAPL = Apple Inc.
        • GOOGL = Google (Alphabet)
        • TSLA = Tesla
        """)
    
    # Period selection
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y"
    }
    period_display = st.sidebar.selectbox("Data Period", list(period_options.keys()), index=3)
    period = period_options[period_display]
    
    if beginner_mode:
        st.sidebar.info(f"""
        📅 **You selected {period_display}**
        
        This means we'll look at the last {period_display.lower()} of price data to make predictions.
        """)
    
    # Prediction days
    prediction_days = st.sidebar.slider("Prediction Period (days)", 7, 60, 30)
    
    if beginner_mode:
        st.sidebar.info(f"""
        🔮 **{prediction_days} Day Forecast**
        
        Our AI will predict where the stock price might go over the next {prediction_days} days.
        """)
    
    # Analysis button
    if st.sidebar.button("🚀 Start AI Analysis", type="primary"):
        if symbol:
            with st.spinner("🤖 AI Analysis in Progress..."):
                # Initialize predictor
                predictor = AIStockPredictor()
                
                # Progress tracking
                progress = st.progress(0)
                status = st.empty()
                
                # Step 1: Fetch data
                status.text("📊 Fetching stock data...")
                progress.progress(20)
                
                data, info, error = predictor.fetch_stock_data(symbol, period)
                
                if error:
                    st.error(error)
                    if beginner_mode:
                        st.info("💡 **Tip**: Try entering 'DEMO' for sample data, or check if your stock symbol is correct.")
                    return
                
                # Step 2: Create features
                status.text("🧠 Engineering features...")
                progress.progress(40)
                
                featured_data = predictor.create_features(data)
                
                # Step 3: Train models
                status.text("🤖 Training AI models...")
                progress.progress(60)
                
                model_scores = predictor.train_models(featured_data)
                
                # Step 4: Make predictions
                status.text("🔮 Generating predictions...")
                progress.progress(80)
                
                predictions, model_predictions = predictor.predict_future(featured_data, prediction_days)
                
                # Step 5: Calculate risk
                status.text("⚡ Calculating risk metrics...")
                progress.progress(100)
                
                risk_metrics = predictor.calculate_risk_metrics(featured_data)
                
                # Clear progress
                progress.empty()
                status.empty()
                
                # Display results
                st.success("✅ **Analysis Complete!**")
                
                # Company info
                if info and 'longName' in info:
                    st.subheader(f"📈 {info.get('longName', symbol)} ({symbol})")
                else:
                    st.subheader(f"📈 {symbol} Stock Analysis")
                
                # Real-time explanations for beginners
                if beginner_mode:
                    explanations = create_real_time_explanations(data, predictions, risk_metrics)
                    
                    st.markdown("### 🎓 **What This Means for You:**")
                    for explanation in explanations.values():
                        st.info(explanation)
                
                # Dashboard metrics
                create_dashboard_metrics(data, risk_metrics, model_scores, beginner_mode)
                
                # Enhanced Price Chart
                if beginner_mode:
                    st.markdown("### 📈 **Price Chart Explained:**")
                    st.info("""
                    🕯️ **How to read this chart:**
                    • **Green candles** = Price went UP that day 📈
                    • **Red candles** = Price went DOWN that day 📉  
                    • **Orange line (MA 20)** = Average price over last 20 days
                    • **Blue line (MA 50)** = Average price over last 50 days
                    • **Purple stars** = AI's weekly price predictions
                    • **Gray shaded area** = AI's confidence range (uncertainty)
                    """)
                
                st.plotly_chart(
                    create_enhanced_price_chart(data, predictions), 
                    use_container_width=True
                )
                
                # Key Milestones Cards
                if predictions is not None:
                    milestones = create_key_milestones(predictions, data['Close'].iloc[-1])
                    
                    st.subheader("🎯 Key Prediction Milestones")
                    
                    if beginner_mode:
                        st.info("""
                        💡 **What these cards mean:** These show our AI's best guesses for future prices. 
                        Green numbers = profit, Red numbers = loss. Remember: these are predictions, not guarantees!
                        """)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        change_color = "🟢" if milestones['next_week']['change'] > 0 else "🔴"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📅 Next Week Target</h4>
                            <h2>${milestones['next_week']['price']:.2f}</h2>
                            <p>{milestones['next_week']['date'].strftime('%a, %b %d')}</p>
                            <h3>{change_color} {milestones['next_week']['change']:+.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        change_color = "🟢" if milestones['one_month']['change'] > 0 else "🔴"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>📆 One Month Target</h4>
                            <h2>${milestones['one_month']['price']:.2f}</h2>
                            <p>{milestones['one_month']['date'].strftime('%a, %b %d')}</p>
                            <h3>{change_color} {milestones['one_month']['change']:+.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="success-card">
                            <h4>🏆 Best Predicted Price</h4>
                            <h2>${milestones['best_price']['price']:.2f}</h2>
                            <p>{milestones['best_price']['date'].strftime('%a, %b %d')}</p>
                            <h3>🟢 {milestones['best_price']['change']:+.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Enhanced Charts Section
                if predictions is not None:
                    st.subheader("📊 Advanced Prediction Analysis")
                    
                    if beginner_mode:
                        st.info("""
                        📚 **Understanding the charts below:**
                        • **Timeline Chart**: Shows expected returns over time (green = good, red = be careful)
                        • **Weekly Targets**: Bar chart of weekly price goals
                        • **Risk & Return Gauges**: Speedometer-style charts showing risk level and expected returns
                        """)
                    
                    # Row 1: Timeline and Targets
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        timeline_chart = create_prediction_timeline_chart(predictions, data['Close'].iloc[-1])
                        if timeline_chart:
                            st.plotly_chart(timeline_chart, use_container_width=True)
                    
                    with col2:
                        targets_chart = create_weekly_targets_chart(predictions, data['Close'].iloc[-1])
                        if targets_chart:
                            st.plotly_chart(targets_chart, use_container_width=True)
                    
                    # Row 2: Risk & Return Gauges
                    gauges_chart = create_risk_return_gauges(risk_metrics, predictions, data['Close'].iloc[-1])
                    if gauges_chart:
                        st.plotly_chart(gauges_chart, use_container_width=True)
                
                # Model performance
                if model_scores:
                    st.subheader("🤖 AI Model Performance")
                    
                    if beginner_mode:
                        st.info("""
                        🧠 **What you're seeing:** Our AI uses 4 different "brains" to make predictions. 
                        Each brain has a score from 0 to 1 (like a grade). Higher scores = better predictions.
                        """)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Model scores chart
                        model_df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['Score'])
                        fig_models = px.bar(
                            model_df, 
                            y=model_df.index, 
                            x='Score',
                            title="Model Accuracy Scores",
                            color='Score',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_models, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 📊 **Model Insights**")
                        if beginner_mode:
                            st.markdown("**🎓 How to read these scores:**")
                        
                        for name, score in model_scores.items():
                            if score > 0.7:
                                icon = "✅"
                                grade = "Excellent"
                                explanation = "Very reliable!" if beginner_mode else ""
                            elif score > 0.5:
                                icon = "📊"
                                grade = "Good"
                                explanation = "Pretty trustworthy" if beginner_mode else ""
                            elif score > 0.3:
                                icon = "⚠️"
                                grade = "Fair"
                                explanation = "Okay, but be cautious" if beginner_mode else ""
                            else:
                                icon = "❌"
                                grade = "Poor"
                                explanation = "Don't rely on this one" if beginner_mode else ""
                            
                            display_text = f"{icon} **{name}**: {score:.3f} ({grade})"
                            if beginner_mode:
                                display_text += f" - {explanation}"
                            
                            if score > 0.7:
                                st.success(display_text)
                            elif score > 0.5:
                                st.info(display_text)
                            elif score > 0.3:
                                st.warning(display_text)
                            else:
                                st.error(display_text)
                
                # Predictions
                if predictions is not None:
                    st.subheader("🔮 AI Predictions")
                    
                    if beginner_mode:
                        st.info("""
                        💡 **Investment Decision Helper:** Look at the percentages below. 
                        Positive (+) = potential profit, Negative (-) = potential loss.
                        """)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📈 **Price Targets**")
                        current_price = data['Close'].iloc[-1]
                        future_price = predictions['Predicted_Price'].iloc[-1]
                        price_change = future_price - current_price
                        price_change_pct = (price_change / current_price) * 100
                        
                        if price_change_pct > 5:
                            st.success(f"🚀 **Bullish**: {price_change_pct:+.1f}% expected return")
                            if beginner_mode:
                                st.info(f"💰 **For beginners**: This means if you invested $100, you might have ${100 * (1 + price_change_pct/100):.2f}")
                        elif price_change_pct > 0:
                            st.info(f"📈 **Positive**: {price_change_pct:+.1f}% expected return")
                            if beginner_mode:
                                st.info(f"💰 **For beginners**: This means if you invested $100, you might have ${100 * (1 + price_change_pct/100):.2f}")
                        elif price_change_pct > -5:
                            st.warning(f"📊 **Neutral**: {price_change_pct:+.1f}% expected return")
                            if beginner_mode:
                                st.warning(f"⚠️ **For beginners**: Small loss expected. If you invested $100, you might have ${100 * (1 + price_change_pct/100):.2f}")
                        else:
                            st.error(f"📉 **Bearish**: {price_change_pct:+.1f}% expected return")
                            if beginner_mode:
                                st.error(f"🚨 **For beginners**: Significant loss expected. If you invested $100, you might have ${100 * (1 + price_change_pct/100):.2f}")
                        
                        st.write(f"• **Current Price**: ${current_price:.2f}")
                        st.write(f"• **Target Price**: ${future_price:.2f}")
                        st.write(f"• **Expected Change**: ${price_change:+.2f}")
                    
                    with col2:
                        st.markdown("### ⚠️ **Risk Assessment**")
                        volatility = risk_metrics['volatility']
                        
                        if volatility < 20:
                            st.success("🟢 **Low Risk**: Stable stock")
                            if beginner_mode:
                                st.info("👍 **Perfect for beginners!** This stock doesn't jump around much.")
                        elif volatility < 40:
                            st.info("🟡 **Medium Risk**: Moderate volatility")
                            if beginner_mode:
                                st.info("📊 **Good for learning**: Some ups and downs, but manageable.")
                        elif volatility < 60:
                            st.warning("🟠 **High Risk**: Volatile stock")
                            if beginner_mode:
                                st.warning("⚠️ **Be careful!** This stock can have big swings. Start small.")
                        else:
                            st.error("🔴 **Very High Risk**: Extremely volatile")
                            if beginner_mode:
                                st.error("🚨 **For experts only!** This stock is like a roller coaster. Very risky for beginners.")
                        
                        st.write(f"• **Volatility**: {volatility:.1f}%")
                        st.write(f"• **Max Drawdown**: {risk_metrics['max_drawdown']:+.1f}%")
                        st.write(f"• **Value at Risk (95%)**: {risk_metrics['var_95']:+.2f}%")
                        
                        if beginner_mode:
                            st.info("""
                            📚 **What these numbers mean:**
                            • **Volatility**: How jumpy the price is
                            • **Max Drawdown**: Biggest loss period in history  
                            • **Value at Risk**: Worst case scenario (5% chance of happening)
                            """)
                
                # Detailed Prediction Timeline Table
                if predictions is not None:
                    st.subheader("📅 Detailed Prediction Timeline")
                    
                    prediction_table = create_prediction_table(predictions, data['Close'].iloc[-1])
                    if prediction_table is not None:
                        st.markdown("### 📋 **Weekly Breakdown with AI Signals**")
                        
                        # Style the dataframe
                        def style_signal(val):
                            if "Strong Buy" in val:
                                return 'background-color: #d4edda; color: #155724'
                            elif "Buy" in val:
                                return 'background-color: #fff3cd; color: #856404'
                            elif "Hold" in val:
                                return 'background-color: #ffeaa7; color: #6c757d'
                            elif "Sell" in val:
                                return 'background-color: #f8d7da; color: #721c24'
                            return ''
                        
                        styled_table = prediction_table.style.applymap(style_signal, subset=['Signal'])
                        st.dataframe(styled_table, use_container_width=True, hide_index=True)
                        
                        # Signal legend
                        st.markdown("""
                        **Signal Legend:**
                        - 🟢 **Strong Buy**: Expected return > +5%
                        - 🟡 **Buy**: Expected return +2% to +5%
                        - 🟠 **Hold**: Expected return -2% to +2%
                        - 🔴 **Sell**: Expected return < -2%
                        """)
                
                # Technical analysis
                st.subheader("📊 Technical Analysis")
                
                # Recent data table
                st.markdown("### 📋 **Recent Trading Data**")
                if beginner_mode:
                    st.info("""
                    📊 **How to read this table:** Each row is one trading day. 
                    • **Open**: Price when market opened
                    • **High/Low**: Highest and lowest prices that day
                    • **Close**: Price when market closed
                    • **Volume**: How many shares were traded
                    • **Change**: Daily percentage change
                    """)
                
                recent_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
                recent_data['Change'] = recent_data['Close'].pct_change() * 100
                recent_data.index = recent_data.index.strftime('%Y-%m-%d')
                st.dataframe(recent_data.round(2), use_container_width=True)
                
                # Beginner Summary Section
                if beginner_mode:
                    st.markdown("---")
                    st.markdown("## 📝 **Investment Summary for Beginners**")
                    
                    # Create simple recommendation
                    current_price = data['Close'].iloc[-1]
                    future_price = predictions['Predicted_Price'].iloc[-1] if predictions is not None else current_price
                    expected_return = ((future_price - current_price) / current_price * 100) if predictions is not None else 0
                    volatility = risk_metrics['volatility']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if expected_return > 5 and volatility < 30:
                            recommendation = "🟢 **GOOD FOR BEGINNERS**"
                            reason = "Low risk with positive expected returns"
                        elif expected_return > 0 and volatility < 50:
                            recommendation = "🟡 **PROCEED WITH CAUTION**"
                            reason = "Moderate risk, start with small amounts"
                        elif volatility > 50:
                            recommendation = "🔴 **NOT FOR BEGINNERS**"
                            reason = "Too risky for new investors"
                        else:
                            recommendation = "🟠 **NEUTRAL**"
                            reason = "No clear direction, consider waiting"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🎓 Beginner Recommendation</h4>
                            <h3>{recommendation}</h3>
                            <p>{reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        risk_level = "Low" if volatility < 30 else "Medium" if volatility < 50 else "High"
                        risk_color = "🟢" if volatility < 30 else "🟡" if volatility < 50 else "🔴"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>⚠️ Risk Level</h4>
                            <h3>{risk_color} {risk_level} Risk</h3>
                            <p>{volatility:.1f}% volatility</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        return_color = "🟢" if expected_return > 5 else "🟡" if expected_return > 0 else "🔴"
                        return_text = "Good" if expected_return > 5 else "Modest" if expected_return > 0 else "Negative"
                        
                        st.markdown(f"""
                        <div class="success-card">
                            <h4>💰 Expected Return</h4>
                            <h3>{return_color} {return_text}</h3>
                            <p>{expected_return:+.1f}% predicted</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 📚 **Final Reminders:**")
                    st.info("""
                    💡 **Remember:**
                    • **Never invest money you can't afford to lose**
                    • **Start small and learn as you go**
                    • **Diversify across multiple stocks**
                    • **Think long-term (years, not days)**
                    • **Our AI helps, but you make the decisions**
                    • **Consider talking to a financial advisor**
                    """)
        else:
            st.warning("⚠️ Please enter a stock symbol to analyze.")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ **About**")
    st.sidebar.info("""
    🤖 **AI-Powered Analysis**
    - 4 ML models ensemble
    - Real-time data from Yahoo Finance
    - Advanced technical indicators
    - Risk assessment & predictions
    
    💡 **Tips:**
    - Use 'DEMO' for sample data
    - Try popular symbols: AAPL, GOOGL, TSLA
    """)
    
    if beginner_mode:
        st.sidebar.markdown("### 🎓 **Beginner Resources**")
        st.sidebar.info("""
        📚 **Learn More:**
        - Toggle beginner mode ON for explanations
        - Start with DEMO to practice
        - Focus on large, stable companies
        - Take your time to understand
        """)

if __name__ == "__main__":
    main()
