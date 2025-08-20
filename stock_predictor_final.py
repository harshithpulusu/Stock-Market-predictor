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
    
    /* TECHNICAL IMPROVEMENTS - Enhanced Styling & Interactivity */
    
    /* Professional Color Schemes */
    :root {
        --primary-color: #2E86AB;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --error-color: #F44336;
        --info-color: #2196F3;
        --neutral-color: #9E9E9E;
    }
    
    /* Mobile-Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .metric-card, .success-card {
            padding: 1rem;
            margin: 0.25rem 0;
        }
        .metric-card h2, .success-card h2 {
            font-size: 1.5rem;
        }
        .chart-comparison {
            flex-direction: column;
        }
        .chart-comparison > div {
            min-width: 100%;
        }
    }
    
    /* Enhanced Chart Container */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin: 20px 0;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
        width: 100%;
        height: 650px;
        overflow: hidden;
        resize: none !important;
        position: relative;
    }
    
    .chart-container:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Fixed Price Chart Specific Styling */
    .chart-container .js-plotly-plot {
        width: 100% !important;
        height: 600px !important;
        max-width: none !important;
        max-height: 600px !important;
        resize: none !important;
    }
    
    .chart-container .plotly-graph-div {
        width: 100% !important;
        height: 600px !important;
        resize: none !important;
    }
    
    /* Side-by-side Layout */
    .chart-comparison {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin: 20px 0;
    }
    
    .chart-comparison > div {
        flex: 1;
        min-width: 300px;
    }
    
    /* Visual Hierarchy Headers */
    .section-header {
        background: linear-gradient(135deg, var(--primary-color), #4A90E2);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        margin: 25px 0 15px 0;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Confidence Indicators */
    .confidence-high {
        border-left: 5px solid #4CAF50;
        background: rgba(76, 175, 80, 0.1);
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        position: relative;
    }
    
    .confidence-medium {
        border-left: 5px solid #FF9800;
        background: rgba(255, 152, 0, 0.1);
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        position: relative;
    }
    
    .confidence-low {
        border-left: 5px solid #F44336;
        background: rgba(244, 67, 54, 0.1);
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        position: relative;
    }
</style>
""", unsafe_allow_html=True)

# TECHNICAL IMPROVEMENTS - Enhanced Chart Functions

def get_enhanced_chart_config():
    """Returns enhanced configuration for all charts with improved interactivity"""
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'stock_prediction_chart',
            'height': 600,
            'width': 1000,
            'scale': 2
        },
        'responsive': True,
        'scrollZoom': True,
        'doubleClick': 'reset+autosize'
    }

def get_fixed_price_chart_config():
    """Returns fixed configuration for the main price chart - non-resizable"""
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ],
        'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'resetScale2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'stock_prediction_chart',
            'height': 600,
            'width': 1200,
            'scale': 2
        },
        'responsive': False,
        'scrollZoom': False,
        'doubleClick': 'reset',
        'staticPlot': False,
        'editable': False
    }

def get_enhanced_hover_template(chart_type):
    """Returns enhanced hover templates for different chart types"""
    templates = {
        'price': '<b>📅 Date:</b> %{x}<br><b>💰 Price:</b> $%{y:.2f}<br><b>📊 Volume:</b> %{customdata:,.0f}<br><extra></extra>',
        'prediction': '<b>🔮 Predicted Price:</b> $%{y:.2f}<br><b>📅 Date:</b> %{x}<br><b>📈 Change:</b> %{customdata:+.1f}%<br><extra></extra>',
        'milestone': '<b>⭐ Milestone:</b> %{text}<br><b>📅 Date:</b> %{x}<br><b>💰 Target:</b> $%{y:.2f}<br><b>🎯 Confidence:</b> %{customdata:.0f}%<br><extra></extra>',
        'signal': '<b>🎯 Signal:</b> %{text}<br><b>📅 Date:</b> %{x}<br><b>💰 Price:</b> $%{y:.2f}<br><b>📊 Strength:</b> %{customdata:.0f}%<br><extra></extra>'
    }
    return templates.get(chart_type, templates['price'])

def assess_prediction_confidence(model_scores, volatility, data_quality=1.0):
    """Assess the confidence level of predictions based on multiple factors"""
    # Base confidence from model performance
    avg_model_score = np.mean(list(model_scores.values())) if model_scores else 0
    
    # Adjust for volatility (higher volatility = lower confidence)
    volatility_factor = max(0, 1 - (volatility / 100))
    
    # Calculate overall confidence
    confidence = (avg_model_score * 0.6 + volatility_factor * 0.3 + data_quality * 0.1) * 100
    confidence = max(0, min(100, confidence))  # Clamp between 0-100
    
    # Categorize confidence levels
    if confidence >= 75:
        return confidence, "high", "🎯 High Confidence"
    elif confidence >= 50:
        return confidence, "medium", "⚠️ Medium Confidence"
    else:
        return confidence, "low", "❌ Low Confidence"

def create_enhanced_layout_config(title, height=600):
    """Creates enhanced layout configuration for charts"""
    return {
        'title': {
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2E86AB', 'family': 'Arial Black'}
        },
        'height': height,
        'hovermode': 'x unified',
        'showlegend': True,
        'legend': {
            'orientation': "h",
            'yanchor': "bottom",
            'y': 1.02,
            'xanchor': "right",
            'x': 1,
            'bgcolor': 'rgba(255,255,255,0.8)',
            'bordercolor': '#2E86AB',
            'borderwidth': 1
        },
        'plot_bgcolor': 'rgba(248,249,250,0.8)',
        'paper_bgcolor': 'rgba(255,255,255,0.95)',
        'xaxis': {
            'gridcolor': 'lightgray',
            'gridwidth': 0.5,
            'zeroline': False,
            'showspikes': True,
            'spikecolor': '#2E86AB',
            'spikethickness': 1,
            'spikemode': 'across'
        },
        'yaxis': {
            'gridcolor': 'lightgray',
            'gridwidth': 0.5,
            'zeroline': False,
            'showspikes': True,
            'spikecolor': '#2E86AB',
            'spikethickness': 1,
            'spikemode': 'across'
        },
        'annotations': [],
        'shapes': []
    }

# Import the exact same class and functions from the original file
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

# Educational functions (placeholder - you can add the full implementations)
def create_beginner_tutorial():
    st.markdown("## 🎓 Stock Market Tutorial")
    st.info("Educational content for beginners...")

def create_terminology_guide():
    st.markdown("## 📖 Financial Dictionary") 
    st.info("Key terms and definitions...")

def create_investment_tips():
    st.markdown("## 💡 Investment Tips")
    st.info("Practical advice for new investors...")

def create_example_scenarios():
    st.markdown("## 📝 Example Scenarios")
    st.info("Real-world investment examples...")

# Main function with session state
def main():
    """Main Streamlit application with session state management"""
    # Header
    st.markdown('<h1 class="main-header">🚀 AI Stock Market Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Analysis Settings")
    
    # Beginner Mode Toggle (moved to sidebar)
    beginner_mode = st.sidebar.checkbox("🎓 **Beginner Mode** (Show explanations and tutorials)", value=True)
    
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
    
    # Educational sections for beginners (moved after controls)
    if beginner_mode:
        st.markdown("---")
        st.markdown("## 📚 **Learning Center**")
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📖 Tutorial", "📚 Dictionary", "💡 Tips", "📝 Examples"])
        
        with tab1:
            create_beginner_tutorial()
        
        with tab2:
            create_terminology_guide()
        
        with tab3:
            create_investment_tips()
        
        with tab4:
            create_example_scenarios()
        
        st.markdown("---")
    
    # Analysis button
    if st.sidebar.button("🚀 Start AI Analysis", type="primary"):
        if symbol:
            # Store inputs in session state
            st.session_state.symbol = symbol
            st.session_state.period = period
            st.session_state.prediction_days = prediction_days
            
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
                
                # Store results in session state
                st.session_state.data = data
                st.session_state.info = info
                st.session_state.predictions = predictions
                st.session_state.model_scores = model_scores
                st.session_state.risk_metrics = risk_metrics
                st.session_state.analysis_complete = True
                
                # Clear progress
                progress.empty()
                status.empty()
        else:
            st.error("⚠️ Please enter a stock symbol to analyze.")
    
    # Display results if analysis is complete
    if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
        # Get data from session state
        data = st.session_state.data
        info = st.session_state.info
        predictions = st.session_state.predictions
        model_scores = st.session_state.model_scores
        risk_metrics = st.session_state.risk_metrics
        symbol = st.session_state.symbol
        
        st.success("✅ **Analysis Complete!**")
        
        # Company info
        if info and 'longName' in info:
            st.subheader(f"📈 {info.get('longName', symbol)} ({symbol})")
        else:
            st.subheader(f"📈 {symbol} Stock Analysis")
        
        # Real-time explanations for beginners
        if beginner_mode:
            st.markdown("### 🎓 **What This Means for You:**")
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            
            if price_change_pct > 2:
                st.info(f"📈 **Good News!** This stock went up {price_change_pct:.1f}% yesterday.")
            elif price_change_pct < -2:
                st.warning(f"📉 **Heads up!** This stock went down {abs(price_change_pct):.1f}% yesterday.")
            else:
                st.info(f"📊 **Steady!** This stock barely moved yesterday ({price_change_pct:+.1f}%).")
        
        # Dashboard metrics
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
                    label="📊 Volatility",
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
                    label="📈 Sharpe Ratio",
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
        
        # NEW: Modern Enhanced Price Chart with Black Background & Legend
        if predictions is not None:
            st.markdown('<div class="section-header">📈 Modern Price Chart & AI Predictions</div>', unsafe_allow_html=True)
            
            # Create modern chart with black background
            fig = go.Figure()
            
            # Calculate data for chart
            data_extended = data.tail(60).copy()
            current_price = data['Close'].iloc[-1]
            
            # Get prediction data for chart
            pred_dates = predictions['Date'][:30]
            pred_prices = predictions['Predicted_Price'][:30]
            
            # Calculate moving averages
            data_extended['MA20'] = data_extended['Close'].rolling(window=20).mean()
            data_extended['MA50'] = data_extended['Close'].rolling(window=50).mean()
            
            # 1. HISTORICAL PRICE LINE (Neon Blue)
            fig.add_trace(go.Scatter(
                x=data_extended.index, 
                y=data_extended['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='#00D4FF', width=3, shape='spline'),
                hovertemplate='<b>📅 Date:</b> %{x}<br><b>💰 Historical Price:</b> $%{y:.2f}<extra></extra>',
                showlegend=True
            ))
            
            # 2. Moving Averages
            fig.add_trace(go.Scatter(
                x=data_extended.index,
                y=data_extended['MA20'],
                mode='lines',
                name='20-Day MA',
                line=dict(color='#A23B72', width=1.5, dash='dot'),
                hovertemplate='<b>20-Day MA:</b> $%{y:.2f}<br><extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=data_extended.index,
                y=data_extended['MA50'],
                mode='lines',
                name='50-Day MA',
                line=dict(color='#F18F01', width=1.5, dash='dash'),
                hovertemplate='<b>50-Day MA:</b> $%{y:.2f}<br><extra></extra>'
            ))
            
            # 3. Current Price Reference Line
            current_price = data['Close'].iloc[-1]
            fig.add_hline(
                y=current_price,
                line_dash="solid",
                line_color="#C73E1D",
                line_width=2,
                annotation_text=f"Current: ${current_price:.2f}",
                annotation_position="top left"
            )
            
            # 4. AI Predictions with Confidence Bands
            pred_dates = predictions['Date'][:30]
            pred_prices = predictions['Predicted_Price'][:30]
            
            # Calculate confidence bands (±5% uncertainty range)
            confidence_upper = pred_prices * 1.05
            confidence_lower = pred_prices * 0.95
            
            # Confidence bands
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=confidence_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=confidence_lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(128, 0, 128, 0.2)',
                name='Confidence Band',
                hovertemplate='<b>Confidence Range:</b><br>Upper: $%{y:.2f}<br><extra></extra>'
            ))
            
            # Main prediction line
            fig.add_trace(go.Scatter(
                x=pred_dates, 
                y=pred_prices,
                mode='lines+markers',
                name='AI Predictions',
                line=dict(color='#8A2BE2', width=3),
                marker=dict(size=6, color='#8A2BE2'),
                hovertemplate='<b>Predicted Price:</b> $%{y:.2f}<br><b>Date:</b> %{x}<br><extra></extra>'
            ))
            
            # 5. Weekly Milestone Markers (Golden Stars)
            weekly_milestones = []
            for i, date in enumerate(pred_dates):
                if i % 7 == 0:  # Every 7 days
                    weekly_milestones.append((date, pred_prices.iloc[i]))
            
            if weekly_milestones:
                milestone_dates, milestone_prices = zip(*weekly_milestones)
                fig.add_trace(go.Scatter(
                    x=milestone_dates,
                    y=milestone_prices,
                    mode='markers',
                    name='Weekly Milestones',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='gold',
                        line=dict(color='orange', width=2)
                    ),
                    hovertemplate='<b>⭐ Weekly Milestone</b><br><b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<br><extra></extra>'
                ))
            
            # 6. Color-coded Prediction Zones
            min_price = min(data_extended['Close'].min(), pred_prices.min())
            max_price = max(data_extended['Close'].max(), pred_prices.max())
            price_range = max_price - min_price
            
            # Bullish zone (top 1/3)
            fig.add_hrect(
                y0=min_price + (2/3) * price_range,
                y1=max_price,
                fillcolor="rgba(0, 255, 0, 0.1)",
                layer="below",
                line_width=0,
                annotation_text="🐂 Bullish Zone",
                annotation_position="top right"
            )
            
            # Neutral zone (middle 1/3)
            fig.add_hrect(
                y0=min_price + (1/3) * price_range,
                y1=min_price + (2/3) * price_range,
                fillcolor="rgba(255, 255, 0, 0.1)",
                layer="below",
                line_width=0,
                annotation_text="⚖️ Neutral Zone",
                annotation_position="top left"
            )
            
            # Bearish zone (bottom 1/3)
            fig.add_hrect(
                y0=min_price,
                y1=min_price + (1/3) * price_range,
                fillcolor="rgba(255, 0, 0, 0.1)",
                layer="below",
                line_width=0,
                annotation_text="🐻 Bearish Zone",
                annotation_position="bottom right"
            )
            
            # Enhanced layout with technical improvements
            enhanced_config = create_enhanced_layout_config(f"📈 {symbol} - Enhanced Price Analysis & AI Predictions")
            
            # Add confidence assessment
            confidence_score, confidence_level, confidence_text = assess_prediction_confidence(
                model_scores, risk_metrics['volatility']
            )
            
            fig.update_layout(**enhanced_config)
            
            # APPLY BLACK BACKGROUND THEME
            fig.update_layout(
                # BLACK BACKGROUND
                plot_bgcolor='#000000',
                paper_bgcolor='#111111',
                
                # WHITE TEXT AND GRID
                title=dict(font=dict(color='#FFFFFF', size=24)),
                xaxis=dict(
                    titlefont=dict(color='#FFFFFF'),
                    tickfont=dict(color='#CCCCCC'),
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    zerolinecolor='rgba(255, 255, 255, 0.2)'
                ),
                yaxis=dict(
                    titlefont=dict(color='#FFFFFF'),
                    tickfont=dict(color='#CCCCCC'),
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    zerolinecolor='rgba(255, 255, 255, 0.2)'
                ),
                
                # WHITE LEGEND
                legend=dict(
                    bgcolor="rgba(0, 0, 0, 0.8)",
                    bordercolor="#FFFFFF",
                    borderwidth=1,
                    font=dict(color="#FFFFFF")
                ),
                
                # WHITE FONT
                font=dict(color="#FFFFFF")
            )
            
            # Set fixed height for the chart
            fig.update_layout(height=600, width=None)  # Fixed height, auto width
            
            # Add confidence indicator to chart
            fig.add_annotation(
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                text=f"{confidence_text}<br>Score: {confidence_score:.0f}%",
                showarrow=False,
                font=dict(size=12, color="white"),
                bgcolor={"high": "#4CAF50", "medium": "#FF9800", "low": "#F44336"}[confidence_level],
                bordercolor="white",
                borderwidth=1,
                borderpad=4
            )
            
            # Enhanced container with confidence styling
            confidence_class = f"confidence-{confidence_level}"
            st.markdown(f'<div class="chart-container {confidence_class}">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=False, height=600, config=get_fixed_price_chart_config())
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced Chart Legend/Key
            st.markdown("### 📊 Chart Legend & Key")
            
            # Create comprehensive legend in organized columns
            legend_col1, legend_col2, legend_col3 = st.columns(3)
            
            with legend_col1:
                st.markdown("""
                **📈 Price Lines:**
                - <span style="color: #2E86AB; font-weight: bold;">━━━ Blue Solid Line:</span> Historical Stock Prices
                - <span style="color: #8A2BE2; font-weight: bold;">━━━ Purple Solid Line:</span> AI Price Predictions
                - <span style="color: #C73E1D; font-weight: bold;">━━━ Red Horizontal Line:</span> Current Price Reference
                """, unsafe_allow_html=True)
                
                st.markdown("""
                **📊 Moving Averages:**
                - <span style="color: #A23B72; font-weight: bold;">··· Pink Dotted Line:</span> 20-Day Moving Average (Short-term trend)
                - <span style="color: #F18F01; font-weight: bold;">--- Orange Dashed Line:</span> 50-Day Moving Average (Long-term trend)
                """, unsafe_allow_html=True)
            
            with legend_col2:
                st.markdown("""
                **🌟 Special Markers:**
                - <span style="color: gold; font-size: 18px;">⭐</span> **Golden Stars:** Weekly Milestone Markers (every 7 days)
                - <span style="color: #8A2BE2;">●</span> **Purple Dots:** Individual Prediction Points
                """, unsafe_allow_html=True)
                
                st.markdown("""
                **🎨 Confidence Band:**
                - <span style="background: rgba(128, 0, 128, 0.2); padding: 2px 8px; border-radius: 4px;">Purple Shaded Area</span> Prediction Uncertainty Range (±5%)
                """, unsafe_allow_html=True)
            
            with legend_col3:
                st.markdown("""
                **🌈 Background Zones:**
                - <span style="background: rgba(0, 255, 0, 0.1); padding: 2px 8px; border-radius: 4px;">🐂 Green Zone</span> Bullish Territory (Top 1/3)
                - <span style="background: rgba(255, 255, 0, 0.1); padding: 2px 8px; border-radius: 4px;">⚖️ Yellow Zone</span> Neutral Territory (Middle 1/3)
                - <span style="background: rgba(255, 0, 0, 0.1); padding: 2px 8px; border-radius: 4px;">🐻 Red Zone</span> Bearish Territory (Bottom 1/3)
                """, unsafe_allow_html=True)
                
                st.markdown("""
                **🎯 Confidence Indicator:**
                - **Chart Corner:** Shows AI prediction confidence level
                - **Green:** High Confidence (75%+)
                - **Orange:** Medium Confidence (50-75%)
                - **Red:** Low Confidence (<50%)
                """, unsafe_allow_html=True)
            
            # Visual legend with actual chart colors
            st.markdown("### 🎨 Visual Color Reference")
            
            # Create a mini visual reference
            color_ref_col1, color_ref_col2, color_ref_col3, color_ref_col4 = st.columns(4)
            
            with color_ref_col1:
                st.markdown("""
                <div style="
                    background: linear-gradient(90deg, #2E86AB, #2E86AB);
                    height: 20px; 
                    border-radius: 10px; 
                    margin: 5px 0;
                "></div>
                <small><strong>Historical Prices</strong></small>
                """, unsafe_allow_html=True)
            
            with color_ref_col2:
                st.markdown("""
                <div style="
                    background: linear-gradient(90deg, #8A2BE2, #8A2BE2);
                    height: 20px; 
                    border-radius: 10px; 
                    margin: 5px 0;
                "></div>
                <small><strong>AI Predictions</strong></small>
                """, unsafe_allow_html=True)
            
            with color_ref_col3:
                st.markdown("""
                <div style="
                    background: linear-gradient(90deg, #A23B72, #F18F01);
                    height: 20px; 
                    border-radius: 10px; 
                    margin: 5px 0;
                "></div>
                <small><strong>Moving Averages</strong></small>
                """, unsafe_allow_html=True)
            
            with color_ref_col4:
                st.markdown("""
                <div style="
                    background: linear-gradient(90deg, gold, orange);
                    height: 20px; 
                    border-radius: 10px; 
                    margin: 5px 0;
                "></div>
                <small><strong>Weekly Milestones</strong></small>
                """, unsafe_allow_html=True)
            
            # Enhanced chart features explanation
            if beginner_mode:
                with st.expander("🔍 Interactive Chart Guide - How to Read Everything"):
                    st.markdown("""
                    ### 📖 **Complete Chart Reading Guide:**
                    
                    **🎯 What Each Element Tells You:**
                    
                    1. **📈 Blue Historical Line:** 
                       - Shows actual past stock prices
                       - Look for trends: going up = bullish, going down = bearish
                    
                    2. **🔮 Purple Prediction Line:**
                       - AI's best guess for future prices
                       - Higher than current = growth expected
                       - Lower than current = decline expected
                    
                    3. **🌫️ Purple Shaded Area (Confidence Band):**
                       - Shows uncertainty in predictions
                       - Wider band = more uncertain
                       - Narrower band = more confident
                    
                    4. **⭐ Golden Stars (Weekly Milestones):**
                       - Mark important weekly targets
                       - Use these for planning buy/sell timing
                    
                    5. **📊 Moving Average Lines:**
                       - **Pink Dotted (20-day):** Short-term trend direction
                       - **Orange Dashed (50-day):** Long-term trend direction
                       - When short-term crosses above long-term = bullish signal
                    
                    6. **🌈 Background Color Zones:**
                       - **Green Zone:** Good price levels (higher prices)
                       - **Yellow Zone:** Neutral price levels (middle range)
                       - **Red Zone:** Lower price levels (could be bargains or problems)
                    
                    7. **🔴 Red Horizontal Line:**
                       - Today's current price
                       - Everything above = gains, everything below = losses
                    
                    **💡 How to Use This Information:**
                    - **Buying:** Look for predictions trending upward with narrow confidence bands
                    - **Selling:** Consider when predictions peak or start declining
                    - **Timing:** Use weekly milestones for specific entry/exit dates
                    - **Risk:** Wider confidence bands = higher uncertainty
                    """)
                    
                    st.success("""
                    🎯 **Quick Decision Framework:**
                    
                    **🟢 Good to Buy When:**
                    - Purple line trending upward
                    - Narrow confidence bands
                    - Price in green or yellow zones
                    - High confidence indicator
                    
                    **🔴 Consider Selling When:**
                    - Purple line peaks or declines
                    - Price reaching red zones
                    - Wide confidence bands
                    - Low confidence indicator
                    """)
                    
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **📊 Chart Elements:**
                    - **Blue Line**: Historical stock prices
                    - **Purple Line**: AI predictions for future
                    - **Purple Shaded Area**: Confidence bands (uncertainty range)
                    - **Golden Stars**: Weekly milestone markers
                    """)
                with col2:
                    st.markdown("""
                        **📈 Technical Indicators:**
                        - **Dotted Line**: 20-day moving average (short-term trend)
                        - **Dashed Line**: 50-day moving average (long-term trend)
                        - **Red Line**: Current price reference
                        - **Color Zones**: Bullish (green), Neutral (yellow), Bearish (red)
                        """)
                    
                    st.info("💡 **Tip**: Hover over any point for detailed information!")
            
            # Chart insights summary
            st.markdown("### 🎯 Chart Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_direction = "📈 Upward" if pred_prices.iloc[-1] > current_price else "📉 Downward"
                st.metric("Trend Direction", trend_direction)
            
            with col2:
                price_change = ((pred_prices.iloc[-1] - current_price) / current_price) * 100
                st.metric("30-Day Outlook", f"{price_change:+.1f}%")
            
            with col3:
                volatility_level = "High" if abs(price_change) > 10 else "Moderate" if abs(price_change) > 5 else "Low"
                st.metric("Volatility", volatility_level)
            
            # NEW: Prediction Timeline Chart
            st.markdown('<div class="section-header">📅 Prediction Timeline Chart</div>', unsafe_allow_html=True)
            if beginner_mode:
                st.info("📊 **Timeline explanation**: Shows percentage change from today's price over time. Green zones = good times to buy, Red zones = consider selling")
            
            # Create prediction timeline chart
            timeline_fig = go.Figure()
            
            # Calculate percentage changes from current price
            pred_dates = predictions['Date'][:30]
            pred_prices = predictions['Predicted_Price'][:30]
            percentage_changes = ((pred_prices - current_price) / current_price * 100)
            
            # Create color-coded markers based on performance
            colors = []
            buy_sell_signals = []
            for pct in percentage_changes:
                if pct >= 10:
                    colors.append('#00C851')  # Strong Green - Strong Buy
                    buy_sell_signals.append('🟢 Strong Buy')
                elif pct >= 5:
                    colors.append('#39C0ED')  # Light Blue - Buy
                    buy_sell_signals.append('🔵 Buy')
                elif pct >= 0:
                    colors.append('#FFD700')  # Gold - Hold
                    buy_sell_signals.append('🟡 Hold')
                elif pct >= -5:
                    colors.append('#FF8800')  # Orange - Caution
                    buy_sell_signals.append('🟠 Caution')
                else:
                    colors.append('#FF4444')  # Red - Sell
                    buy_sell_signals.append('🔴 Sell')
            
            # Add the main timeline line
            timeline_fig.add_trace(go.Scatter(
                x=pred_dates,
                y=percentage_changes,
                mode='lines+markers',
                name='Price Change Timeline',
                line=dict(color='#8A2BE2', width=3),
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                text=[f"{sig}<br>{pct:+.1f}%" for sig, pct in zip(buy_sell_signals, percentage_changes)],
                hovertemplate='<b>Date:</b> %{x}<br><b>Change:</b> %{y:+.1f}%<br><b>Signal:</b> %{text}<br><extra></extra>'
            ))
            
            # Add break-even line at 0%
            timeline_fig.add_hline(
                y=0,
                line_dash="solid",
                line_color="black",
                line_width=2,
                annotation_text="Break-Even Line (0%)",
                annotation_position="top right"
            )
            
            # Add recommendation zones
            max_change = max(percentage_changes.max(), abs(percentage_changes.min()))
            
            # Strong Buy Zone (>10%)
            timeline_fig.add_hrect(
                y0=10,
                y1=max_change + 5,
                fillcolor="rgba(0, 200, 81, 0.15)",
                layer="below",
                line_width=0,
                annotation_text="🟢 Strong Buy Zone (>10%)",
                annotation_position="top left"
            )
            
            # Buy Zone (5-10%)
            timeline_fig.add_hrect(
                y0=5,
                y1=10,
                fillcolor="rgba(57, 192, 237, 0.15)",
                layer="below",
                line_width=0,
                annotation_text="🔵 Buy Zone (5-10%)",
                annotation_position="top left"
            )
            
            # Hold Zone (0-5%)
            timeline_fig.add_hrect(
                y0=0,
                y1=5,
                fillcolor="rgba(255, 215, 0, 0.15)",
                layer="below",
                line_width=0,
                annotation_text="🟡 Hold Zone (0-5%)",
                annotation_position="top left"
            )
            
            # Caution Zone (0 to -5%)
            timeline_fig.add_hrect(
                y0=-5,
                y1=0,
                fillcolor="rgba(255, 136, 0, 0.15)",
                layer="below",
                line_width=0,
                annotation_text="🟠 Caution Zone (0 to -5%)",
                annotation_position="bottom left"
            )
            
            # Sell Zone (<-5%)
            timeline_fig.add_hrect(
                y0=-(max_change + 5),
                y1=-5,
                fillcolor="rgba(255, 68, 68, 0.15)",
                layer="below",
                line_width=0,
                annotation_text="🔴 Sell Zone (<-5%)",
                annotation_position="bottom left"
            )
            
            # Add weekly milestone markers on timeline
            weekly_timeline_milestones = []
            for i, (date, pct) in enumerate(zip(pred_dates, percentage_changes)):
                if i % 7 == 0:  # Every 7 days
                    weekly_timeline_milestones.append((date, pct))
            
            if weekly_timeline_milestones:
                milestone_dates_tl, milestone_pcts = zip(*weekly_timeline_milestones)
                timeline_fig.add_trace(go.Scatter(
                    x=milestone_dates_tl,
                    y=milestone_pcts,
                    mode='markers',
                    name='Weekly Milestones',
                    marker=dict(
                        symbol='star',
                        size=18,
                        color='gold',
                        line=dict(color='orange', width=3)
                    ),
                    hovertemplate='<b>⭐ Weekly Milestone</b><br><b>Date:</b> %{x}<br><b>Change:</b> %{y:+.1f}%<br><extra></extra>'
                ))
            
            # Enhanced timeline layout with technical improvements
            timeline_enhanced_config = create_enhanced_layout_config(
                f"📅 {symbol} - Prediction Timeline & Trading Signals", 500
            )
            timeline_fig.update_layout(**timeline_enhanced_config)
            
            # Add side-by-side chart comparison layout
            st.markdown('<div class="chart-comparison">', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(timeline_fig, use_container_width=True, config=get_enhanced_chart_config())
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Timeline insights and recommendations
            if beginner_mode:
                with st.expander("📊 Timeline Chart Guide"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **📅 Understanding the Timeline:**
                        - **X-axis**: Future dates (when predictions occur)
                        - **Y-axis**: Percentage change from today's price
                        - **Colored Dots**: Each represents a different trading signal
                        - **Golden Stars**: Weekly milestone markers
                        """)
                    with col2:
                        st.markdown("""
                        **🎨 Color Code Guide:**
                        - **🟢 Dark Green**: Strong Buy signal (>10% gain expected)
                        - **🔵 Blue**: Buy signal (5-10% gain expected)
                        - **🟡 Yellow**: Hold signal (0-5% gain expected)
                        - **🟠 Orange**: Caution signal (0 to -5% change)
                        - **🔴 Red**: Sell signal (>5% loss expected)
                        """)
                    
                    st.info("💡 **How to use this**: Look for clusters of green dots for good buying opportunities, and red dots for potential selling points!")
            
            # Trading signal summary
            st.markdown("### 🎯 Trading Signal Summary")
            
            # Count signals
            signal_counts = {}
            for signal in buy_sell_signals:
                signal_type = signal.split()[1]  # Extract Buy, Sell, Hold, etc.
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                buy_signals = signal_counts.get('Buy', 0) + signal_counts.get('Buy', 0)  # Strong Buy + Buy
                st.metric("🟢 Buy Signals", f"{buy_signals} days")
            
            with col2:
                hold_signals = signal_counts.get('Hold', 0)
                st.metric("🟡 Hold Signals", f"{hold_signals} days")
            
            with col3:
                caution_signals = signal_counts.get('Caution', 0)
                st.metric("🟠 Caution Signals", f"{caution_signals} days")
            
            with col4:
                sell_signals = signal_counts.get('Sell', 0)
                st.metric("🔴 Sell Signals", f"{sell_signals} days")
            
            # Best and worst days
            best_day_idx = percentage_changes.idxmax()
            worst_day_idx = percentage_changes.idxmin()
            best_day_change = percentage_changes.iloc[best_day_idx]
            worst_day_change = percentage_changes.iloc[worst_day_idx]
            
            st.markdown("### 📈 Key Prediction Dates")
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **🎯 Best Predicted Day**
                - **Date**: {pred_dates.iloc[best_day_idx].strftime('%B %d, %Y')}
                - **Expected Change**: +{best_day_change:.1f}%
                - **Predicted Price**: ${pred_prices.iloc[best_day_idx]:.2f}
                """)
            
            with col2:
                st.error(f"""
                **⚠️ Most Challenging Day**
                - **Date**: {pred_dates.iloc[worst_day_idx].strftime('%B %d, %Y')}
                - **Expected Change**: {worst_day_change:.1f}%
                - **Predicted Price**: ${pred_prices.iloc[worst_day_idx]:.2f}
                """)
            
            # NEW: Weekly Targets Bar Chart
            st.markdown('<div class="section-header">📊 Weekly Targets Bar Chart</div>', unsafe_allow_html=True)
            if beginner_mode:
                st.info("📊 **Weekly targets explanation**: Each bar shows predicted price for that week. Green bars = price going up, Red bars = price going down from current level")
            
            # Create weekly targets bar chart
            weekly_fig = go.Figure()
            
            # Group predictions into weeks and calculate weekly targets
            pred_dates = predictions['Date'][:30]
            pred_prices = predictions['Predicted_Price'][:30]
            
            # Create weekly groupings (every 7 days)
            weekly_targets = []
            weekly_labels = []
            weekly_changes = []
            weekly_percentages = []
            
            for i in range(0, min(len(pred_dates), 28), 7):  # 4 weeks max
                week_end_idx = min(i + 6, len(pred_dates) - 1)
                week_target_price = pred_prices.iloc[week_end_idx]
                week_start_date = pred_dates.iloc[i]
                week_end_date = pred_dates.iloc[week_end_idx]
                
                # Calculate change from current price
                price_change = week_target_price - current_price
                percentage_change = (price_change / current_price) * 100
                
                weekly_targets.append(week_target_price)
                weekly_changes.append(price_change)
                weekly_percentages.append(percentage_change)
                
                # Create week label
                week_label = f"Week {len(weekly_targets)}\n({week_start_date.strftime('%m/%d')} - {week_end_date.strftime('%m/%d')})"
                weekly_labels.append(week_label)
            
            # Create color-coded bars
            bar_colors = []
            for change in weekly_changes:
                if change >= current_price * 0.1:  # >10% gain
                    bar_colors.append('#00C851')  # Strong Green
                elif change >= current_price * 0.05:  # 5-10% gain
                    bar_colors.append('#4CAF50')  # Green
                elif change >= 0:  # 0-5% gain
                    bar_colors.append('#8BC34A')  # Light Green
                elif change >= -current_price * 0.05:  # 0-5% loss
                    bar_colors.append('#FF9800')  # Orange
                else:  # >5% loss
                    bar_colors.append('#F44336')  # Red
            
            # Add the weekly target bars
            weekly_fig.add_trace(go.Bar(
                x=weekly_labels,
                y=weekly_targets,
                name='Weekly Price Targets',
                marker=dict(
                    color=bar_colors,
                    line=dict(color='white', width=2)
                ),
                text=[f"${price:.2f}<br>{change:+.1f}%<br>${abs(weekly_changes[i]):+.2f}" 
                      for i, (price, change) in enumerate(zip(weekly_targets, weekly_percentages))],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>' +
                             '<b>Target Price:</b> $%{y:.2f}<br>' +
                             '<b>Change:</b> %{text}<br>' +
                             '<extra></extra>'
            ))
            
            # Add current price reference line
            weekly_fig.add_hline(
                y=current_price,
                line_dash="solid",
                line_color="#2E86AB",
                line_width=3,
                annotation_text=f"Current Price: ${current_price:.2f}",
                annotation_position="top right",
                annotation=dict(
                    bgcolor="white",
                    bordercolor="#2E86AB",
                    borderwidth=2
                )
            )
            
            # Add target zones
            max_target = max(weekly_targets)
            min_target = min(weekly_targets)
            price_range = max_target - min_target
            
            # Bullish zone (above current price)
            if max_target > current_price:
                weekly_fig.add_hrect(
                    y0=current_price,
                    y1=max_target + price_range * 0.1,
                    fillcolor="rgba(76, 175, 80, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="🐂 Bullish Territory",
                    annotation_position="top left"
                )
            
            # Bearish zone (below current price)
            if min_target < current_price:
                weekly_fig.add_hrect(
                    y0=min_target - price_range * 0.1,
                    y1=current_price,
                    fillcolor="rgba(244, 67, 54, 0.1)",
                    layer="below",
                    line_width=0,
                    annotation_text="🐻 Bearish Territory",
                    annotation_position="bottom left"
                )
            
            # Enhanced weekly chart layout with technical improvements
            weekly_enhanced_config = create_enhanced_layout_config(
                f"📊 {symbol} - Weekly Price Targets & Performance", 450
            )
            weekly_fig.update_layout(**weekly_enhanced_config)
            weekly_fig.update_layout(showlegend=False)  # Override for bar chart
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(weekly_fig, use_container_width=True, config=get_enhanced_chart_config())
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Weekly targets summary
            st.markdown("### 📈 Weekly Performance Outlook")
            
            cols = st.columns(len(weekly_targets))
            
            for i, (week_label, target, change, pct_change) in enumerate(zip(weekly_labels, weekly_targets, weekly_changes, weekly_percentages)):
                with cols[i]:
                    # Determine performance indicator
                    if pct_change >= 10:
                        performance = "🟢 Excellent"
                        performance_color = "success"
                    elif pct_change >= 5:
                        performance = "🔵 Good"
                        performance_color = "info"
                    elif pct_change >= 0:
                        performance = "🟡 Positive"
                        performance_color = "warning"
                    elif pct_change >= -5:
                        performance = "🟠 Cautious"
                        performance_color = "warning"
                    else:
                        performance = "🔴 Negative"
                        performance_color = "error"
                    
                    # Clean up week label for display
                    week_num = week_label.split('\n')[0]
                    date_range = week_label.split('\n')[1]
                    
                    st.markdown(f"""
                    <div style="
                        border: 2px solid {bar_colors[i]};
                        border-radius: 10px;
                        padding: 15px;
                        text-align: center;
                        background-color: rgba(255,255,255,0.1);
                        margin: 5px;
                    ">
                        <h4 style="margin: 0; color: {bar_colors[i]};">{week_num}</h4>
                        <p style="margin: 5px 0; font-size: 12px;">{date_range}</p>
                        <h3 style="margin: 5px 0; color: {bar_colors[i]};">${target:.2f}</h3>
                        <p style="margin: 0; font-weight: bold; color: {bar_colors[i]};">{pct_change:+.1f}%</p>
                        <p style="margin: 5px 0; font-size: 11px;">{performance}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Weekly insights for beginners
            if beginner_mode:
                with st.expander("📊 Weekly Bar Chart Guide"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **📊 Reading the Bar Chart:**
                        - **Height of Bar**: Predicted price for that week
                        - **Bar Color**: Performance indicator
                        - **Blue Line**: Current price reference
                        - **Text on Bars**: Target price and percentage change
                        """)
                    with col2:
                        st.markdown("""
                        **🎨 Bar Color Meanings:**
                        - **🟢 Dark Green**: Excellent gains (>10%)
                        - **🔵 Green**: Good gains (5-10%)
                        - **🟡 Light Green**: Small gains (0-5%)
                        - **🟠 Orange**: Small losses (0-5%)
                        - **🔴 Red**: Larger losses (>5%)
                        """)
                    
                    st.info("💡 **Trading Strategy**: Look for consistently green bars for good buying opportunities, and red bars might indicate selling points!")
            
            # Best and worst week analysis
            best_week_idx = weekly_percentages.index(max(weekly_percentages))
            worst_week_idx = weekly_percentages.index(min(weekly_percentages))
            
            st.markdown("### 🎯 Key Weekly Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"""
                **🏆 Best Week Predicted**
                - **{weekly_labels[best_week_idx].split('(')[0].strip()}**
                - **Target**: ${weekly_targets[best_week_idx]:.2f}
                - **Expected Gain**: +{weekly_percentages[best_week_idx]:.1f}%
                """)
            
            with col2:
                st.error(f"""
                **⚠️ Most Challenging Week**
                - **{weekly_labels[worst_week_idx].split('(')[0].strip()}**
                - **Target**: ${weekly_targets[worst_week_idx]:.2f}
                - **Expected Change**: {weekly_percentages[worst_week_idx]:.1f}%
                """)
            
            with col3:
                avg_weekly_change = sum(weekly_percentages) / len(weekly_percentages)
                positive_weeks = sum(1 for pct in weekly_percentages if pct > 0)
                
                if avg_weekly_change > 5:
                    outlook = "🟢 Very Positive"
                elif avg_weekly_change > 0:
                    outlook = "🔵 Positive"
                elif avg_weekly_change > -5:
                    outlook = "🟡 Mixed"
                else:
                    outlook = "🔴 Challenging"
                
                st.info(f"""
                **📊 Overall Weekly Outlook**
                - **{outlook}**
                - **Avg Change**: {avg_weekly_change:+.1f}%
                - **Positive Weeks**: {positive_weeks}/{len(weekly_targets)}
                """)
            
            # NEW: Risk & Return Gauge Dashboard
            st.markdown('<div class="section-header">🎮 Risk & Return Gauge Dashboard</div>', unsafe_allow_html=True)
            if beginner_mode:
                st.info("🎮 **Gauge explanation**: These speedometer-like gauges show your risk level and expected returns. Green = good/safe, Yellow = moderate, Red = high risk/poor returns")
            
            # Calculate metrics for gauges
            expected_30day_return = ((pred_prices.iloc[-1] - current_price) / current_price) * 100
            risk_score = min(risk_metrics['volatility'] * 2, 100)  # Scale volatility to 0-100
            
            # Create gauge dashboard
            gauge_col1, gauge_col2 = st.columns(2)
            
            with gauge_col1:
                # Expected Return Gauge
                return_fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = expected_30day_return,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "📈 Expected 30-Day Return (%)", 'font': {'size': 16, 'color': '#2E86AB'}},
                    delta = {'reference': 5, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge = {
                        'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#8A2BE2", 'thickness': 0.3},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [-20, 0], 'color': "#FFD6D6"},    # Light Red - Losses
                            {'range': [0, 5], 'color': "#FFF8DC"},      # Light Yellow - Low gains
                            {'range': [5, 15], 'color': "#D6FFD6"},     # Light Green - Good gains
                            {'range': [15, 50], 'color': "#90EE90"}     # Green - Excellent gains
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 10  # Target return threshold
                        }
                    }
                ))
                
                return_fig.update_layout(
                    height=350,
                    font={'color': "darkblue", 'family': "Arial"},
                    paper_bgcolor='rgba(248,249,250,0.95)',
                    plot_bgcolor='rgba(255,255,255,0.8)',
                    margin=dict(l=20, r=20, t=80, b=20)
                )
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(return_fig, use_container_width=True, config=get_enhanced_chart_config())
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Return interpretation
                if expected_30day_return >= 15:
                    return_status = "🟢 Excellent"
                    return_desc = "Outstanding expected returns!"
                elif expected_30day_return >= 5:
                    return_status = "🔵 Good"
                    return_desc = "Solid positive returns expected"
                elif expected_30day_return >= 0:
                    return_status = "🟡 Modest"
                    return_desc = "Small gains expected"
                elif expected_30day_return >= -5:
                    return_status = "🟠 Caution"
                    return_desc = "Small losses possible"
                else:
                    return_status = "🔴 High Risk"
                    return_desc = "Significant losses possible"
                
                st.markdown(f"""
                <div style="
                    border: 2px solid {'#4CAF50' if expected_30day_return >= 5 else '#FF9800' if expected_30day_return >= 0 else '#F44336'};
                    border-radius: 15px;
                    padding: 15px;
                    text-align: center;
                    background: linear-gradient(135deg, {'rgba(76, 175, 80, 0.1)' if expected_30day_return >= 5 else 'rgba(255, 152, 0, 0.1)' if expected_30day_return >= 0 else 'rgba(244, 67, 54, 0.1)'});
                    margin: 10px 0;
                ">
                    <h4 style="margin: 0; color: {'#4CAF50' if expected_30day_return >= 5 else '#FF9800' if expected_30day_return >= 0 else '#F44336'};">Return Assessment</h4>
                    <h3 style="margin: 5px 0; color: {'#4CAF50' if expected_30day_return >= 5 else '#FF9800' if expected_30day_return >= 0 else '#F44336'};">{return_status}</h3>
                    <p style="margin: 0; font-size: 14px;">{return_desc}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with gauge_col2:
                # Risk Score Gauge
                risk_fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "⚠️ Risk Score (0-100)", 'font': {'size': 16, 'color': '#2E86AB'}},
                    delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#FF6B6B", 'thickness': 0.3},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 25], 'color': "#D6FFD6"},      # Green - Low Risk
                            {'range': [25, 50], 'color': "#FFF8DC"},     # Yellow - Medium Risk
                            {'range': [50, 75], 'color': "#FFE4B5"},     # Orange - High Risk
                            {'range': [75, 100], 'color': "#FFD6D6"}     # Red - Very High Risk
                        ],
                        'threshold': {
                            'line': {'color': "orange", 'width': 4},
                            'thickness': 0.75,
                            'value': 40  # Acceptable risk threshold
                        }
                    }
                ))
                
                risk_fig.update_layout(
                    height=350,
                    font={'color': "darkblue", 'family': "Arial"},
                    paper_bgcolor='rgba(248,249,250,0.95)',
                    plot_bgcolor='rgba(255,255,255,0.8)',
                    margin=dict(l=20, r=20, t=80, b=20)
                )
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(risk_fig, use_container_width=True, config=get_enhanced_chart_config())
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Risk interpretation
                if risk_score <= 25:
                    risk_status = "🟢 Low Risk"
                    risk_desc = "Conservative investment"
                elif risk_score <= 50:
                    risk_status = "🟡 Medium Risk"
                    risk_desc = "Moderate volatility expected"
                elif risk_score <= 75:
                    risk_status = "🟠 High Risk"
                    risk_desc = "Significant price swings likely"
                else:
                    risk_status = "🔴 Very High Risk"
                    risk_desc = "Extreme volatility expected"
                
                st.markdown(f"""
                <div style="
                    border: 2px solid {'#4CAF50' if risk_score <= 25 else '#FFC107' if risk_score <= 50 else '#FF9800' if risk_score <= 75 else '#F44336'};
                    border-radius: 15px;
                    padding: 15px;
                    text-align: center;
                    background: linear-gradient(135deg, {'rgba(76, 175, 80, 0.1)' if risk_score <= 25 else 'rgba(255, 193, 7, 0.1)' if risk_score <= 50 else 'rgba(255, 152, 0, 0.1)' if risk_score <= 75 else 'rgba(244, 67, 54, 0.1)'});
                    margin: 10px 0;
                ">
                    <h4 style="margin: 0; color: {'#4CAF50' if risk_score <= 25 else '#FFC107' if risk_score <= 50 else '#FF9800' if risk_score <= 75 else '#F44336'};">Risk Assessment</h4>
                    <h3 style="margin: 5px 0; color: {'#4CAF50' if risk_score <= 25 else '#FFC107' if risk_score <= 50 else '#FF9800' if risk_score <= 75 else '#F44336'};">{risk_status}</h3>
                    <p style="margin: 0; font-size: 14px;">{risk_desc}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk-Return Matrix Analysis
            st.markdown("### 🎯 Risk-Return Analysis Matrix")
            
            # Create risk-return matrix
            matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
            
            with matrix_col1:
                # Investment Profile
                if expected_30day_return >= 10 and risk_score <= 40:
                    profile = "🏆 Ideal Investment"
                    profile_color = "#4CAF50"
                    profile_desc = "High returns with manageable risk"
                elif expected_30day_return >= 5 and risk_score <= 60:
                    profile = "👍 Good Opportunity"
                    profile_color = "#2196F3"
                    profile_desc = "Decent returns with acceptable risk"
                elif expected_30day_return >= 0 and risk_score <= 50:
                    profile = "⚖️ Balanced Option"
                    profile_color = "#FF9800"
                    profile_desc = "Modest returns with controlled risk"
                elif expected_30day_return < 0 and risk_score <= 30:
                    profile = "🛡️ Capital Preservation"
                    profile_color = "#9C27B0"
                    profile_desc = "Low risk but limited upside"
                else:
                    profile = "⚠️ High Risk/Reward"
                    profile_color = "#F44336"
                    profile_desc = "Significant risk with uncertain returns"
                
                st.markdown(f"""
                <div style="
                    border: 3px solid {profile_color};
                    border-radius: 20px;
                    padding: 20px;
                    text-align: center;
                    background: linear-gradient(135deg, {profile_color}22, {profile_color}11);
                    margin: 10px 0;
                ">
                    <h4 style="margin: 0; color: {profile_color};">Investment Profile</h4>
                    <h2 style="margin: 10px 0; color: {profile_color};">{profile}</h2>
                    <p style="margin: 0; font-size: 14px; color: {profile_color};">{profile_desc}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with matrix_col2:
                # Risk-Adjusted Score
                if risk_score > 0:
                    risk_adjusted_return = expected_30day_return / (risk_score / 20)  # Normalize risk score
                else:
                    risk_adjusted_return = expected_30day_return
                
                if risk_adjusted_return >= 5:
                    adj_status = "🟢 Excellent"
                    adj_color = "#4CAF50"
                elif risk_adjusted_return >= 2:
                    adj_status = "🔵 Good" 
                    adj_color = "#2196F3"
                elif risk_adjusted_return >= 0:
                    adj_status = "🟡 Fair"
                    adj_color = "#FF9800"
                else:
                    adj_status = "🔴 Poor"
                    adj_color = "#F44336"
                
                st.markdown(f"""
                <div style="
                    border: 2px solid {adj_color};
                    border-radius: 15px;
                    padding: 20px;
                    text-align: center;
                    background: linear-gradient(135deg, {adj_color}22, {adj_color}11);
                    margin: 10px 0;
                ">
                    <h4 style="margin: 0; color: {adj_color};">Risk-Adjusted Return</h4>
                    <h2 style="margin: 10px 0; color: {adj_color};">{adj_status}</h2>
                    <h3 style="margin: 5px 0; color: {adj_color};">{risk_adjusted_return:.1f}</h3>
                    <p style="margin: 0; font-size: 12px; color: {adj_color};">Return per unit of risk</p>
                </div>
                """, unsafe_allow_html=True)
            
            with matrix_col3:
                # Recommendation
                if expected_30day_return >= 8 and risk_score <= 50:
                    recommendation = "🟢 BUY"
                    rec_color = "#4CAF50"
                    rec_desc = "Strong buy signal"
                elif expected_30day_return >= 3 and risk_score <= 60:
                    recommendation = "🔵 CONSIDER"
                    rec_color = "#2196F3"
                    rec_desc = "Worth considering"
                elif expected_30day_return >= 0:
                    recommendation = "🟡 HOLD"
                    rec_color = "#FF9800"
                    rec_desc = "Monitor closely"
                else:
                    recommendation = "🔴 AVOID"
                    rec_color = "#F44336"
                    rec_desc = "High risk, avoid"
                
                st.markdown(f"""
                <div style="
                    border: 3px solid {rec_color};
                    border-radius: 20px;
                    padding: 20px;
                    text-align: center;
                    background: linear-gradient(135deg, {rec_color}33, {rec_color}22);
                    margin: 10px 0;
                ">
                    <h4 style="margin: 0; color: {rec_color};">AI Recommendation</h4>
                    <h1 style="margin: 10px 0; color: {rec_color};">{recommendation}</h1>
                    <p style="margin: 0; font-size: 14px; color: {rec_color};">{rec_desc}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Gauge explanation for beginners
            if beginner_mode:
                with st.expander("🎮 Gauge Dashboard Guide"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **📈 Return Gauge Explained:**
                        - **Green Zone (5-50%)**: Excellent expected returns
                        - **Yellow Zone (0-5%)**: Small positive returns
                        - **Red Zone (Below 0%)**: Expected losses
                        - **Red Needle**: Target return threshold (10%)
                        """)
                    with col2:
                        st.markdown("""
                        **⚠️ Risk Gauge Explained:**
                        - **Green Zone (0-25)**: Low risk, stable stock
                        - **Yellow Zone (25-50)**: Medium risk, some volatility
                        - **Orange Zone (50-75)**: High risk, volatile
                        - **Red Zone (75-100)**: Very high risk, extreme volatility
                        """)
                    
                    st.info("💡 **Investment Tip**: Ideal investments have high returns (green) with low risk (green). Avoid high risk with low returns!")
            
            # NEW: Detailed Prediction Timeline Table
            st.markdown('<div class="section-header">📋 Detailed Prediction Timeline Table</div>', unsafe_allow_html=True)
            if beginner_mode:
                st.info("📋 **Table explanation**: This detailed table shows exactly when to buy or sell, with precise dates, prices, and AI recommendations for each week")
            
            # Create comprehensive timeline data
            timeline_data = []
            base_date = pd.Timestamp.now().tz_localize(None)  # Make timezone-naive
            
            # Generate weekly data points (every 7 days for 4 weeks)
            for week in range(4):
                week_start_idx = week * 7
                week_end_idx = min(week_start_idx + 6, len(pred_dates) - 1)
                
                if week_end_idx < len(pred_dates):
                    target_date = pred_dates.iloc[week_end_idx]
                    # Ensure target_date is timezone-naive
                    if hasattr(target_date, 'tz_localize'):
                        target_date = target_date.tz_localize(None)
                    elif hasattr(target_date, 'tz'):
                        target_date = target_date.tz_localize(None) if target_date.tz is None else target_date.tz_convert(None).tz_localize(None)
                    
                    target_price = pred_prices.iloc[week_end_idx]
                    price_change = target_price - current_price
                    percentage_change = (price_change / current_price) * 100
                    
                    # Calculate days from now
                    days_from_now = (target_date - base_date).days
                    
                    # Generate AI signal based on percentage change
                    if percentage_change >= 10:
                        signal = "🟢 Strong Buy"
                        signal_color = "#4CAF50"
                        row_color = "rgba(76, 175, 80, 0.1)"
                    elif percentage_change >= 5:
                        signal = "🔵 Buy"
                        signal_color = "#2196F3"
                        row_color = "rgba(33, 150, 243, 0.1)"
                    elif percentage_change >= 0:
                        signal = "🟡 Hold"
                        signal_color = "#FF9800"
                        row_color = "rgba(255, 152, 0, 0.1)"
                    elif percentage_change >= -5:
                        signal = "🟠 Caution"
                        signal_color = "#FF5722"
                        row_color = "rgba(255, 87, 34, 0.1)"
                    else:
                        signal = "🔴 Sell"
                        signal_color = "#F44336"
                        row_color = "rgba(244, 67, 54, 0.1)"
                    
                    timeline_data.append({
                        'week': f"Week {week + 1}",
                        'date': target_date.strftime("%a, %b %d, %Y"),
                        'target_price': f"${target_price:.2f}",
                        'price_change': f"${price_change:+.2f}",
                        'percentage_change': f"{percentage_change:+.1f}%",
                        'signal': signal,
                        'signal_color': signal_color,
                        'row_color': row_color,
                        'days_from_now': f"{days_from_now} days"
                    })
            
            # Add daily predictions for first week (more granular)
            daily_data = []
            for day in range(min(7, len(pred_dates))):
                target_date = pred_dates.iloc[day]
                # Ensure target_date is timezone-naive
                if hasattr(target_date, 'tz_localize'):
                    target_date = target_date.tz_localize(None)
                elif hasattr(target_date, 'tz'):
                    target_date = target_date.tz_localize(None) if target_date.tz is None else target_date.tz_convert(None).tz_localize(None)
                
                target_price = pred_prices.iloc[day]
                price_change = target_price - current_price
                percentage_change = (price_change / current_price) * 100
                
                # Calculate days from now
                days_from_now = (target_date - base_date).days
                
                # Generate AI signal
                if percentage_change >= 8:
                    signal = "🟢 Strong Buy"
                    signal_color = "#4CAF50"
                    row_color = "rgba(76, 175, 80, 0.1)"
                elif percentage_change >= 3:
                    signal = "🔵 Buy"
                    signal_color = "#2196F3"
                    row_color = "rgba(33, 150, 243, 0.1)"
                elif percentage_change >= 0:
                    signal = "🟡 Hold"
                    signal_color = "#FF9800"
                    row_color = "rgba(255, 152, 0, 0.1)"
                elif percentage_change >= -3:
                    signal = "🟠 Caution"
                    signal_color = "#FF5722"
                    row_color = "rgba(255, 87, 34, 0.1)"
                else:
                    signal = "🔴 Sell"
                    signal_color = "#F44336"
                    row_color = "rgba(244, 67, 54, 0.1)"
                
                daily_data.append({
                    'day': f"Day {day + 1}",
                    'date': target_date.strftime("%a, %b %d, %Y"),
                    'target_price': f"${target_price:.2f}",
                    'price_change': f"${price_change:+.2f}",
                    'percentage_change': f"{percentage_change:+.1f}%",
                    'signal': signal,
                    'signal_color': signal_color,
                    'row_color': row_color,
                    'days_from_now': f"{days_from_now} days"
                })
            
            # Create tabs for different timeframes
            timeline_tab1, timeline_tab2 = st.tabs(["📊 Weekly Predictions", "📅 Daily Predictions (Next 7 Days)"])
            
            with timeline_tab1:
                st.markdown("#### 📊 Weekly Prediction Timeline")
                
                # Create styled table for weekly data
                table_html = """
                <style>
                .prediction-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 14px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    border-radius: 10px;
                    overflow: hidden;
                }
                .prediction-table th {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    font-weight: bold;
                    padding: 15px 10px;
                    text-align: center;
                    border: none;
                }
                .prediction-table td {
                    padding: 12px 10px;
                    text-align: center;
                    border: 1px solid #ddd;
                    font-weight: 500;
                }
                .prediction-table tr:hover {
                    background-color: rgba(0,0,0,0.05);
                    transform: scale(1.01);
                    transition: all 0.2s ease;
                }
                </style>
                
                <table class="prediction-table">
                    <thead>
                        <tr>
                            <th>📅 Week</th>
                            <th>📆 Target Date</th>
                            <th>💰 Target Price</th>
                            <th>📈 Price Change</th>
                            <th>📊 % Change</th>
                            <th>🎯 AI Signal</th>
                            <th>⏰ Timeline</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for data in timeline_data:
                    table_html += f"""
                        <tr style="background-color: {data['row_color']};">
                            <td><strong>{data['week']}</strong></td>
                            <td>{data['date']}</td>
                            <td><strong>{data['target_price']}</strong></td>
                            <td style="color: {'green' if '+' in data['price_change'] else 'red'};">{data['price_change']}</td>
                            <td style="color: {'green' if '+' in data['percentage_change'] else 'red'};">{data['percentage_change']}</td>
                            <td style="color: {data['signal_color']}; font-weight: bold;">{data['signal']}</td>
                            <td>{data['days_from_now']}</td>
                        </tr>
                    """
                
                table_html += """
                    </tbody>
                </table>
                """
                
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Weekly summary statistics
                st.markdown("#### 📊 Weekly Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                buy_weeks = sum(1 for data in timeline_data if 'Buy' in data['signal'])
                hold_weeks = sum(1 for data in timeline_data if 'Hold' in data['signal'])
                caution_weeks = sum(1 for data in timeline_data if 'Caution' in data['signal'])
                sell_weeks = sum(1 for data in timeline_data if 'Sell' in data['signal'])
                
                with col1:
                    st.metric("🟢 Buy Signals", f"{buy_weeks} weeks")
                with col2:
                    st.metric("🟡 Hold Signals", f"{hold_weeks} weeks")
                with col3:
                    st.metric("🟠 Caution Signals", f"{caution_weeks} weeks")
                with col4:
                    st.metric("🔴 Sell Signals", f"{sell_weeks} weeks")
            
            with timeline_tab2:
                st.markdown("#### 📅 Daily Prediction Timeline (Next 7 Days)")
                
                # Create styled table for daily data
                daily_table_html = """
                <table class="prediction-table">
                    <thead>
                        <tr>
                            <th>📅 Day</th>
                            <th>📆 Date</th>
                            <th>💰 Target Price</th>
                            <th>📈 Price Change</th>
                            <th>📊 % Change</th>
                            <th>🎯 AI Signal</th>
                            <th>⏰ Timeline</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for data in daily_data:
                    daily_table_html += f"""
                        <tr style="background-color: {data['row_color']};">
                            <td><strong>{data['day']}</strong></td>
                            <td>{data['date']}</td>
                            <td><strong>{data['target_price']}</strong></td>
                            <td style="color: {'green' if '+' in data['price_change'] else 'red'};">{data['price_change']}</td>
                            <td style="color: {'green' if '+' in data['percentage_change'] else 'red'};">{data['percentage_change']}</td>
                            <td style="color: {data['signal_color']}; font-weight: bold;">{data['signal']}</td>
                            <td>{data['days_from_now']}</td>
                        </tr>
                    """
                
                daily_table_html += """
                    </tbody>
                </table>
                """
                
                st.markdown(daily_table_html, unsafe_allow_html=True)
                
                # Daily summary
                st.markdown("#### 📈 Daily Outlook")
                
                best_day = max(daily_data, key=lambda x: float(x['percentage_change'].replace('%', '').replace('+', '')))
                worst_day = min(daily_data, key=lambda x: float(x['percentage_change'].replace('%', '').replace('+', '')))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"""
                    **🎯 Best Day Ahead**
                    - **{best_day['day']}** ({best_day['date']})
                    - **Target**: {best_day['target_price']}
                    - **Change**: {best_day['percentage_change']}
                    - **Signal**: {best_day['signal']}
                    """)
                
                with col2:
                    st.error(f"""
                    **⚠️ Most Challenging Day**
                    - **{worst_day['day']}** ({worst_day['date']})
                    - **Target**: {worst_day['target_price']}
                    - **Change**: {worst_day['percentage_change']}
                    - **Signal**: {worst_day['signal']}
                    """)
            
            # Action plan for beginners
            if beginner_mode:
                st.markdown("### 📝 Your Action Plan")
                
                with st.expander("📋 How to Use This Timeline Table"):
                    st.markdown("""
                    **📅 Reading the Timeline:**
                    1. **Target Date**: When the prediction is for
                    2. **Target Price**: What the AI thinks the stock will cost
                    3. **Price Change**: Dollar amount difference from today
                    4. **% Change**: Percentage gain or loss from today
                    5. **AI Signal**: What the AI recommends you do
                    6. **Timeline**: How many days from now
                    
                    **🎯 Using the Signals:**
                    - **🟢 Strong Buy**: Great time to buy more shares
                    - **🔵 Buy**: Good time to buy shares
                    - **🟡 Hold**: Keep your current shares, don't buy/sell
                    - **🟠 Caution**: Be careful, watch closely
                    - **🔴 Sell**: Consider selling your shares
                    
                    **💡 Smart Strategy:**
                    - Look for clusters of green signals for buying
                    - Red signals might be good selling opportunities
                    - Use the exact dates to plan your trades
                    - Start small and learn as you go!
                    """)
                
                # Simple recommendation
                buy_signals = sum(1 for data in timeline_data if 'Buy' in data['signal'])
                total_signals = len(timeline_data)
                
                if buy_signals >= total_signals * 0.75:
                    overall_rec = "🟢 **Strong Buy Recommendation**"
                    rec_desc = "Most weeks show buying opportunities!"
                elif buy_signals >= total_signals * 0.5:
                    overall_rec = "🔵 **Moderate Buy Recommendation**"
                    rec_desc = "Several good buying opportunities ahead."
                elif buy_signals >= total_signals * 0.25:
                    overall_rec = "🟡 **Hold Recommendation**"
                    rec_desc = "Mixed signals, be cautious."
                else:
                    overall_rec = "🔴 **Avoid Recommendation**"
                    rec_desc = "Most signals suggest selling or avoiding."
                
                st.markdown(f"""
                <div style="
                    border: 3px solid {'#4CAF50' if 'Strong Buy' in overall_rec else '#2196F3' if 'Moderate Buy' in overall_rec else '#FF9800' if 'Hold' in overall_rec else '#F44336'};
                    border-radius: 15px;
                    padding: 20px;
                    text-align: center;
                    background: linear-gradient(135deg, {'rgba(76, 175, 80, 0.1)' if 'Strong Buy' in overall_rec else 'rgba(33, 150, 243, 0.1)' if 'Moderate Buy' in overall_rec else 'rgba(255, 152, 0, 0.1)' if 'Hold' in overall_rec else 'rgba(244, 67, 54, 0.1)'});
                    margin: 15px 0;
                ">
                    <h3 style="margin: 0; color: {'#4CAF50' if 'Strong Buy' in overall_rec else '#2196F3' if 'Moderate Buy' in overall_rec else '#FF9800' if 'Hold' in overall_rec else '#F44336'};">Overall Timeline Recommendation</h3>
                    <h2 style="margin: 10px 0; color: {'#4CAF50' if 'Strong Buy' in overall_rec else '#2196F3' if 'Moderate Buy' in overall_rec else '#FF9800' if 'Hold' in overall_rec else '#F44336'};">{overall_rec}</h2>
                    <p style="margin: 0; font-size: 16px;">{rec_desc}</p>
                    <p style="margin: 10px 0; font-size: 14px; opacity: 0.8;">{buy_signals} out of {total_signals} weeks show buy signals</p>
                </div>
                """, unsafe_allow_html=True)
            
            # NEW: Key Prediction Milestones
            st.markdown('<div class="section-header">🎯 Key Prediction Milestones</div>', unsafe_allow_html=True)
            if beginner_mode:
                st.info("🎯 **Milestones explanation**: These cards show your three most important future targets - next week, one month out, and the best predicted price point")
            
            # Calculate milestone data
            pred_dates = predictions['Date'][:30]
            pred_prices = predictions['Predicted_Price'][:30]
            
            # Next Week Target (7 days)
            next_week_idx = min(6, len(pred_dates) - 1)
            next_week_date = pred_dates.iloc[next_week_idx]
            next_week_price = pred_prices.iloc[next_week_idx]
            next_week_change = ((next_week_price - current_price) / current_price) * 100
            
            # One Month Target (30 days or end of predictions)
            one_month_idx = min(29, len(pred_dates) - 1)
            one_month_date = pred_dates.iloc[one_month_idx]
            one_month_price = pred_prices.iloc[one_month_idx]
            one_month_change = ((one_month_price - current_price) / current_price) * 100
            
            # Best Predicted Price (highest in the timeline)
            best_price_idx = pred_prices.idxmax()
            best_price_date = pred_dates.iloc[best_price_idx]
            best_price = pred_prices.iloc[best_price_idx]
            best_price_change = ((best_price - current_price) / current_price) * 100
            
            # Create milestone cards
            milestone_col1, milestone_col2, milestone_col3 = st.columns(3)
            
            with milestone_col1:
                # Next Week Target Card
                next_week_color = "#4CAF50" if next_week_change >= 3 else "#FF9800" if next_week_change >= 0 else "#F44336"
                next_week_status = "🟢 Bullish" if next_week_change >= 3 else "🟡 Neutral" if next_week_change >= 0 else "🔴 Bearish"
                
                st.markdown(f"""
                <div style="
                    border: 3px solid {next_week_color};
                    border-radius: 20px;
                    padding: 25px;
                    text-align: center;
                    background: linear-gradient(135deg, {next_week_color}22, {next_week_color}11);
                    margin: 15px 0;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                    transition: transform 0.2s ease;
                ">
                    <div style="font-size: 24px; margin-bottom: 10px;">📅</div>
                    <h3 style="margin: 0; color: {next_week_color}; font-size: 18px;">Next Week Target</h3>
                    <h1 style="margin: 10px 0; color: {next_week_color}; font-size: 32px;">${next_week_price:.2f}</h1>
                    <p style="margin: 5px 0; font-size: 16px; color: {next_week_color}; font-weight: bold;">{next_week_change:+.1f}%</p>
                    <p style="margin: 5px 0; font-size: 14px; color: #666;">{next_week_date.strftime('%a, %b %d, %Y')}</p>
                    <p style="margin: 10px 0; font-size: 14px; color: {next_week_color}; font-weight: bold;">{next_week_status}</p>
                    <div style="font-size: 12px; color: #888; margin-top: 10px;">
                        7-Day Outlook
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Next week details
                now_tz_naive = pd.Timestamp.now().tz_localize(None)
                next_week_tz_naive = next_week_date.tz_localize(None) if hasattr(next_week_date, 'tz_localize') else next_week_date
                days_to_target = (next_week_tz_naive - now_tz_naive).days
                st.markdown(f"""
                <div style="text-align: center; margin-top: 10px;">
                    <small style="color: #666;">
                        <strong>Target in {days_to_target} days</strong><br>
                        Expected change: ${next_week_price - current_price:+.2f}
                    </small>
                </div>
                """, unsafe_allow_html=True)
            
            with milestone_col2:
                # One Month Target Card
                one_month_color = "#4CAF50" if one_month_change >= 5 else "#FF9800" if one_month_change >= 0 else "#F44336"
                one_month_status = "🟢 Strong Growth" if one_month_change >= 10 else "🔵 Growth" if one_month_change >= 5 else "🟡 Stable" if one_month_change >= 0 else "🔴 Decline"
                
                st.markdown(f"""
                <div style="
                    border: 3px solid {one_month_color};
                    border-radius: 20px;
                    padding: 25px;
                    text-align: center;
                    background: linear-gradient(135deg, {one_month_color}22, {one_month_color}11);
                    margin: 15px 0;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                    transition: transform 0.2s ease;
                ">
                    <div style="font-size: 24px; margin-bottom: 10px;">📈</div>
                    <h3 style="margin: 0; color: {one_month_color}; font-size: 18px;">One Month Target</h3>
                    <h1 style="margin: 10px 0; color: {one_month_color}; font-size: 32px;">${one_month_price:.2f}</h1>
                    <p style="margin: 5px 0; font-size: 16px; color: {one_month_color}; font-weight: bold;">{one_month_change:+.1f}%</p>
                    <p style="margin: 5px 0; font-size: 14px; color: #666;">{one_month_date.strftime('%a, %b %d, %Y')}</p>
                    <p style="margin: 10px 0; font-size: 14px; color: {one_month_color}; font-weight: bold;">{one_month_status}</p>
                    <div style="font-size: 12px; color: #888; margin-top: 10px;">
                        30-Day Outlook
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # One month details
                one_month_tz_naive = one_month_date.tz_localize(None) if hasattr(one_month_date, 'tz_localize') else one_month_date
                days_to_month = (one_month_tz_naive - now_tz_naive).days
                st.markdown(f"""
                <div style="text-align: center; margin-top: 10px;">
                    <small style="color: #666;">
                        <strong>Target in {days_to_month} days</strong><br>
                        Expected change: ${one_month_price - current_price:+.2f}
                    </small>
                </div>
                """, unsafe_allow_html=True)
            
            with milestone_col3:
                # Best Predicted Price Card
                best_color = "#4CAF50" if best_price_change >= 10 else "#2196F3" if best_price_change >= 5 else "#FF9800"
                best_status = "🏆 Peak Target" if best_price_change >= 15 else "🎯 High Target" if best_price_change >= 10 else "📈 Good Target"
                
                st.markdown(f"""
                <div style="
                    border: 3px solid {best_color};
                    border-radius: 20px;
                    padding: 25px;
                    text-align: center;
                    background: linear-gradient(135deg, {best_color}22, {best_color}11);
                    margin: 15px 0;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                    transition: transform 0.2s ease;
                    position: relative;
                ">
                    <div style="font-size: 24px; margin-bottom: 10px;">🏆</div>
                    <h3 style="margin: 0; color: {best_color}; font-size: 18px;">Best Predicted Price</h3>
                    <h1 style="margin: 10px 0; color: {best_color}; font-size: 32px;">${best_price:.2f}</h1>
                    <p style="margin: 5px 0; font-size: 16px; color: {best_color}; font-weight: bold;">{best_price_change:+.1f}%</p>
                    <p style="margin: 5px 0; font-size: 14px; color: #666;">{best_price_date.strftime('%a, %b %d, %Y')}</p>
                    <p style="margin: 10px 0; font-size: 14px; color: {best_color}; font-weight: bold;">{best_status}</p>
                    <div style="font-size: 12px; color: #888; margin-top: 10px;">
                        Optimal Target
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Best price details
                best_price_tz_naive = best_price_date.tz_localize(None) if hasattr(best_price_date, 'tz_localize') else best_price_date
                days_to_best = (best_price_tz_naive - now_tz_naive).days
                st.markdown(f"""
                <div style="text-align: center; margin-top: 10px;">
                    <small style="color: #666;">
                        <strong>Peak in {days_to_best} days</strong><br>
                        Potential gain: ${best_price - current_price:+.2f}
                    </small>
                </div>
                """, unsafe_allow_html=True)
            
            # Milestone Analysis Summary
            st.markdown("### 📊 Milestone Analysis")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                # Trajectory Analysis
                trajectory_trend = "📈 Upward" if one_month_change > next_week_change else "📉 Downward" if one_month_change < next_week_change else "➡️ Stable"
                trajectory_color = "#4CAF50" if "Upward" in trajectory_trend else "#F44336" if "Downward" in trajectory_trend else "#FF9800"
                
                st.markdown(f"""
                <div style="
                    border: 2px solid {trajectory_color};
                    border-radius: 15px;
                    padding: 20px;
                    background: linear-gradient(135deg, {trajectory_color}15, {trajectory_color}08);
                ">
                    <h4 style="margin: 0; color: {trajectory_color};">📈 Price Trajectory</h4>
                    <h3 style="margin: 10px 0; color: {trajectory_color};">{trajectory_trend}</h3>
                    <p style="margin: 0; font-size: 14px;">
                        Short-term: {next_week_change:+.1f}%<br>
                        Long-term: {one_month_change:+.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with analysis_col2:
                # Investment Window Analysis
                if best_price_change >= 10:
                    window_status = "🟢 Excellent Opportunity"
                    window_color = "#4CAF50"
                    window_desc = "Strong upside potential"
                elif best_price_change >= 5:
                    window_status = "🔵 Good Opportunity"
                    window_color = "#2196F3"
                    window_desc = "Solid growth potential"
                elif best_price_change >= 0:
                    window_status = "🟡 Modest Opportunity"
                    window_color = "#FF9800"
                    window_desc = "Limited upside potential"
                else:
                    window_status = "🔴 Challenging Period"
                    window_color = "#F44336"
                    window_desc = "Consider alternative timing"
                
                st.markdown(f"""
                <div style="
                    border: 2px solid {window_color};
                    border-radius: 15px;
                    padding: 20px;
                    background: linear-gradient(135deg, {window_color}15, {window_color}08);
                ">
                    <h4 style="margin: 0; color: {window_color};">🎯 Investment Window</h4>
                    <h3 style="margin: 10px 0; color: {window_color};">{window_status}</h3>
                    <p style="margin: 0; font-size: 14px;">
                        {window_desc}<br>
                        Peak potential: {best_price_change:+.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Strategic Milestone Recommendations
            st.markdown("### 🎯 Strategic Milestone Recommendations")
            
            # Create recommendation based on milestone analysis
            if next_week_change >= 3 and one_month_change >= 5:
                strategy = "🟢 Aggressive Growth Strategy"
                strategy_color = "#4CAF50"
                strategy_desc = "Both short and long-term targets are positive. Consider increasing position size."
                actions = [
                    f"✅ **Week 1**: Target ${next_week_price:.2f} ({next_week_change:+.1f}%)",
                    f"✅ **Month 1**: Target ${one_month_price:.2f} ({one_month_change:+.1f}%)",
                    f"🏆 **Peak Target**: ${best_price:.2f} on {best_price_date.strftime('%b %d')}"
                ]
            elif next_week_change >= 0 and one_month_change >= 0:
                strategy = "🔵 Conservative Growth Strategy"
                strategy_color = "#2196F3"
                strategy_desc = "Positive trajectory with moderate gains. Suitable for steady growth."
                actions = [
                    f"📈 **Week 1**: Modest gain to ${next_week_price:.2f}",
                    f"📈 **Month 1**: Steady growth to ${one_month_price:.2f}",
                    f"🎯 **Best Case**: ${best_price:.2f} represents {best_price_change:+.1f}% upside"
                ]
            else:
                strategy = "🟡 Defensive Strategy"
                strategy_color = "#FF9800"
                strategy_desc = "Mixed signals suggest caution. Focus on risk management."
                actions = [
                    f"⚠️ **Week 1**: Monitor closely around ${next_week_price:.2f}",
                    f"⚠️ **Month 1**: Reassess position near ${one_month_price:.2f}",
                    f"🛡️ **Risk Management**: Set stops below current levels"
                ]
            
            st.markdown(f"""
            <div style="
                border: 3px solid {strategy_color};
                border-radius: 20px;
                padding: 25px;
                background: linear-gradient(135deg, {strategy_color}20, {strategy_color}10);
                margin: 20px 0;
            ">
                <h3 style="margin: 0; color: {strategy_color}; text-align: center;">{strategy}</h3>
                <p style="margin: 15px 0; text-align: center; font-size: 16px;">{strategy_desc}</p>
                <div style="margin: 20px 0;">
            """, unsafe_allow_html=True)
            
            for action in actions:
                st.markdown(f"- {action}")
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Milestone guide for beginners
            if beginner_mode:
                with st.expander("🎯 Understanding Your Milestones"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **📅 Next Week Target:**
                        - Your short-term (7-day) price goal
                        - Good for quick gains or losses
                        - Use for immediate trading decisions
                        
                        **📈 One Month Target:**
                        - Your medium-term (30-day) price goal
                        - Shows overall trend direction
                        - Better for strategic planning
                        """)
                    with col2:
                        st.markdown("""
                        **🏆 Best Predicted Price:**
                        - The highest price AI expects in 30 days
                        - Your optimal sell target
                        - Shows maximum potential gain
                        
                        **💡 How to Use:**
                        - Compare all three milestones
                        - Look for consistent upward trend
                        - Plan your buy/sell strategy accordingly
                        """)
                    
                    st.info("🎯 **Smart Strategy**: If all three milestones are higher than today's price, it's generally a good sign for buying!")
            
            # Summary for beginners
            if beginner_mode:
                future_price = predictions['Predicted_Price'].iloc[-1]
                expected_return = ((future_price - current_price) / current_price) * 100
                
                st.markdown("### 📝 **Simple Summary:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if expected_return > 5:
                        recommendation = "🟢 **LOOKS GOOD**"
                        reason = "AI predicts good returns"
                    elif expected_return > 0:
                        recommendation = "🟡 **OKAY**" 
                        reason = "Small gains expected"
                    else:
                        recommendation = "🔴 **BE CAREFUL**"
                        reason = "AI predicts losses"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎓 Simple Recommendation</h4>
                        <h3>{recommendation}</h3>
                        <p>{reason}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    risk_level = "Low" if risk_metrics['volatility'] < 30 else "Medium" if risk_metrics['volatility'] < 50 else "High"
                    risk_color = "🟢" if risk_metrics['volatility'] < 30 else "🟡" if risk_metrics['volatility'] < 50 else "🔴"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>⚠️ Risk Level</h4>
                        <h3>{risk_color} {risk_level} Risk</h3>
                        <p>{risk_metrics['volatility']:.1f}% volatility</p>
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
        st.info("👆 **Click 'Start AI Analysis' in the sidebar to begin!**")
    
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
