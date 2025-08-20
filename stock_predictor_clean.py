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
</style>
""", unsafe_allow_html=True)

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
        
        # Simple chart display (placeholder for full chart functionality)
        if predictions is not None:
            st.subheader("📈 Price Chart & Predictions")
            if beginner_mode:
                st.info("🕯️ **Chart explanation**: Green = price went up, Red = price went down, Purple = AI predictions")
            
            # Create a simple line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index[-30:], 
                y=data['Close'].tail(30),
                mode='lines',
                name='Historical Price',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions['Date'][:15], 
                y=predictions['Predicted_Price'][:15],
                mode='lines+markers',
                name='AI Predictions',
                line=dict(color='purple', dash='dash')
            ))
            
            fig.update_layout(
                title="Stock Price & AI Predictions",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
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
