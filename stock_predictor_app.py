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

warnings.filterwarnings('ignore')

# Page configuration
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
</style>
""", unsafe_allow_html=True)

class StockPredictorApp:
    """Main application class for stock prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.data = None
        self.predictions = None
        self.stock_info = None
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_data(_self, symbol, period="2y"):
        """Fetch stock data with caching"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            stock_info = stock.info
            
            if len(data) < 50:
                raise ValueError("Insufficient data")
                
            return data, stock_info, True
            
        except Exception as e:
            return None, None, False
    
    def create_technical_features(self, data):
        """Create technical indicators"""
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Price_MA_5'] = df['Close'].rolling(5).mean()
        df['Price_MA_20'] = df['Close'].rolling(20).mean()
        df['Price_MA_50'] = df['Close'].rolling(50).mean()
        
        # Moving average ratios
        df['MA_Ratio_5_20'] = df['Price_MA_5'] / df['Price_MA_20']
        df['MA_Ratio_20_50'] = df['Price_MA_20'] / df['Price_MA_50']
        df['Price_to_MA20'] = df['Close'] / df['Price_MA_20']
        
        # Volatility features
        df['Volatility_5'] = df['Returns'].rolling(5).std()
        df['Volatility_20'] = df['Returns'].rolling(20).std()
        
        # Volume features
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Normalized'] = (df['RSI'] - 50) / 50
        
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
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
            df[f'Volume_Ratio_Lag_{lag}'] = df['Volume_Ratio'].shift(lag)
        
        # Target variable
        df['Target'] = df['Returns'].shift(-1)
        
        # Feature selection
        self.feature_columns = [
            'MA_Ratio_5_20', 'MA_Ratio_20_50', 'Price_to_MA20',
            'Volatility_5', 'Volatility_20', 'Volume_Ratio',
            'RSI_Normalized', 'MACD', 'MACD_Histogram', 'BB_Position',
            'Returns_Lag_1', 'Returns_Lag_2', 'Returns_Lag_3', 'Returns_Lag_5',
            'Volume_Ratio_Lag_1', 'Volume_Ratio_Lag_2', 'Volume_Ratio_Lag_3'
        ]
        
        return df
    
    def train_model(self, data):
        """Train machine learning models"""
        # Prepare data
        feature_data = data[self.feature_columns + ['Target']].dropna()
        X = feature_data[self.feature_columns]
        y = feature_data['Target']
        
        # Models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
            'Linear Regression': LinearRegression()
        }
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = -np.inf
        best_model = None
        best_name = ""
        
        model_results = {}
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
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

def create_interactive_charts(data, predictions, symbol):
    """Create interactive Plotly charts"""
    
    # Main price chart with predictions
    fig_main = go.Figure()
    
    # Historical prices
    recent_data = data.tail(100)
    fig_main.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue', width=2)
    ))
    
    # Moving averages
    fig_main.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['Price_MA_20'],
        mode='lines',
        name='20-day MA',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    # Predictions
    if predictions is not None:
        fig_main.add_trace(go.Scatter(
            x=predictions['Date'],
            y=predictions['Predicted_Price'],
            mode='lines',
            name='AI Predictions',
            line=dict(color='red', width=2, dash='dot')
        ))
    
    fig_main.update_layout(
        title=f'{symbol} - Price Analysis & AI Predictions',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        height=500
    )
    
    return fig_main

def create_technical_indicators_chart(data):
    """Create technical indicators chart"""
    recent_data = data.tail(100)
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxis=True,
        subplot_titles=('RSI', 'MACD', 'Bollinger Bands', 'Volume'),
        vertical_spacing=0.05,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    # RSI
    fig.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['RSI'],
        name='RSI', line=dict(color='purple')
    ), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['MACD'],
        name='MACD', line=dict(color='blue')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['MACD_Signal'],
        name='Signal', line=dict(color='red')
    ), row=2, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['Close'],
        name='Price', line=dict(color='black')
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['BB_Upper'],
        name='Upper Band', line=dict(color='gray', dash='dash')
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=recent_data.index, y=recent_data['BB_Lower'],
        name='Lower Band', line=dict(color='gray', dash='dash'),
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)'
    ), row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=recent_data.index, y=recent_data['Volume'],
        name='Volume', marker_color='lightblue'
    ), row=4, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def main():
    # App header
    st.markdown('<h1 class="main-header">🚀 AI Stock Market Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Analysis Settings")
    
    # Stock symbol input
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter a valid stock symbol (e.g., AAPL, TSLA, MSFT)"
    ).upper().strip()
    
    # Time period selection
    period_options = {
        "6 months": "6mo",
        "1 year": "1y",
        "2 years": "2y",
        "5 years": "5y"
    }
    period_display = st.sidebar.selectbox("Data Period", list(period_options.keys()), index=2)
    period = period_options[period_display]
    
    # Prediction days
    prediction_days = st.sidebar.slider("Prediction Days", 7, 60, 30)
    
    # Analysis button
    analyze_button = st.sidebar.button("🔮 Analyze Stock", type="primary")
    
    # Initialize predictor
    predictor = StockPredictorApp()
    
    if analyze_button and symbol:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch data
        status_text.text("📥 Fetching stock data...")
        progress_bar.progress(20)
        
        data, stock_info, success = predictor.fetch_data(symbol, period)
        
        if not success:
            st.error("❌ Failed to fetch data. Please check the stock symbol.")
            return
        
        predictor.data = data
        predictor.stock_info = stock_info
        
        # Create features
        status_text.text("🔬 Creating technical features...")
        progress_bar.progress(40)
        
        enhanced_data = predictor.create_technical_features(data)
        
        # Train model
        status_text.text("🤖 Training AI models...")
        progress_bar.progress(60)
        
        model_results, best_model_name, best_score = predictor.train_model(enhanced_data)
        
        # Make predictions
        status_text.text("🔮 Generating predictions...")
        progress_bar.progress(80)
        
        predictions = predictor.make_predictions(enhanced_data, prediction_days)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success("🎉 Analysis completed successfully!")
        
        # Company information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Price",
                f"${data['Close'].iloc[-1]:.2f}",
                f"{((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):+.2f}%"
            )
        
        with col2:
            company_name = stock_info.get('longName', 'N/A') if stock_info else 'N/A'
            st.metric("Company", company_name[:20] + "..." if len(company_name) > 20 else company_name)
        
        with col3:
            sector = stock_info.get('sector', 'N/A') if stock_info else 'N/A'
            st.metric("Sector", sector)
        
        # Prediction summary
        current_price = data['Close'].iloc[-1]
        predicted_price = predictions['Predicted_Price'].iloc[-1]
        expected_return = (predicted_price - current_price) / current_price * 100
        
        st.subheader("🎯 AI Prediction Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                f"{prediction_days}-Day Target",
                f"${predicted_price:.2f}",
                f"{expected_return:+.1f}%"
            )
        
        with col2:
            confidence = max(0, min(1, (best_score + 1) / 2))
            st.metric("AI Confidence", f"{confidence:.1%}")
        
        with col3:
            volatility = enhanced_data['Returns'].std() * np.sqrt(252) * 100
            st.metric("Volatility", f"{volatility:.1f}%")
        
        with col4:
            if expected_return > 5:
                signal = "🟢 BUY"
            elif expected_return < -5:
                signal = "🔴 SELL"
            else:
                signal = "🟡 HOLD"
            st.metric("Signal", signal)
        
        # Main chart
        st.subheader("📈 Price Analysis & Predictions")
        main_chart = create_interactive_charts(data, predictions, symbol)
        st.plotly_chart(main_chart, use_container_width=True)
        
        # Technical indicators
        st.subheader("📊 Technical Indicators")
        tech_chart = create_technical_indicators_chart(enhanced_data)
        st.plotly_chart(tech_chart, use_container_width=True)
        
        # Model performance
        st.subheader("🤖 AI Model Performance")
        
        model_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'R² Score': [results['mean_score'] for results in model_results.values()],
            'Std Dev': [results['std_score'] for results in model_results.values()]
        }).sort_values('R² Score', ascending=False)
        
        st.dataframe(model_df, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(predictor.model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': predictor.feature_columns,
                'Importance': predictor.model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Features'
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Weekly predictions table
        st.subheader("📅 Weekly Price Targets")
        
        weekly_predictions = predictions.iloc[::7].head(8)  # Every 7 days
        weekly_df = pd.DataFrame({
            'Week': [f"Week {i+1}" for i in range(len(weekly_predictions))],
            'Date': weekly_predictions['Date'].dt.strftime('%Y-%m-%d'),
            'Target Price': [f"${price:.2f}" for price in weekly_predictions['Predicted_Price']],
            'Expected Change': [f"{((price - current_price) / current_price * 100):+.1f}%" 
                              for price in weekly_predictions['Predicted_Price']]
        })
        
        st.dataframe(weekly_df, use_container_width=True)
        
        # Download predictions
        st.subheader("💾 Download Results")
        
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
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Important Disclaimer**: 
        This analysis is for educational purposes only. Past performance does not guarantee future results. 
        The AI model's predictions are based on historical patterns and may not reflect actual future prices. 
        Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
        """)
    
    else:
        # Welcome message
        st.info("""
        👋 **Welcome to the AI Stock Market Predictor!**
        
        This advanced application uses machine learning to analyze stocks and generate predictions based on:
        - 📊 Technical indicators (RSI, MACD, Bollinger Bands)
        - 🤖 Ensemble AI models (Random Forest, Gradient Boosting, Linear Regression)
        - 📈 Price patterns and volume analysis
        - 🔮 Time series forecasting
        
        **How to use:**
        1. Enter a stock symbol in the sidebar (e.g., AAPL, TSLA, MSFT)
        2. Choose your analysis period and prediction timeframe
        3. Click "Analyze Stock" to start the AI analysis
        4. View interactive charts, predictions, and download results
        
        **Features:**
        - Real-time stock data fetching
        - Interactive Plotly charts
        - AI model performance metrics
        - Weekly price targets
        - Downloadable predictions
        """)
        
        # Sample analysis showcase
        st.subheader("🎯 Sample Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Technical Analysis:**
            - Moving averages (5, 20, 50-day)
            - Relative Strength Index (RSI)
            - MACD indicators
            - Bollinger Bands
            - Volume analysis
            """)
        
        with col2:
            st.markdown("""
            **🤖 AI Predictions:**
            - Ensemble machine learning models
            - Time series cross-validation
            - Feature importance analysis
            - Confidence scoring
            - Risk assessment
            """)

if __name__ == "__main__":
    main()
