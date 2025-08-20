import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="📈 Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📈 AI Stock Market Predictor")
st.markdown("### Predict stock prices with advanced machine learning and comprehensive analytics")

# Sidebar for stock selection
st.sidebar.header("🔧 Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL", help="Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)")
period_options = {
    "6 Months": "6mo", 
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}
period_display = st.sidebar.selectbox(
    "📊 Historical Data Period", 
    list(period_options.keys()), 
    index=1, 
    help="Select the historical data timeframe for training the AI model"
)
period = period_options[period_display]

# Prediction days slider
prediction_days = st.sidebar.slider(
    "📅 Prediction Days", 
    min_value=7, 
    max_value=90, 
    value=30, 
    step=1,
    help="Select how many days ahead to predict (7-90 days)"
)

# Beginner mode toggle
beginner_mode = st.sidebar.checkbox("🎓 Beginner Mode", value=True, help="Show detailed explanations")

if st.sidebar.button("🚀 Analyze Stock"):
    try:
        # Fetch stock data
        with st.spinner(f"Fetching data for {symbol.upper()}..."):
            stock = yf.Ticker(symbol.upper())
            data = stock.history(period=period)
            
            if data.empty:
                st.error("❌ Could not fetch data. Please check the stock symbol.")
                st.stop()
        
        # Basic info
        info = stock.info
        company_name = info.get('longName', symbol.upper())
        current_price = data['Close'].iloc[-1]
        
        st.success(f"✅ Successfully loaded data for **{company_name}** ({symbol.upper()})")
        
        # Enhanced feature engineering for better accuracy
        data['Price_Next'] = data['Close'].shift(-1)
        
        # Technical indicators
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_30'] = data['Close'].rolling(window=30).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD (Moving Average Convergence Divergence)
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI (Relative Strength Index) with safe division
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Avoid division by zero
        rs = gain / (loss + 1e-10)  # Add small epsilon to prevent division by zero
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Cap RSI between 0 and 100
        data['RSI'] = np.clip(data['RSI'], 0, 100)
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        # Safe BB Position calculation
        bb_range = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (bb_range + 1e-10)
        data['BB_Position'] = np.clip(data['BB_Position'], -2, 3)  # Cap extreme values
        
        # Volume indicators with safe division
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / (data['Volume_MA'] + 1e-10)
        data['Volume_Ratio'] = np.clip(data['Volume_Ratio'], 0, 10)  # Cap extreme ratios
        
        # Price momentum and volatility
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_2d'] = data['Close'].pct_change(periods=2)
        data['Price_Change_5d'] = data['Close'].pct_change(periods=5)
        data['Volatility_10'] = data['Close'].rolling(window=10).std()
        data['Volatility_20'] = data['Close'].rolling(window=20).std()
        
        # High-Low indicators with safe division
        data['High_Low_Ratio'] = data['High'] / (data['Low'] + 1e-10)
        data['High_Low_Ratio'] = np.clip(data['High_Low_Ratio'], 0.5, 5)  # Cap extreme ratios
        
        data['Close_High_Ratio'] = data['Close'] / (data['High'] + 1e-10)
        data['Close_High_Ratio'] = np.clip(data['Close_High_Ratio'], 0, 1)
        
        data['Close_Low_Ratio'] = data['Close'] / (data['Low'] + 1e-10)
        data['Close_Low_Ratio'] = np.clip(data['Close_Low_Ratio'], 0.5, 5)
        
        # Price position within daily range with safe division
        daily_range = data['High'] - data['Low']
        data['Price_Position'] = (data['Close'] - data['Low']) / (daily_range + 1e-10)
        data['Price_Position'] = np.clip(data['Price_Position'], 0, 1)
        
        # Lag features (previous days' prices)
        for lag in [1, 2, 3, 5, 10]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            data[f'Close_Max_{window}'] = data['Close'].rolling(window=window).max()
            data[f'Close_Min_{window}'] = data['Close'].rolling(window=window).min()
            data[f'Close_Range_{window}'] = data[f'Close_Max_{window}'] - data[f'Close_Min_{window}']
        
        # Advanced features
        data['Price_Acceleration'] = data['Price_Change'].diff()
        data['Volume_Price_Trend'] = data['Volume'] * data['Price_Change']
        
        # Create comprehensive feature list
        features = [
            'Close', 'Volume', 'High', 'Low', 'Open',
            'MA_5', 'MA_10', 'MA_20', 'MA_30', 'MA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'RSI', 'BB_Width', 'BB_Position',
            'Volume_Ratio', 'Price_Change', 'Price_Change_2d', 'Price_Change_5d',
            'Volatility_10', 'Volatility_20',
            'High_Low_Ratio', 'Close_High_Ratio', 'Close_Low_Ratio', 'Price_Position',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
            'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3',
            'Close_Range_5', 'Close_Range_10', 'Close_Range_20',
            'Price_Acceleration', 'Volume_Price_Trend'
        ]
        
        # Clean data and prepare for training with proper validation
        data_clean = data[features + ['Price_Next']].dropna()
        
        # Handle infinite and extreme values
        for col in features + ['Price_Next']:
            if col in data_clean.columns:
                # Replace infinite values with NaN
                data_clean[col] = data_clean[col].replace([np.inf, -np.inf], np.nan)
                
                # Cap extreme values (beyond 3 standard deviations)
                if data_clean[col].std() > 0:
                    mean_val = data_clean[col].mean()
                    std_val = data_clean[col].std()
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    data_clean[col] = np.clip(data_clean[col], lower_bound, upper_bound)
        
        # Remove any remaining NaN values
        data_clean = data_clean.dropna()
        
        # Ensure we have enough data for training
        if len(data_clean) < 50:
            st.error("❌ Insufficient clean data for reliable predictions. Try a longer historical period.")
            st.stop()
        
        X = data_clean[features]
        y = data_clean['Price_Next']
        
        # Additional validation for X and y
        if X.isnull().any().any() or y.isnull().any():
            st.error("❌ Data contains missing values after cleaning.")
            st.stop()
        
        if np.isinf(X.values).any() or np.isinf(y.values).any():
            st.error("❌ Data contains infinite values after cleaning.")
            st.stop()
        
        # Enhanced model training with ensemble methods
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Split data with time-series aware approach (no random shuffling)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Final validation before scaling
        for df_name, df in [("X_train", X_train), ("X_test", X_test), ("y_train", y_train), ("y_test", y_test)]:
            if hasattr(df, 'values'):
                values = df.values
            else:
                values = df
            
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                st.error(f"❌ {df_name} contains NaN or infinite values.")
                st.stop()
        
        # Feature scaling for better performance with error handling
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Validate scaled data
            if np.any(np.isnan(X_train_scaled)) or np.any(np.isinf(X_train_scaled)):
                st.error("❌ Scaled training data contains NaN or infinite values.")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error during feature scaling: {str(e)}")
            st.stop()
        
        # Create ensemble of models for better accuracy
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            ),
            'LinearRegression': LinearRegression()
        }
        
        # Train models and calculate weights based on performance
        model_weights = {}
        model_predictions = {}
        trained_models = {}
        
        for name, model in models.items():
            if name == 'LinearRegression':
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
            
            trained_models[name] = model
            model_predictions[name] = pred
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            # Weight based on R² score (higher R² = better model = higher weight)
            model_weights[name] = max(0.1, r2)  # Minimum weight of 0.1
        
        # Normalize weights
        total_weight = sum(model_weights.values())
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
        
        # Calculate ensemble accuracy
        ensemble_pred = np.zeros_like(y_test)
        for name, weight in model_weights.items():
            ensemble_pred += weight * model_predictions[name]
        
        accuracy = 100 - (mean_absolute_error(y_test, ensemble_pred) / np.mean(y_test) * 100)
        
        # Conservative and realistic prediction approach
        # Instead of recursive prediction, use trend-based forecasting with constraints
        
        # Get recent price trends and volatility
        recent_prices = data['Close'].tail(30).values
        recent_returns = data['Close'].pct_change().tail(30).dropna().values
        historical_volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
        
        # Calculate realistic daily return expectations
        avg_daily_return = np.mean(recent_returns)
        median_daily_return = np.median(recent_returns)
        
        # Conservative approach: use median return and cap predictions
        base_daily_return = median_daily_return
        
        # Predict only a few days using ML, then use trend extrapolation
        ml_prediction_window = min(7, prediction_days)  # Only use ML for first week
        
        predictions = []
        dates = []
        confidence_intervals = []
        
        # Get last known data point
        last_price = current_price
        last_data = X.iloc[-1:].values
        last_data_scaled = scaler.transform(last_data)
        
        for i in range(prediction_days):
            prediction_date = datetime.now() + timedelta(days=i+1)
            dates.append(prediction_date)
            
            if i < ml_prediction_window:
                # Use ML models for short-term predictions
                model_preds = []
                
                # RandomForest prediction
                rf_pred = trained_models['RandomForest'].predict(last_data)[0]
                model_preds.append(rf_pred * model_weights['RandomForest'])
                
                # GradientBoosting prediction
                gb_pred = trained_models['GradientBoosting'].predict(last_data)[0]
                model_preds.append(gb_pred * model_weights['GradientBoosting'])
                
                # LinearRegression prediction
                lr_pred = trained_models['LinearRegression'].predict(last_data_scaled)[0]
                model_preds.append(lr_pred * model_weights['LinearRegression'])
                
                # Ensemble prediction
                ml_prediction = sum(model_preds)
                
                # Apply conservative constraints to ML prediction
                max_daily_change = 0.05  # 5% max daily change
                min_price = last_price * (1 - max_daily_change)
                max_price = last_price * (1 + max_daily_change)
                
                predicted_price = np.clip(ml_prediction, min_price, max_price)
                
                # Calculate confidence for ML predictions
                individual_preds = [
                    trained_models['RandomForest'].predict(last_data)[0],
                    trained_models['GradientBoosting'].predict(last_data)[0],
                    trained_models['LinearRegression'].predict(last_data_scaled)[0]
                ]
                pred_std = np.std(individual_preds)
                confidence_intervals.append(min(pred_std, last_price * 0.02))  # Cap at 2% of price
                
            else:
                # Use trend-based prediction for longer-term forecasts
                days_from_ml = i - ml_prediction_window + 1
                
                # Apply conservative trend with decay
                trend_decay = np.exp(-days_from_ml / 30)  # Trend decays over time
                adjusted_return = base_daily_return * trend_decay
                
                # Add some randomness but keep it realistic
                volatility_factor = min(0.01, historical_volatility / 252)  # Daily volatility, capped at 1%
                random_factor = np.random.normal(0, volatility_factor)
                
                daily_return = adjusted_return + random_factor
                # Cap daily returns to realistic range
                daily_return = np.clip(daily_return, -0.05, 0.05)  # ±5% max daily change
                
                predicted_price = last_price * (1 + daily_return)
                
                # Confidence decreases with time
                confidence_intervals.append(last_price * min(0.05, 0.01 * days_from_ml))
            
            predictions.append(predicted_price)
            last_price = predicted_price
            
            # Update features minimally and conservatively for next prediction
            if i < ml_prediction_window - 1:  # Only update features for ML window
                new_row = last_data[0].copy()
                new_row[0] = predicted_price  # Close price
                
                # Very conservative feature updates
                if len(predictions) >= 5:
                    new_row[5] = np.mean(predictions[-5:])  # MA_5
                if len(predictions) >= 10:
                    new_row[6] = np.mean(predictions[-10:])  # MA_10
                
                # Keep other features relatively stable
                if len(predictions) > 1:
                    price_change = (predicted_price - predictions[-2]) / predictions[-2]
                    new_row[14] = np.clip(price_change, -0.05, 0.05)  # Price_Change, capped
                
                last_data = new_row.reshape(1, -1)
                last_data_scaled = scaler.transform(last_data)
        
        # Create realistic predictions DataFrame
        pred_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Price': predictions,
            'Change_from_Current': [(p - current_price) / current_price * 100 for p in predictions],
            'Confidence_Interval': confidence_intervals
        })
        
        # Apply final sanity checks
        # Cap total percentage changes to realistic ranges
        max_total_change = min(50, prediction_days * 2)  # Max 2% per day or 50% total
        pred_df['Change_from_Current'] = np.clip(
            pred_df['Change_from_Current'], 
            -max_total_change, 
            max_total_change
        )
        
        # Recalculate prices based on capped changes
        pred_df['Predicted_Price'] = current_price * (1 + pred_df['Change_from_Current'] / 100)
        
        # === STOCK OVERVIEW DASHBOARD ===
        st.markdown("---")
        st.markdown("## 📊 Stock Overview Dashboard")
        if beginner_mode:
            st.info("🎯 Key metrics about the stock including current price, volatility, AI confidence, and buy/sell recommendation.")
        
        # Calculate key metrics
        volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
        expected_return = pred_df['Change_from_Current'].iloc[-1]  # Last day prediction
        
        # AI Confidence calculation (based on prediction consistency and model accuracy)
        prediction_std = np.std(pred_df['Predicted_Price'])
        price_range = data['Close'].max() - data['Close'].min()
        consistency_score = max(0, 100 - (prediction_std / price_range * 100))
        ai_confidence = (accuracy * 0.6 + consistency_score * 0.4)  # Weighted average
        
        # Buy/Sell/Hold recommendation
        if expected_return >= 10 and volatility < 30:
            recommendation = "🟢 STRONG BUY"
            rec_color = "#00C851"
            rec_explanation = "High expected return with manageable risk"
        elif expected_return >= 5 and volatility < 40:
            recommendation = "🟡 BUY"
            rec_color = "#FFD700"
            rec_explanation = "Positive outlook with moderate risk"
        elif expected_return >= 0 and volatility < 50:
            recommendation = "🟠 HOLD"
            rec_color = "#FF8800"
            rec_explanation = "Neutral outlook, maintain current position"
        elif expected_return >= -5:
            recommendation = "🔴 SELL"
            rec_color = "#FF4444"
            rec_explanation = "Negative outlook, consider reducing position"
        else:
            recommendation = "🛑 STRONG SELL"
            rec_color = "#CC0000"
            rec_explanation = "High risk of significant losses"
        
        # Display dashboard in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            ">
                <h4>💰 Current Price</h4>
                <h2>${current_price:.2f}</h2>
                <p style="margin: 0; opacity: 0.9;">Last Close</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Volatility color coding
            if volatility < 20:
                vol_color = "#00C851"  # Green - Low volatility
                vol_status = "Low Risk"
            elif volatility < 40:
                vol_color = "#FFD700"  # Yellow - Medium volatility
                vol_status = "Medium Risk"
            else:
                vol_color = "#FF4444"  # Red - High volatility
                vol_status = "High Risk"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            ">
                <h4>📈 Volatility</h4>
                <h2 style="color: {vol_color};">{volatility:.1f}%</h2>
                <p style="margin: 0; opacity: 0.9;">{vol_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # AI Confidence color coding
            if ai_confidence >= 80:
                conf_color = "#00C851"  # Green - High confidence
                conf_status = "Very High"
            elif ai_confidence >= 70:
                conf_color = "#FFD700"  # Yellow - Good confidence
                conf_status = "High"
            elif ai_confidence >= 60:
                conf_color = "#FF8800"  # Orange - Medium confidence
                conf_status = "Medium"
            else:
                conf_color = "#FF4444"  # Red - Low confidence
                conf_status = "Low"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            ">
                <h4>🤖 AI Confidence</h4>
                <h2 style="color: {conf_color};">{ai_confidence:.0f}%</h2>
                <p style="margin: 0; opacity: 0.9;">{conf_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: #333;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            ">
                <h4>🎯 Recommendation</h4>
                <h2 style="color: {rec_color}; font-size: 1.5em;">{recommendation}</h2>
                <p style="margin: 0; opacity: 0.8; font-size: 0.9em;">{rec_explanation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional detailed metrics row
        st.markdown("### 📋 Detailed Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Price trend analysis
            recent_trend = data['Close'].tail(5).pct_change().mean() * 100
            trend_direction = "📈 Upward" if recent_trend > 0.5 else "📉 Downward" if recent_trend < -0.5 else "➡️ Sideways"
            
            st.metric(
                label="5-Day Trend",
                value=trend_direction,
                delta=f"{recent_trend:.2f}% avg daily change"
            )
        
        with col2:
            # Expected return for selected period
            st.metric(
                label=f"{prediction_days}-Day Expected Return",
                value=f"{expected_return:+.1f}%",
                delta=f"${(current_price * expected_return / 100):+.2f} per share"
            )
        
        with col3:
            # Risk-Return Ratio
            risk_return_ratio = abs(expected_return) / max(volatility, 1)  # Avoid division by zero
            ratio_status = "Excellent" if risk_return_ratio > 0.5 else "Good" if risk_return_ratio > 0.3 else "Fair" if risk_return_ratio > 0.1 else "Poor"
            
            st.metric(
                label="Risk-Return Ratio",
                value=f"{risk_return_ratio:.2f}",
                delta=ratio_status
            )
        
        # === AI MODEL PERFORMANCE DASHBOARD ===
        st.markdown("---")
        st.markdown("## 🤖 AI Model Performance Dashboard")
        if beginner_mode:
            st.info("🎯 Conservative prediction system: AI models predict first 7 days, then realistic trend extrapolation for longer periods. This prevents absurd long-term predictions.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Model Contributions")
            for model_name, weight in model_weights.items():
                st.metric(
                    label=f"{model_name}",
                    value=f"{weight*100:.1f}%",
                    help=f"Contribution of {model_name} to final predictions"
                )
        
        with col2:
            st.markdown("### 🎯 Ensemble Accuracy")
            st.metric(
                label="Overall Model Accuracy",
                value=f"{accuracy:.1f}%",
                delta="Ensemble of 3 AI models" if accuracy > 70 else "Consider longer historical data"
            )
            
            # Confidence score based on model agreement
            avg_confidence = np.mean(pred_df['Confidence_Interval'])
            confidence_score = max(0, 100 - (avg_confidence / current_price * 100))
            st.metric(
                label="Prediction Confidence",
                value=f"{confidence_score:.1f}%",
                help="How much the AI models agree on predictions"
            )
        
        with col3:
            st.markdown("### 📈 Feature Importance")
            # Show top contributing features (simplified)
            feature_importance_info = [
                "📊 Technical Indicators (RSI, MACD)",
                "📈 Moving Averages (5, 10, 20, 30, 50)",
                "💹 Price Momentum & Volatility",
                "📊 Volume & Price Relationships",
                "🔄 Historical Price Patterns"
            ]
            
            for i, feature in enumerate(feature_importance_info[:3]):
                st.write(f"**{i+1}.** {feature}")
        
        # === 1. ENHANCED MAIN PRICE CHART ===
        st.markdown("---")
        st.markdown("## 📈 Enhanced Main Price Chart")
        if beginner_mode:
            st.info("📊 This chart shows historical prices (blue) with AI predictions (purple), confidence bands, and moving averages to help you understand price trends.")
        
        fig_main = go.Figure()
        
        # Historical data (last 60 days)
        hist_data = data.tail(60)
        fig_main.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#2E86AB', width=3),
            hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # 20-day and 50-day moving averages
        fig_main.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'].rolling(20).mean(),
            mode='lines',
            name='20-Day MA',
            line=dict(color='#A23B72', width=2, dash='dot'),
            hovertemplate='<b>20-Day MA:</b> $%{y:.2f}<extra></extra>'
        ))
        
        fig_main.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'].rolling(50).mean(),
            mode='lines',
            name='50-Day MA',
            line=dict(color='#F18F01', width=2, dash='dash'),
            hovertemplate='<b>50-Day MA:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # AI Predictions
        fig_main.add_trace(go.Scatter(
            x=pred_df['Date'],
            y=pred_df['Predicted_Price'],
            mode='lines+markers',
            name='AI Predictions',
            line=dict(color='#8A2BE2', width=3),
            marker=dict(size=4, color='#8A2BE2'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Predicted:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Enhanced confidence bands based on model agreement
        upper_band = pred_df['Predicted_Price'] + pred_df['Confidence_Interval']
        lower_band = pred_df['Predicted_Price'] - pred_df['Confidence_Interval']
        
        fig_main.add_trace(go.Scatter(
            x=pred_df['Date'],
            y=upper_band,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_main.add_trace(go.Scatter(
            x=pred_df['Date'],
            y=lower_band,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(138, 43, 226, 0.2)',
            name='AI Confidence Band',
            hovertemplate='<b>Model Agreement Range</b><extra></extra>'
        ))
        
        # Weekly milestone markers (golden stars)
        for week in range(0, 4):  # 4 weeks
            if week * 7 < len(pred_df):
                fig_main.add_trace(go.Scatter(
                    x=[pred_df['Date'].iloc[week * 7]],
                    y=[pred_df['Predicted_Price'].iloc[week * 7]],
                    mode='markers',
                    marker=dict(size=15, color='gold', symbol='star'),
                    name=f'Week {week + 1}' if week == 0 else '',
                    showlegend=True if week == 0 else False,
                    hovertemplate=f'<b>Week {week + 1} Target</b><br><b>Date:</b> %{{x}}<br><b>Price:</b> $%{{y:.2f}}<extra></extra>'
                ))
        
        # Current price reference line
        fig_main.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="#C73E1D",
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="top left"
        )
        
        # Color-coded prediction zones
        price_range = max(hist_data['Close'].max(), pred_df['Predicted_Price'].max()) - min(hist_data['Close'].min(), pred_df['Predicted_Price'].min())
        min_price = min(hist_data['Close'].min(), pred_df['Predicted_Price'].min())
        
        # Bullish zone (top third) - Green
        fig_main.add_hrect(
            y0=min_price + (2/3) * price_range,
            y1=min_price + price_range,
            fillcolor="rgba(0, 255, 0, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="🐂 Bullish Zone",
            annotation_position="top left"
        )
        
        # Bearish zone (bottom third) - Red
        fig_main.add_hrect(
            y0=min_price,
            y1=min_price + (1/3) * price_range,
            fillcolor="rgba(255, 0, 0, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="🐻 Bearish Zone",
            annotation_position="bottom left"
        )
        
        fig_main.update_layout(
            title=f"📈 {company_name} - AI Price Prediction Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            hovermode='x unified',
            legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)")
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
        
        # === 2. PREDICTION TIMELINE CHART ===
        st.markdown("---")
        st.markdown("## 🎯 Prediction Timeline Chart")
        if beginner_mode:
            st.info(f"📅 This timeline shows percentage changes from today's price over the next {prediction_days} days. Green = good times to buy, Red = consider selling.")
        
        fig_timeline = go.Figure()
        
        # Color-code based on percentage change
        colors = ['#00C851' if change >= 5 else '#39C0ED' if change >= 0 else '#FF8800' if change >= -5 else '#FF4444' 
                 for change in pred_df['Change_from_Current']]
        
        fig_timeline.add_trace(go.Scatter(
            x=pred_df['Date'],
            y=pred_df['Change_from_Current'],
            mode='lines+markers',
            name='Predicted % Change',
            line=dict(color='#8A2BE2', width=2),
            marker=dict(size=6, color=colors),
            hovertemplate='<b>Date:</b> %{x}<br><b>Change:</b> %{y:.1f}%<br><b>Price:</b> $%{customdata:.2f}<extra></extra>',
            customdata=pred_df['Predicted_Price']
        ))
        
        # Break-even line at 0%
        fig_timeline.add_hline(
            y=0,
            line_dash="dash",
            line_color="#666666",
            annotation_text="Break-even",
            annotation_position="right"
        )
        
        # Buy zone (above 5%)
        fig_timeline.add_hrect(
            y0=5, y1=max(pred_df['Change_from_Current'].max(), 10),
            fillcolor="rgba(0, 200, 81, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="🟢 Strong Buy Zone",
            annotation_position="top right"
        )
        
        # Sell zone (below -5%)
        fig_timeline.add_hrect(
            y0=min(pred_df['Change_from_Current'].min(), -10), y1=-5,
            fillcolor="rgba(255, 68, 68, 0.1)",
            layer="below",
            line_width=0,
            annotation_text="🔴 Sell Zone",
            annotation_position="bottom right"
        )
        
        fig_timeline.update_layout(
            title=f"📊 {prediction_days}-Day Price Change Timeline",
            xaxis_title="Date",
            yaxis_title="% Change from Current Price",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # === 3. WEEKLY TARGETS BAR CHART ===
        st.markdown("---")
        st.markdown("## 📅 Weekly Targets Bar Chart")
        if beginner_mode:
            st.info("📊 Visual bar chart showing weekly price targets. Green bars = expected gains, Red bars = expected losses.")
        
        # Create weekly data
        weekly_data = []
        for week in range(4):
            week_idx = min(week * 7, len(pred_df) - 1)
            price = pred_df['Predicted_Price'].iloc[week_idx]
            change = pred_df['Change_from_Current'].iloc[week_idx]
            date = pred_df['Date'].iloc[week_idx]
            
            weekly_data.append({
                'Week': f'Week {week + 1}',
                'Target_Price': price,
                'Change_Percent': change,
                'Date': date.strftime('%b %d'),
                'Price_Change': price - current_price
            })
        
        weekly_df = pd.DataFrame(weekly_data)
        
        fig_weekly = go.Figure()
        
        # Color bars based on change
        bar_colors = ['#00C851' if change >= 0 else '#FF4444' for change in weekly_df['Change_Percent']]
        
        fig_weekly.add_trace(go.Bar(
            x=weekly_df['Week'],
            y=weekly_df['Target_Price'],
            marker_color=bar_colors,
            name='Weekly Targets',
            hovertemplate='<b>%{x}</b><br><b>Target:</b> $%{y:.2f}<br><b>Change:</b> %{customdata:.1f}%<br><b>Date:</b> %{text}<extra></extra>',
            customdata=weekly_df['Change_Percent'],
            text=weekly_df['Date']
        ))
        
        # Current price reference line
        fig_weekly.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="#C73E1D",
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="right"
        )
        
        fig_weekly.update_layout(
            title="📈 Weekly Price Targets",
            xaxis_title="Week",
            yaxis_title="Target Price ($)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # === 4. RISK & RETURN GAUGE DASHBOARD ===
        st.markdown("---")
        st.markdown("## 📊 Risk & Return Gauge Dashboard")
        if beginner_mode:
            st.info(f"🎮 Interactive gauges showing expected {prediction_days}-day return and risk score. Green = low risk/good return, Red = high risk/poor return.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Expected return gauge for selected period
            expected_return_gauge = pred_df['Change_from_Current'].iloc[-1]  # Last day prediction
            
            fig_return = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=expected_return_gauge,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Expected {prediction_days}-Day Return (%)"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [None, 20]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 10], 'color': "yellow"},
                        {'range': [10, 20], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 15
                    }
                }
            ))
            
            fig_return.update_layout(height=300)
            st.plotly_chart(fig_return, use_container_width=True)
        
        with col2:
            # Risk score gauge (based on volatility)
            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
            risk_score = min(100, volatility * 3)  # Scale to 0-100
            
            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score (0-100)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig_risk.update_layout(height=300)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        # === 5. DETAILED PREDICTION TIMELINE TABLE ===
        st.markdown("---")
        st.markdown("## 📅 Detailed Prediction Timeline Table")
        if beginner_mode:
            st.info("📋 Comprehensive weekly breakdown with exact dates, prices, changes, and AI-generated buy/sell signals.")
        
        # Create detailed weekly data
        detailed_data = []
        for week in range(4):
            week_idx = min(week * 7, len(pred_df) - 1)
            price = pred_df['Predicted_Price'].iloc[week_idx]
            change_pct = pred_df['Change_from_Current'].iloc[week_idx]
            change_dollar = price - current_price
            date = pred_df['Date'].iloc[week_idx]
            
            # Generate AI signal
            if change_pct >= 10:
                signal = "🟢 Strong Buy"
                row_color = "#d4edda"
            elif change_pct >= 5:
                signal = "🟡 Buy"
                row_color = "#fff3cd"
            elif change_pct >= 0:
                signal = "🟠 Hold"
                row_color = "#ffeeba"
            else:
                signal = "🔴 Sell"
                row_color = "#f8d7da"
            
            days_from_now = week * 7 + 1
            
            detailed_data.append({
                'Week': f'Week {week + 1}',
                'Date': date.strftime('%a, %b %d, %Y'),
                'Target Price': f"${price:.2f}",
                'Price Change': f"${change_dollar:+.2f}",
                'Percentage Change': f"{change_pct:+.1f}%",
                'AI Signal': signal,
                'Days from Now': f"{days_from_now} days"
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        
        # Display as styled table
        st.dataframe(
            detailed_df,
            use_container_width=True,
            hide_index=True
        )
        
        # === 6. KEY PREDICTION MILESTONES ===
        st.markdown("---")
        st.markdown("## 🎯 Key Prediction Milestones")
        if beginner_mode:
            st.info("📊 Three key milestone cards showing next week, one month, and best predicted price targets.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Next Week Target
            next_week_price = pred_df['Predicted_Price'].iloc[6]  # 7 days
            next_week_change = pred_df['Change_from_Current'].iloc[6]
            next_week_date = pred_df['Date'].iloc[6]
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin: 10px 0;
            ">
                <h3>📅 Next Week Target</h3>
                <h2>${next_week_price:.2f}</h2>
                <p><strong>{next_week_date.strftime('%b %d, %Y')}</strong></p>
                <p>Change: {next_week_change:+.1f}%</p>
                <p>{next_week_price - current_price:+.2f} from current</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # End Period Target
            end_period_idx = min(29, len(pred_df) - 1)  # Use 30 days or end of predictions
            end_period_price = pred_df['Predicted_Price'].iloc[end_period_idx]
            end_period_change = pred_df['Change_from_Current'].iloc[end_period_idx]
            end_period_date = pred_df['Date'].iloc[end_period_idx]
            
            period_label = "One Month" if prediction_days >= 30 else f"{prediction_days}-Day"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin: 10px 0;
            ">
                <h3>📈 {period_label} Target</h3>
                <h2>${end_period_price:.2f}</h2>
                <p><strong>{end_period_date.strftime('%b %d, %Y')}</strong></p>
                <p>Change: {end_period_change:+.1f}%</p>
                <p>{end_period_price - current_price:+.2f} from current</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Best Predicted Price
            best_price_idx = pred_df['Predicted_Price'].idxmax()
            best_price = pred_df['Predicted_Price'].iloc[best_price_idx]
            best_price_change = pred_df['Change_from_Current'].iloc[best_price_idx]
            best_price_date = pred_df['Date'].iloc[best_price_idx]
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: white;
                margin: 10px 0;
            ">
                <h3>🎯 Best Predicted Price</h3>
                <h2>${best_price:.2f}</h2>
                <p><strong>{best_price_date.strftime('%b %d, %Y')}</strong></p>
                <p>Change: {best_price_change:+.1f}%</p>
                <p>{best_price - current_price:+.2f} from current</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("---")
        st.markdown("## 📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Model Accuracy", f"{accuracy:.1f}%")
        
        with col3:
            avg_change = pred_df['Change_from_Current'].mean()
            st.metric("Avg Period Change", f"{avg_change:+.1f}%")
        
        with col4:
            volatility_display = f"{volatility:.1f}%"
            st.metric("Volatility (Annual)", volatility_display)
        
        # Disclaimer
        st.markdown("---")
        st.markdown("""
        ### ⚠️ Important Disclaimer
        
        This prediction is based on historical data and machine learning models. **Stock market investments carry risk**, and past performance does not guarantee future results. 
        
        **Always:**
        - Do your own research
        - Consult with financial advisors
        - Never invest more than you can afford to lose
        - Consider your risk tolerance and investment goals
        
        This tool is for educational and informational purposes only.
        """)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.error("Please check the stock symbol and try again.")

else:
    st.info("👆 Please select a stock symbol and click 'Analyze Stock' to begin the prediction.")
    
    # Show sample screenshots or demo info
    st.markdown("""
    ## 🌟 Features Include:
    
    ### 📈 **Enhanced Main Price Chart**
    - Confidence bands around predictions
    - Weekly milestone markers (golden stars)
    - Multiple moving averages (20-day and 50-day)
    - Current price reference line
    - Advanced hover tooltips
    - Color-coded prediction zones
    
    ### 🎯 **Prediction Timeline Chart**
    - Exact dates when predictions will occur
    - Percentage change from current price
    - Green/Red zones for buy/sell recommendations
    - Visual break-even line at 0%
    
    ### 📅 **Weekly Targets Bar Chart**
    - Visual bar chart of weekly price targets
    - Color-coded bars (green for gains, red for losses)
    - Current price reference line
    
    ### 📊 **Risk & Return Gauge Dashboard**
    - Interactive gauge meters for expected return and risk
    - Color-coded zones (green=low risk, red=high risk)
    
    ### 📅 **Detailed Prediction Timeline Table**
    - Comprehensive weekly breakdown
    - Exact dates, target prices, price changes
    - AI-generated signals (Strong Buy, Buy, Hold, Sell)
    
    ### 🎯 **Key Prediction Milestones**
    - Next Week Target
    - One Month Target  
    - Best Predicted Price
    """)
