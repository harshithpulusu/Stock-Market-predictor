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
period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "5y"], index=0)

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
        
        # Prepare data for prediction
        data['Price_Next'] = data['Close'].shift(-1)
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_30'] = data['Close'].rolling(window=30).mean()
        data['Volatility'] = data['Close'].rolling(window=10).std()
        data['Price_Change'] = data['Close'].pct_change()
        
        # Create features
        features = ['Close', 'Volume', 'MA_10', 'MA_30', 'Volatility', 'Price_Change']
        data_clean = data[features + ['Price_Next']].dropna()
        
        X = data_clean[features]
        y = data_clean['Price_Next']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions for selected number of days
        last_data = X.iloc[-1:].values
        predictions = []
        dates = []
        
        for i in range(prediction_days):
            pred = model.predict(last_data)[0]
            predictions.append(pred)
            dates.append(datetime.now() + timedelta(days=i+1))
            
            # Update features for next prediction (simplified)
            new_row = last_data[0].copy()
            new_row[0] = pred  # Close price
            new_row[2] = np.mean([new_row[0]] + [predictions[j] for j in range(max(0, len(predictions)-10), len(predictions))])  # MA_10
            new_row[3] = np.mean([new_row[0]] + [predictions[j] for j in range(max(0, len(predictions)-30), len(predictions))])  # MA_30
            last_data = new_row.reshape(1, -1)
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Price': predictions,
            'Change_from_Current': [(p - current_price) / current_price * 100 for p in predictions]
        })
        
        # Calculate model accuracy
        test_predictions = model.predict(X_test)
        accuracy = 100 - (mean_absolute_error(y_test, test_predictions) / np.mean(y_test) * 100)
        
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
        
        # Confidence bands (±5%)
        upper_band = pred_df['Predicted_Price'] * 1.05
        lower_band = pred_df['Predicted_Price'] * 0.95
        
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
            name='Confidence Band (±5%)',
            hovertemplate='<b>Confidence Range:</b> ±5%<extra></extra>'
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
