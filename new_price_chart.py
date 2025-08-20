            # Remove old beginner explanation
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
            
            # 2. AI PREDICTION LINE (Electric Purple)
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=pred_prices,
                mode='lines+markers',
                name='AI Predictions',
                line=dict(color='#FF00FF', width=3, shape='spline'),
                marker=dict(size=6, color='#FF00FF', symbol='diamond'),
                hovertemplate='<b>🔮 Date:</b> %{x}<br><b>🎯 Predicted Price:</b> $%{y:.2f}<extra></extra>',
                showlegend=True
            ))
            
            # 3. CONFIDENCE BANDS (Glowing Purple Area)
            upper_band = pred_prices * 1.05
            lower_band = pred_prices * 0.95
            
            # Upper confidence band
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=upper_band,
                mode='lines',
                line=dict(width=0),
                name='Confidence Band',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Lower confidence band with fill
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=lower_band,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.15)',
                name='Confidence Band',
                showlegend=True,
                hovertemplate='<b>🌫️ Confidence Range:</b> ±5%<extra></extra>'
            ))
            
            # 4. MOVING AVERAGES (Neon Colors)
            fig.add_trace(go.Scatter(
                x=data_extended.index,
                y=data_extended['MA20'],
                mode='lines',
                name='20-Day Moving Average',
                line=dict(color='#00FF00', width=2, dash='dot'),
                hovertemplate='<b>📊 20-Day MA:</b> $%{y:.2f}<extra></extra>',
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=data_extended.index,
                y=data_extended['MA50'],
                mode='lines',
                name='50-Day Moving Average',
                line=dict(color='#FFFF00', width=2, dash='dash'),
                hovertemplate='<b>📊 50-Day MA:</b> $%{y:.2f}<extra></extra>',
                showlegend=True
            ))
            
            # 5. CURRENT PRICE LINE (Bright Red)
            fig.add_hline(
                y=current_price,
                line_dash="solid",
                line_color="#FF0000",
                line_width=2,
                annotation_text=f"Current: ${current_price:.2f}",
                annotation_position="top left",
                annotation=dict(font=dict(color="#FF0000", size=12))
            )
            
            # 6. WEEKLY MILESTONE MARKERS (Golden Stars)
            for i in range(0, min(len(pred_dates), 28), 7):
                if i < len(pred_dates):
                    fig.add_trace(go.Scatter(
                        x=[pred_dates.iloc[i]],
                        y=[pred_prices.iloc[i]],
                        mode='markers',
                        marker=dict(size=15, color='gold', symbol='star', line=dict(color='orange', width=2)),
                        name='Weekly Milestones' if i == 0 else '',
                        showlegend=True if i == 0 else False,
                        hovertemplate=f'<b>⭐ Week {i//7 + 1} Target:</b><br><b>Date:</b> %{{x}}<br><b>Price:</b> $%{{y:.2f}}<extra></extra>'
                    ))
            
            # 7. SUPPORT/RESISTANCE ZONES (Subtle Background Colors)
            price_range = max(max(data_extended['Close']), max(pred_prices)) - min(min(data_extended['Close']), min(pred_prices))
            min_price = min(min(data_extended['Close']), min(pred_prices))
            max_price = max(max(data_extended['Close']), max(pred_prices))
            
            # Resistance zone (top third - green glow)
            fig.add_hrect(
                y0=min_price + (2/3) * price_range,
                y1=max_price,
                fillcolor="rgba(0, 255, 0, 0.08)",
                layer="below",
                line_width=0,
                annotation_text="🚀 Resistance Zone",
                annotation_position="top right"
            )
            
            # Support zone (bottom third - red glow)
            fig.add_hrect(
                y0=min_price,
                y1=min_price + (1/3) * price_range,
                fillcolor="rgba(255, 0, 0, 0.08)",
                layer="below",
                line_width=0,
                annotation_text="🛡️ Support Zone",
                annotation_position="bottom right"
            )
            
            # 8. MODERN BLACK THEME LAYOUT
            fig.update_layout(
                title={
                    'text': f'🌟 {symbol} - Advanced AI Price Analysis',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': '#FFFFFF', 'family': 'Arial Black'}
                },
                
                # BLACK BACKGROUND THEME
                plot_bgcolor='#000000',
                paper_bgcolor='#111111',
                
                # NEON GRID
                xaxis=dict(
                    title='📅 Time Period',
                    titlefont=dict(color='#FFFFFF', size=14),
                    tickfont=dict(color='#CCCCCC', size=12),
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    zerolinecolor='rgba(255, 255, 255, 0.2)',
                    showgrid=True
                ),
                yaxis=dict(
                    title='💰 Stock Price ($)',
                    titlefont=dict(color='#FFFFFF', size=14),
                    tickfont=dict(color='#CCCCCC', size=12),
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    zerolinecolor='rgba(255, 255, 255, 0.2)',
                    showgrid=True
                ),
                
                # MODERN LEGEND
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    bgcolor="rgba(0, 0, 0, 0.8)",
                    bordercolor="#FFFFFF",
                    borderwidth=1,
                    font=dict(color="#FFFFFF", size=12)
                ),
                
                # FIXED DIMENSIONS
                height=600,
                margin=dict(l=50, r=150, t=80, b=50),
                
                # PROFESSIONAL STYLING
                font=dict(family="Arial", size=12, color="#FFFFFF"),
                hovermode='x unified'
            )
            
            # Add confidence assessment
            confidence_score, confidence_level, confidence_text = assess_prediction_confidence(
                model_scores, risk_metrics['volatility']
            )
            
            # Display chart with black theme container
            st.markdown(f'''
            <div style="
                background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #000000 100%);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                margin: 20px 0;
                border: 2px solid #333;
            ">
            ''', unsafe_allow_html=True)
            
            st.plotly_chart(fig, use_container_width=False, height=600, config=get_fixed_price_chart_config())
            st.markdown('</div>', unsafe_allow_html=True)
            
            # COMPREHENSIVE CHART LEGEND & GUIDE
            st.markdown("## 🎯 Chart Legend & Analysis Guide")
            
            # Legend in organized columns
            leg_col1, leg_col2, leg_col3 = st.columns(3)
            
            with leg_col1:
                st.markdown("""
                ### 📈 **Price Data**
                
                🔵 **<span style="color: #00D4FF;">Historical Prices</span>**
                - Actual past stock performance
                - Shows real market movements
                
                🟣 **<span style="color: #FF00FF;">AI Predictions</span>**
                - Machine learning forecasts
                - Diamond markers for key points
                
                🌫️ **<span style="background: rgba(255, 0, 255, 0.2); padding: 2px 6px;">Confidence Band</span>**
                - ±5% uncertainty range around predictions
                - Wider = less certain, Narrower = more confident
                """, unsafe_allow_html=True)
            
            with leg_col2:
                st.markdown("""
                ### 📊 **Technical Indicators**
                
                🟢 **<span style="color: #00FF00;">20-Day Moving Average</span>**
                - Short-term trend (dotted green line)
                - Quick market sentiment indicator
                
                🟡 **<span style="color: #FFFF00;">50-Day Moving Average</span>**
                - Long-term trend (dashed yellow line)
                - Overall market direction
                
                🔴 **<span style="color: #FF0000;">Current Price Line</span>**
                - Today's market price reference
                - Baseline for gains/losses
                """, unsafe_allow_html=True)
            
            with leg_col3:
                st.markdown("""
                ### 🎯 **Key Markers & Zones**
                
                ⭐ **<span style="color: gold;">Weekly Milestones</span>**
                - Important weekly price targets
                - Strategic buy/sell timing points
                
                🚀 **<span style="background: rgba(0, 255, 0, 0.1); padding: 2px 6px;">Resistance Zone</span>**
                - Upper price levels (selling pressure)
                
                🛡️ **<span style="background: rgba(255, 0, 0, 0.1); padding: 2px 6px;">Support Zone</span>**
                - Lower price levels (buying opportunity)
                """, unsafe_allow_html=True)
            
            # Interactive Decision Guide
            st.markdown("### 🧠 Smart Trading Signals")
            
            # Current market analysis
            current_ma20 = data_extended['MA20'].iloc[-1] if not data_extended['MA20'].empty else current_price
            current_ma50 = data_extended['MA50'].iloc[-1] if not data_extended['MA50'].empty else current_price
            
            signal_col1, signal_col2, signal_col3 = st.columns(3)
            
            with signal_col1:
                # Trend Analysis
                trend_signal = "🟢 BULLISH" if current_price > current_ma20 > current_ma50 else "🔴 BEARISH" if current_price < current_ma20 < current_ma50 else "🟡 NEUTRAL"
                st.markdown(f"""
                **📈 Trend Signal:**
                
                {trend_signal}
                
                Current: ${current_price:.2f}
                20-Day MA: ${current_ma20:.2f}
                50-Day MA: ${current_ma50:.2f}
                """)
            
            with signal_col2:
                # Prediction Analysis
                next_week_pred = pred_prices.iloc[6] if len(pred_prices) > 6 else pred_prices.iloc[-1]
                pred_change = ((next_week_pred - current_price) / current_price) * 100
                pred_signal = "🚀 BUY" if pred_change > 5 else "💎 HOLD" if pred_change > -2 else "⚠️ CAUTION"
                
                st.markdown(f"""
                **🔮 AI Forecast:**
                
                {pred_signal}
                
                7-Day Target: ${next_week_pred:.2f}
                Expected Change: {pred_change:+.1f}%
                Confidence: {confidence_score:.0f}%
                """)
            
            with signal_col3:
                # Risk Assessment
                risk_level = "🟢 LOW" if confidence_score > 75 else "🟡 MEDIUM" if confidence_score > 50 else "🔴 HIGH"
                st.markdown(f"""
                **⚡ Risk Level:**
                
                {risk_level}
                
                Volatility: {risk_metrics['volatility']:.1f}%
                Model Accuracy: {np.mean(list(model_scores.values())):.1f}%
                Market Stability: {"High" if risk_metrics['volatility'] < 20 else "Medium" if risk_metrics['volatility'] < 40 else "Low"}
                """)
            
            # Quick Action Guide
            if beginner_mode:
                with st.expander("🎓 Beginner's Chart Reading Guide"):
                    st.markdown("""
                    ### 🎯 **How to Read This Chart Like a Pro:**
                    
                    **🔍 Step 1: Check the Trend**
                    - Look at the **blue line** (historical prices)
                    - Is it going up, down, or sideways?
                    - Compare with **green** (20-day) and **yellow** (50-day) moving averages
                    
                    **🔮 Step 2: Analyze AI Predictions**
                    - Follow the **purple line** (AI predictions)
                    - Purple area shows uncertainty - smaller = more confident
                    - **Diamond markers** highlight key prediction points
                    
                    **⭐ Step 3: Use Weekly Milestones**
                    - **Golden stars** mark important weekly targets
                    - Plan your trades around these dates
                    - Each star represents a week from now
                    
                    **🎨 Step 4: Understand the Zones**
                    - **Green background** = Resistance (hard to break above)
                    - **Red background** = Support (good buying opportunity)
                    - **Red line** = Current price (your reference point)
                    
                    **💡 Step 5: Make Smart Decisions**
                    - **BUY** when: Purple line trending up + price near support + high confidence
                    - **SELL** when: Purple line trending down + price near resistance + predictions peak
                    - **HOLD** when: Sideways trends + medium confidence + unclear signals
                    """)
                    
                    st.success("""
                    🏆 **Pro Tip:** Always combine multiple signals! 
                    Don't rely on just one indicator. Look at trends, predictions, and zones together.
                    """)
            
            st.markdown("---")
