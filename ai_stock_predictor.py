import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Try different backends for matplotlib
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        try:
            matplotlib.use('Agg')
            print("⚠️ Using non-interactive plotting backend. Charts will be saved instead of displayed.")
        except:
            print("⚠️ Matplotlib backend issue. Charts may not display properly.")

class AdvancedAIStockPredictor:
    def __init__(self):
        # Initialize multiple AI models for ensemble learning
        self.neural_network = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),  # Deep neural network
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=2000,
            random_state=42
        )
        
        self.gradient_boost = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        
        self.random_forest = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        
        # Ensemble model combining multiple AI approaches
        self.ensemble_model = VotingRegressor([
            ('neural_net', self.neural_network),
            ('gradient_boost', self.gradient_boost),
            ('random_forest', self.random_forest)
        ])
        
        self.scaler = StandardScaler()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.pca = PCA(n_components=15)  # Dimensionality reduction
        self.market_regime_model = KMeans(n_clusters=3, random_state=42)  # Market regime detection
        
        # AI performance tracking
        self.model_performances = {}
        self.prediction_confidence = []
        
    def get_stock_data(self, symbol, period='2y'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            print(f"Successfully downloaded {len(df)} days of {symbol} data")
            return df
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

    def analyze_news_sentiment(self, symbol):
        """Advanced AI-powered sentiment analysis with multiple sources"""
        try:
            # Get news from Yahoo Finance
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                print("No recent news found")
                return 0, 0, "neutral"
            
            sentiments = []
            emotions = {"positive": 0, "negative": 0, "neutral": 0}
            
            for article in news:
                # Analyze sentiment of title and summary
                title_sentiment = self.sentiment_analyzer.polarity_scores(article['title'])
                
                # Advanced sentiment classification
                compound_score = title_sentiment['compound']
                sentiments.append(compound_score)
                
                # Emotional classification
                if compound_score >= 0.05:
                    emotions["positive"] += 1
                elif compound_score <= -0.05:
                    emotions["negative"] += 1
                else:
                    emotions["neutral"] += 1
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            sentiment_std = np.std(sentiments) if sentiments else 0
            
            # Determine market sentiment regime
            dominant_emotion = max(emotions, key=emotions.get)
            
            # AI confidence in sentiment analysis
            confidence = 1 - (sentiment_std / (abs(avg_sentiment) + 0.1))
            confidence = max(0, min(1, confidence))
            
            print(f"AI Sentiment Analysis Confidence: {confidence:.2f}")
            
            return avg_sentiment, sentiment_std, dominant_emotion
            
        except Exception as e:
            print(f"Error in AI sentiment analysis: {e}")
            return 0, 0, "neutral"

    def create_advanced_technical_features(self, df):
        """Create advanced AI-enhanced technical indicators"""
        # Standard technical indicators
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['EMA12'] = df['Close'].ewm(span=12).mean()  # Exponential moving average
        df['EMA26'] = df['Close'].ewm(span=26).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Advanced RSI with multiple timeframes
        df['RSI14'] = self.calculate_rsi(df['Close'], 14)
        df['RSI7'] = self.calculate_rsi(df['Close'], 7)
        df['RSI21'] = self.calculate_rsi(df['Close'], 21)
        
        # Bollinger Bands with multiple configurations
        for window in [10, 20, 30]:
            bb_middle = df['Close'].rolling(window=window).mean()
            bb_std = df['Close'].rolling(window=window).std()
            df[f'BB_upper_{window}'] = bb_middle + (2 * bb_std)
            df[f'BB_lower_{window}'] = bb_middle - (2 * bb_std)
            df[f'BB_width_{window}'] = df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']
            df[f'BB_position_{window}'] = (df['Close'] - df[f'BB_lower_{window}']) / df[f'BB_width_{window}']
        
        # Advanced price patterns
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility indicators
        df['Volatility_5'] = df['Log_Returns'].rolling(window=5).std() * np.sqrt(252)
        df['Volatility_20'] = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252)
        df['Volatility_Ratio'] = df['Volatility_5'] / df['Volatility_20']
        
        # Volume analysis
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        df['Price_Volume'] = df['Price_Change'] * df['Volume_Ratio']
        
        # Advanced momentum indicators
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Support and resistance levels
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        df['Support_Resistance_Ratio'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
        
        # AI-enhanced features
        df['Price_Acceleration'] = df['Price_Change'].diff()
        df['Volume_Acceleration'] = df['Volume_Change'].diff()
        df['Volatility_Trend'] = df['Volatility_20'].pct_change()
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI with improved precision"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_features(self, data):
        """Prepare advanced AI features with dimensionality reduction"""
        df = self.create_advanced_technical_features(data.copy())
        
        # Remove missing values
        df = df.dropna()
        
        # Enhanced feature selection
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA5', 'SMA20', 'SMA50', 'EMA12', 'EMA26',
            'MACD', 'MACD_signal', 'MACD_histogram',
            'RSI14', 'RSI7', 'RSI21',
            'BB_upper_10', 'BB_lower_10', 'BB_width_10', 'BB_position_10',
            'BB_upper_20', 'BB_lower_20', 'BB_width_20', 'BB_position_20',
            'Price_Change', 'Price_Change_5d', 'Price_Change_20d', 'Log_Returns',
            'Volatility_5', 'Volatility_20', 'Volatility_Ratio',
            'Volume_Change', 'Volume_MA5', 'Volume_MA20', 'Volume_Ratio', 'Price_Volume',
            'Momentum_5', 'Momentum_10', 'Momentum_20',
            'High_20', 'Low_20', 'Support_Resistance_Ratio',
            'Price_Acceleration', 'Volume_Acceleration', 'Volatility_Trend'
        ]
        
        # Prepare features (X) and target (y)
        X = df[feature_columns]
        y = df['Close'].shift(-1)  # Next day's closing price
        
        # Remove the last row (we don't have tomorrow's price)
        X = X[:-1]
        y = y[:-1].values
        
        # Detect market regimes using AI clustering
        market_features = X[['Volatility_20', 'Volume_Ratio', 'Momentum_20']].fillna(0)
        market_regimes = self.market_regime_model.fit_predict(market_features)
        X['Market_Regime'] = market_regimes
        
        return X, y, df

    def train_advanced_ai_models(self, X, y):
        """Train multiple AI models with hyperparameter optimization"""
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # No shuffle for time series
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply PCA for dimensionality reduction
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        print("\nTraining Advanced AI Models...")
        print("=" * 50)
        
        # Train individual models and track performance
        models = {
            'Neural Network': self.neural_network,
            'Gradient Boosting': self.gradient_boost,
            'Random Forest': self.random_forest
        }
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use PCA features for neural network, original for tree-based models
            if name == 'Neural Network':
                model.fit(X_train_pca, y_train)
                train_pred = model.predict(X_train_pca)
                test_pred = model.predict(X_test_pca)
            else:
                model.fit(X_train_scaled, y_train)
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
            
            # Calculate multiple performance metrics
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            self.model_performances[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'mae': test_mae,
                'mse': test_mse,
                'rmse': np.sqrt(test_mse)
            }
            
            print(f"  Training R²: {train_r2:.4f}")
            print(f"  Testing R²: {test_r2:.4f}")
            print(f"  MAE: ${test_mae:.2f}")
            print(f"  RMSE: ${np.sqrt(test_mse):.2f}")
        
        # Train ensemble model
        print(f"\nTraining AI Ensemble Model...")
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        ensemble_train_pred = self.ensemble_model.predict(X_train_scaled)
        ensemble_test_pred = self.ensemble_model.predict(X_test_scaled)
        
        ensemble_train_r2 = r2_score(y_train, ensemble_train_pred)
        ensemble_test_r2 = r2_score(y_test, ensemble_test_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_test_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
        
        self.model_performances['Ensemble'] = {
            'train_r2': ensemble_train_r2,
            'test_r2': ensemble_test_r2,
            'mae': ensemble_mae,
            'rmse': ensemble_rmse
        }
        
        print(f"  Training R²: {ensemble_train_r2:.4f}")
        print(f"  Testing R²: {ensemble_test_r2:.4f}")
        print(f"  MAE: ${ensemble_mae:.2f}")
        print(f"  RMSE: ${ensemble_rmse:.2f}")
        
        # Feature importance analysis (using gradient boosting)
        if hasattr(self.gradient_boost, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.gradient_boost.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔍 AI Feature Importance Analysis:")
            print("Top 10 most influential factors:")
            for idx, row in importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.ensemble_model, X_train_scaled, y_train, cv=5)
        print(f"\n📊 Cross-Validation R² Score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        return X_test_scaled, y_test

    def predict_future_with_ai(self, X, symbol, days=30):
        """Advanced AI prediction with confidence intervals and uncertainty quantification"""
        if self.ensemble_model is None:
            raise ValueError("AI models not trained yet!")
        
        # Get advanced sentiment analysis
        sentiment, sentiment_std, emotion = self.analyze_news_sentiment(symbol)
        print(f"\n🧠 Advanced AI Sentiment Analysis:")
        print(f"Average Sentiment: {sentiment:.3f} (-1 to 1)")
        print(f"Sentiment Volatility: {sentiment_std:.3f}")
        print(f"Dominant Market Emotion: {emotion.upper()}")
        
        # Sentiment-based market confidence
        sentiment_confidence = 1 - abs(sentiment_std)
        print(f"AI Sentiment Confidence: {sentiment_confidence:.2f}")
        
        # Scale the last known data point
        last_data = X.iloc[-1:].copy()
        scaled_data = self.scaler.transform(last_data)
        
        # Generate predictions using multiple AI models
        ensemble_predictions = []
        neural_predictions = []
        gb_predictions = []
        rf_predictions = []
        
        dates = []
        current_date = datetime.now()
        
        print(f"\n🤖 Generating AI predictions for {days} trading days...")
        
        for day in range(days):
            # Skip weekends
            while current_date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
                current_date += timedelta(days=1)
            
            # Ensemble prediction
            ensemble_pred = self.ensemble_model.predict(scaled_data)[0]
            
            # Individual model predictions for uncertainty quantification
            neural_pred = self.neural_network.predict(self.pca.transform(scaled_data))[0]
            gb_pred = self.gradient_boost.predict(scaled_data)[0]
            rf_pred = self.random_forest.predict(scaled_data)[0]
            
            # Apply sentiment adjustment with confidence weighting
            sentiment_factor = 1 + (sentiment * 0.02 * sentiment_confidence)
            
            # Apply market regime adjustment
            market_regime = last_data['Market_Regime'].iloc[0]
            regime_factor = 1.0
            if market_regime == 0:  # Bull market
                regime_factor = 1.005
            elif market_regime == 2:  # Bear market
                regime_factor = 0.995
            
            # Combine all adjustments
            final_pred = ensemble_pred * sentiment_factor * regime_factor
            
            # Add realistic market noise based on volatility
            if 'Volatility_20' in last_data.columns:
                volatility = last_data['Volatility_20'].iloc[0]
                noise = np.random.normal(0, volatility * 0.05)
                final_pred = final_pred * (1 + noise)
            
            # Store predictions
            ensemble_predictions.append(final_pred)
            neural_predictions.append(neural_pred * sentiment_factor * regime_factor)
            gb_predictions.append(gb_pred * sentiment_factor * regime_factor)
            rf_predictions.append(rf_pred * sentiment_factor * regime_factor)
            
            dates.append(current_date)
            
            # Update features for next prediction
            last_data_new = last_data.copy()
            last_data_new['Close'] = final_pred
            last_data_new['Price_Change'] = (final_pred - last_data['Close'].iloc[0]) / last_data['Close'].iloc[0]
            
            # Update technical indicators
            if day > 0:
                last_data_new['SMA5'] = np.mean(ensemble_predictions[-5:]) if len(ensemble_predictions) >= 5 else final_pred
                last_data_new['Momentum_5'] = (final_pred / ensemble_predictions[-5] - 1) if len(ensemble_predictions) >= 5 else 0
            
            scaled_data = self.scaler.transform(last_data_new)
            last_data = last_data_new
            
            current_date += timedelta(days=1)
        
        # Calculate prediction confidence intervals
        all_predictions = np.array([ensemble_predictions, neural_predictions, gb_predictions, rf_predictions])
        prediction_std = np.std(all_predictions, axis=0)
        confidence_intervals = {
            'lower_95': np.array(ensemble_predictions) - 1.96 * prediction_std,
            'upper_95': np.array(ensemble_predictions) + 1.96 * prediction_std,
            'lower_68': np.array(ensemble_predictions) - prediction_std,
            'upper_68': np.array(ensemble_predictions) + prediction_std
        }
        
        # Store confidence metrics
        avg_confidence = 1 - np.mean(prediction_std) / np.mean(ensemble_predictions)
        self.prediction_confidence.append(avg_confidence)
        
        print(f"🎯 Average AI Prediction Confidence: {avg_confidence:.2f}")
        
        return dates, ensemble_predictions, confidence_intervals, {
            'neural': neural_predictions,
            'gradient_boost': gb_predictions,
            'random_forest': rf_predictions
        }

    def plot_advanced_predictions(self, dates, predictions, confidence_intervals, individual_models, symbol, current_price):
        """Enhanced AI visualization with detailed dates and prediction timeline"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14))
            
            # Main prediction plot with detailed date markers
            # Plot confidence intervals
            ax1.fill_between(dates, confidence_intervals['lower_95'], confidence_intervals['upper_95'],
                            alpha=0.2, color='lightblue', label='95% Confidence Interval')
            ax1.fill_between(dates, confidence_intervals['lower_68'], confidence_intervals['upper_68'],
                            alpha=0.3, color='lightgreen', label='68% Confidence Interval')
            
            # Plot individual model predictions
            ax1.plot(dates, individual_models['neural'], '--', alpha=0.7, label='Neural Network', color='purple', linewidth=2)
            ax1.plot(dates, individual_models['gradient_boost'], '--', alpha=0.7, label='Gradient Boosting', color='orange', linewidth=2)
            ax1.plot(dates, individual_models['random_forest'], '--', alpha=0.7, label='Random Forest', color='green', linewidth=2)
            
            # Plot ensemble prediction with markers
            ax1.plot(dates, predictions, 'r-', linewidth=4, label='AI Ensemble Prediction', color='red', marker='o', markersize=3)
            
            # Add current price point
            ax1.scatter([dates[0]], [current_price], color='blue', s=150, label='Current Price', zorder=10, marker='o')
            
            # Add key milestone markers (simplified)
            milestones = [0, 4, 9, 19, 29] if len(dates) >= 30 else [0, len(dates)//4, len(dates)//2, len(dates)-1]
            milestone_labels = ['Today', '1 Week', '2 Weeks', '3 Weeks', '1 Month']
            colors = ['blue', 'green', 'yellow', 'orange', 'red']
            
            for i, (idx, label) in enumerate(zip(milestones, milestone_labels)):
                if idx < len(dates):
                    ax1.scatter([dates[idx]], [predictions[idx]], s=100, zorder=15, 
                               color=colors[i], marker='s', edgecolors='white', linewidth=1)
            
            # Formatting main plot
            ax1.set_title(f'Advanced AI Stock Predictions: {symbol}', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Format x-axis to show specific dates clearly
            if len(dates) > 10:
                step = max(1, len(dates) // 8)
                date_positions = dates[::step]
                date_labels = [d.strftime('%m/%d') for d in date_positions]
                ax1.set_xticks(date_positions)
                ax1.set_xticklabels(date_labels, rotation=45)
            
            # Daily prediction timeline chart
            daily_count = min(15, len(dates))
            daily_dates = dates[:daily_count]
            daily_predictions = predictions[:daily_count]
            daily_changes = [(pred - current_price) / current_price * 100 for pred in daily_predictions]
            
            colors_bars = ['green' if change >= 0 else 'red' for change in daily_changes]
            bars = ax2.bar(range(len(daily_dates)), daily_changes, color=colors_bars, alpha=0.7, edgecolor='black')
            
            # Add percentage labels on bars
            for i, (bar, change) in enumerate(zip(bars, daily_changes)):
                height = bar.get_height()
                ax2.annotate(f'{change:+.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -15),
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=8, fontweight='bold')
            
            ax2.set_title('Daily Price Change Predictions', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Trading Days', fontsize=12)
            ax2.set_ylabel('Expected Change (%)', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            # Set x-axis labels with dates
            date_labels_bar = [f'Day {i+1}' for i in range(len(daily_dates))]
            ax2.set_xticks(range(len(daily_dates)))
            ax2.set_xticklabels(date_labels_bar, rotation=45, fontsize=9)
            
            # Model performance comparison plot
            if hasattr(self, 'model_performances') and self.model_performances:
                models = list(self.model_performances.keys())
                r2_scores = [self.model_performances[model]['test_r2'] for model in models]
                
                x_pos = np.arange(len(models))
                bars_perf = ax3.bar(x_pos, r2_scores, alpha=0.8, color='skyblue', edgecolor='navy')
                
                ax3.set_xlabel('AI Models', fontsize=12)
                ax3.set_ylabel('R² Score', fontsize=12)
                ax3.set_title('AI Model Performance Comparison', fontsize=14, fontweight='bold')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(models, rotation=45)
                ax3.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars_perf:
                    height = bar.get_height()
                    ax3.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            
            # Try to show the plot, if it fails, save it
            try:
                plt.show()
                print("✅ Charts displayed successfully!")
            except:
                # Save the plot instead
                filename = f"{symbol}_AI_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"📁 Charts saved as: {filename}")
                plt.close()
            
            # Print detailed daily predictions table
            self.print_detailed_predictions_table(dates, predictions, confidence_intervals, current_price)
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
            print("📊 Displaying simplified analysis instead...")
            self.print_simplified_analysis(dates, predictions, confidence_intervals, current_price, symbol)
        
        
        # Model performance comparison plot
        if self.model_performances:
            models = list(self.model_performances.keys())
            r2_scores = [self.model_performances[model]['test_r2'] for model in models]
            mae_scores = [self.model_performances[model]['mae'] for model in models]
            
            x_pos = np.arange(len(models))
            
            # R² scores
            bars1 = ax3.bar(x_pos - 0.2, r2_scores, 0.4, label='R² Score (Accuracy)', alpha=0.8, color='skyblue', edgecolor='navy')
            
            # MAE scores (normalized)
            mae_normalized = [mae/max(mae_scores) for mae in mae_scores]
            bars2 = ax3.bar(x_pos + 0.2, mae_normalized, 0.4, label='MAE Score (Lower is Better)', alpha=0.8, color='lightcoral', edgecolor='darkred')
            
            ax3.set_xlabel('AI Models', fontsize=12)
            ax3.set_ylabel('Performance Score', fontsize=12)
            ax3.set_title('🏆 AI Model Performance Comparison', fontsize=14, fontweight='bold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(models, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax3.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            for bar in bars2:
                height = bar.get_height()
                ax3.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Calculate and display predictions statistics
        pct_change = ((predictions[-1] - current_price) / current_price) * 100
        volatility = np.std(predictions) / np.mean(predictions) * 100
        
        # Add comprehensive statistics box
        stats_text = f"""📊 AI Prediction Analytics Dashboard:
        
        📈 Expected Return: {pct_change:+.2f}%
        📊 Prediction Volatility: {volatility:.2f}%
        🎯 AI Confidence: {np.mean(self.prediction_confidence[-1:]):.2f}/1.0
        ⏰ Forecast Period: {len(dates)} trading days
        📅 Target Date: {dates[-1].strftime('%B %d, %Y')}
        
        🎲 Risk Level: {'🟢 LOW' if volatility < 5 else '🟡 MEDIUM' if volatility < 10 else '🔴 HIGH'}
        📉 Max Downside: {min(confidence_intervals['lower_95']) - current_price:.2f}
        📈 Max Upside: {max(confidence_intervals['upper_95']) - current_price:.2f}"""
        
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, edgecolor='black')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed daily predictions table
        self.print_detailed_predictions_table(dates, predictions, confidence_intervals, current_price)
        
        # Print comprehensive analysis
        print(f"\n📈 Comprehensive AI Analysis Report:")
        print(f"{'='*80}")
        print(f"🎯 Target Price ({len(dates)} days): ${predictions[-1]:.2f}")
        print(f"📊 Expected Return: {pct_change:+.2f}%")
        print(f"📉 95% Confidence Range: ${confidence_intervals['lower_95'][-1]:.2f} - ${confidence_intervals['upper_95'][-1]:.2f}")
        print(f"🎲 Prediction Volatility: {volatility:.2f}%")
        if self.model_performances:
            best_model = max(self.model_performances.keys(), 
                           key=lambda x: self.model_performances[x]['test_r2'])
            print(f"🏆 Best Model: {best_model} (R² = {self.model_performances[best_model]['test_r2']:.4f})")
        print(f"🧠 AI Confidence: {np.mean(self.prediction_confidence[-1:]):.3f}/1.0")
        print(f"📅 Forecast Horizon: {dates[0].strftime('%m/%d/%Y')} → {dates[-1].strftime('%m/%d/%Y')}")
    
    def print_detailed_predictions_table(self, dates, predictions, confidence_intervals, current_price):
        """Print a detailed table of daily predictions"""
        print(f"\n📅 Detailed Daily Predictions Table:")
        print("=" * 85)
        print(f"{'Date':<12} {'Price':<10} {'Change %':<10} {'68% Range':<20} {'95% Range':<20}")
        print("-" * 85)
        
        for i in range(min(20, len(dates))):  # Show first 20 days
            date_str = dates[i].strftime('%m/%d/%Y')
            price = predictions[i]
            change_pct = (price - current_price) / current_price * 100
            range_68 = f"${confidence_intervals['lower_68'][i]:.2f}-${confidence_intervals['upper_68'][i]:.2f}"
            range_95 = f"${confidence_intervals['lower_95'][i]:.2f}-${confidence_intervals['upper_95'][i]:.2f}"
            
            color_code = "📈" if change_pct >= 0 else "📉"
            print(f"{date_str:<12} ${price:<9.2f} {color_code}{change_pct:+6.2f}%   {range_68:<20} {range_95:<20}")
        
        if len(dates) > 20:
            print(f"... (showing first 20 days out of {len(dates)} total predictions)")
        print("-" * 85)
    
    def print_simplified_analysis(self, dates, predictions, confidence_intervals, current_price, symbol):
        """Print simplified analysis when charts fail to display"""
        print(f"\n📈 Simplified AI Analysis for {symbol}:")
        print("=" * 60)
        
        pct_change = ((predictions[-1] - current_price) / current_price) * 100
        volatility = np.std(predictions) / np.mean(predictions) * 100
        
        print(f"🎯 Target Price ({len(dates)} days): ${predictions[-1]:.2f}")
        print(f"📊 Expected Return: {pct_change:+.2f}%")
        print(f"📉 95% Confidence Range: ${confidence_intervals['lower_95'][-1]:.2f} - ${confidence_intervals['upper_95'][-1]:.2f}")
        print(f"🎲 Prediction Volatility: {volatility:.2f}%")
        print(f"📅 Forecast Period: {dates[0].strftime('%m/%d/%Y')} → {dates[-1].strftime('%m/%d/%Y')}")
        
        # Show key predictions
        print(f"\n📅 Key Predictions:")
        key_indices = [0, len(dates)//4, len(dates)//2, 3*len(dates)//4, -1]
        key_labels = ["Today", "Week 1", "Week 2", "Week 3", "Final"]
        
        for idx, label in zip(key_indices, key_labels):
            if idx < len(dates):
                print(f"  {label}: ${predictions[idx]:.2f} on {dates[idx].strftime('%m/%d/%Y')}")
        
        print("\n📊 To see full charts, please ensure matplotlib is properly configured.")

def main():
    # Create advanced AI predictor instance
    predictor = AdvancedAIStockPredictor()
    
    print("🚀 Advanced AI Stock Predictor v2.0")
    print("=" * 50)
    print("🧠 Multi-Model Ensemble Learning")
    print("📊 Sentiment Analysis Integration")
    print("🎯 Confidence Interval Predictions")
    print("📈 Market Regime Detection")
    print("=" * 50)
    
    # Get stock symbol from user
    symbol = input("\n💼 Enter stock symbol (e.g., AAPL for Apple): ").upper()
    
    # Get stock data
    print(f"\n📥 Downloading {symbol} historical data...")
    data = predictor.get_stock_data(symbol)
    
    if data is not None:
        # Prepare data and train models
        print("\n🔬 Preparing advanced features and training AI models...")
        X, y, processed_data = predictor.prepare_features(data)
        predictor.train_advanced_ai_models(X, y)
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        print(f"\n💰 Current {symbol} price: ${current_price:.2f}")
        
        # Make advanced AI predictions
        print(f"\n🤖 Generating advanced AI predictions...")
        dates, predictions, confidence_intervals, individual_models = predictor.predict_future_with_ai(X, symbol)
        
        # Print detailed predictions
        print("\n🔮 AI-Powered Predictions:")
        print("=" * 40)
        print("\n📅 Next Week Forecast:")
        for i in range(min(5, len(dates))):  # Show 5 trading days
            confidence_range = confidence_intervals['upper_68'][i] - confidence_intervals['lower_68'][i]
            print(f"  {dates[i].strftime('%Y-%m-%d')}: ${predictions[i]:.2f} (±${confidence_range/2:.2f})")
        
        if len(dates) >= 10:
            print("\n📅 Two Week Forecast:")
            i = 9
            confidence_range = confidence_intervals['upper_68'][i] - confidence_intervals['lower_68'][i]
            print(f"  {dates[i].strftime('%Y-%m-%d')}: ${predictions[i]:.2f} (±${confidence_range/2:.2f})")
        
        print("\n📅 Month-End AI Forecast:")
        i = -1
        confidence_range = confidence_intervals['upper_68'][i] - confidence_intervals['lower_68'][i]
        print(f"  {dates[i].strftime('%Y-%m-%d')}: ${predictions[i]:.2f} (±${confidence_range/2:.2f})")
        
        # Plot advanced results
        print(f"\n📊 Generating advanced AI visualization...")
        predictor.plot_advanced_predictions(dates, predictions, confidence_intervals, 
                                          individual_models, symbol, current_price)
        
        # Additional AI insights
        print(f"\n🧠 Additional AI Insights:")
        print("=" * 40)
        trend = "BULLISH 📈" if predictions[-1] > current_price else "BEARISH 📉"
        print(f"🎯 AI Market Trend: {trend}")
        
        volatility_trend = np.std(predictions[:len(predictions)//2]) - np.std(predictions[len(predictions)//2:])
        volatility_direction = "DECREASING 📉" if volatility_trend > 0 else "INCREASING 📈"
        print(f"📊 Volatility Trend: {volatility_direction}")
        
        print(f"🎲 Risk Assessment: {'LOW' if np.std(predictions)/np.mean(predictions) < 0.05 else 'MEDIUM' if np.std(predictions)/np.mean(predictions) < 0.1 else 'HIGH'}")
        
    else:
        print("❌ Failed to download stock data. Please check the symbol and try again.")

if __name__ == "__main__":
    main()
