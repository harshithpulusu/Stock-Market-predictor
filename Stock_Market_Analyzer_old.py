import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class StockMarketAnalyzer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.results = {}

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.feature_names = []

    def load_data(self, filepath): 
        """Load stock market data from CSV with enhanced error handling"""
        try:
            data = pd.read_csv(filepath)
            
            # Ensure Date column exists and is properly formatted
            if 'Date' not in data.columns:
                print("Warning: No 'Date' column found. Adding sequential dates.")
                data['Date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
            else:
                data['Date'] = pd.to_datetime(data['Date'])
            
            # Sort by date to ensure chronological order
            data = data.sort_values('Date').reset_index(drop=True)
            
            print(f"Data loaded successfully: {len(data)} rows, {len(data.columns)} columns")
            print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
            
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    def create_technical_indicators(self, data):
        """Create comprehensive technical indicators"""
        df = data.copy()
        
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA12'] = df['Close'].ewm(span=12).mean()
        df['EMA26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Price changes and volatility
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=10).std()
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Time-based features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        
        return df

    def preprocess_data(self, data, target_days=1):
        """Enhanced preprocessing with multiple prediction horizons"""
        df = self.create_technical_indicators(data)
        
        # Create target variable (future price)
        df[f'Target_{target_days}d'] = df['Close'].shift(-target_days)
        
        # Select features for prediction
        feature_columns = [
            'Open', 'High', 'Low', 'Volume',
            'MA5', 'MA10', 'MA20', 'MA50',
            'EMA12', 'EMA26', 'MACD', 'MACD_signal', 'MACD_histogram',
            'RSI', 'BB_width', 'BB_position',
            'Price_Change', 'High_Low_Pct', 'Volume_Change', 'Volatility',
            'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5',
            'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3', 'Volume_lag_5',
            'DayOfWeek', 'Month', 'Quarter'
        ]
        
        # Filter existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Drop rows with NaN values
        df_clean = df.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No data remaining after cleaning. Check your input data.")
        
        X = df_clean[available_features]
        y = df_clean[f'Target_{target_days}d']
        dates = df_clean['Date']
        
        self.feature_names = available_features
        
        print(f"Features used: {len(available_features)}")
        print(f"Clean data points: {len(X)}")
        
        return X, y, dates

    def train_and_evaluate(self, X, y):
        """Train multiple models and select the best one"""
        # Split data chronologically (important for time series)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_score = float('-inf')
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if name == 'linear':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            self.results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            
            if r2 > best_score:
                best_score = r2
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} (R² = {best_score:.4f})")
        return X_test, y_test

    def get_feature_importance(self):
        """Get feature importance for tree-based models"""
        if self.best_model_name in ['random_forest', 'gradient_boost']:
            importance = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            print("Feature importance only available for tree-based models")
            return None

    def plot_results(self, X_test, y_test):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Model comparison
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['r2'] for name in model_names]
        rmse_scores = [self.results[name]['rmse'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores, color=['blue', 'green', 'red'])
        axes[0, 0].set_title('Model Comparison - R² Score')
        axes[0, 0].set_ylabel('R² Score')
        
        # 2. Actual vs Predicted for best model
        best_results = self.results[self.best_model_name]
        axes[0, 1].scatter(best_results['actual'], best_results['predictions'], alpha=0.6)
        axes[0, 1].plot([best_results['actual'].min(), best_results['actual'].max()], 
                       [best_results['actual'].min(), best_results['actual'].max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Price')
        axes[0, 1].set_ylabel('Predicted Price')
        axes[0, 1].set_title(f'Actual vs Predicted - {self.best_model_name}')
        
        # 3. Time series plot
        test_dates = range(len(best_results['actual']))
        axes[1, 0].plot(test_dates, best_results['actual'], label='Actual', linewidth=2)
        axes[1, 0].plot(test_dates, best_results['predictions'], label='Predicted', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Price')
        axes[1, 0].set_title('Price Prediction Over Time')
        axes[1, 0].legend()
        
        # 4. Residuals plot
        residuals = best_results['actual'] - best_results['predictions']
        axes[1, 1].scatter(best_results['predictions'], residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Price')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Plot')
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance plot
        feature_importance = self.get_feature_importance()
        if feature_importance is not None:
            plt.figure(figsize=(10, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importance - {self.best_model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

    def predict_future(self, data, days_ahead=5):
        """Predict future prices"""
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        
        # Get the latest features
        df_with_indicators = self.create_technical_indicators(data)
        latest_features = df_with_indicators[self.feature_names].iloc[-1:].values
        
        predictions = []
        
        if self.best_model_name == 'linear':
            latest_features_scaled = self.scaler.transform(latest_features)
            for _ in range(days_ahead):
                pred = self.best_model.predict(latest_features_scaled)[0]
                predictions.append(pred)
        else:
            for _ in range(days_ahead):
                pred = self.best_model.predict(latest_features)[0]
                predictions.append(pred)
        
        return predictions

    def generate_report(self):
        """Generate a comprehensive analysis report"""
        if not self.results:
            print("No results available. Please train the model first.")
            return
        
        print("\n" + "="*60)
        print("STOCK MARKET PREDICTION ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nBest performing model: {self.best_model_name}")
        best_results = self.results[self.best_model_name]
        
        print(f"\nModel Performance Metrics:")
        print(f"  • R² Score: {best_results['r2']:.4f}")
        print(f"  • Root Mean Square Error: ${best_results['rmse']:.2f}")
        print(f"  • Mean Absolute Error: ${best_results['mae']:.2f}")
        
        print(f"\nAll Models Comparison:")
        for name, results in self.results.items():
            print(f"  {name}: R²={results['r2']:.4f}, RMSE=${results['rmse']:.2f}")
        
        feature_importance = self.get_feature_importance()
        if feature_importance is not None:
            print(f"\nTop 10 Most Important Features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  • {row['feature']}: {row['importance']:.4f}")

# Example usage and testing
if __name__ == "__main__":
    # Create predictor instance
    predictor = StockMarketAnalyzer()
    
    try:
        # Load data (replace with your actual file)
        print("Loading stock market data...")
        data = predictor.load_data("stock_data.csv")
        
        if data is not None:
            # Preprocess data
            print("\nPreprocessing data and creating technical indicators...")
            X, y, dates = predictor.preprocess_data(data, target_days=1)
            
            # Train models
            print("\nTraining multiple models...")
            X_test, y_test = predictor.train_and_evaluate(X, y)
            
            # Generate predictions for next 5 days
            print("\nGenerating future predictions...")
            future_predictions = predictor.predict_future(data, days_ahead=5)
            
            print(f"\nFuture price predictions (next 5 days):")
            for i, pred in enumerate(future_predictions, 1):
                print(f"  Day {i}: ${pred:.2f}")
            
            # Generate comprehensive report
            predictor.generate_report()
            
            # Plot results (uncomment to show plots)
            # predictor.plot_results(X_test, y_test)
            
    except FileNotFoundError:
        print("Error: stock_data.csv not found. Please ensure you have a CSV file with columns:")
        print("Date, Open, High, Low, Close, Volume")
        print("\nCreating sample data for demonstration...")
        
        # Create sample data for demonstration
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            'High': 0,
            'Low': 0,
            'Close': 0,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Generate realistic OHLC data
        for i in range(len(sample_data)):
            open_price = sample_data.loc[i, 'Open']
            high_price = open_price + np.random.uniform(0, 5)
            low_price = open_price - np.random.uniform(0, 5)
            close_price = np.random.uniform(low_price, high_price)
            
            sample_data.loc[i, 'High'] = high_price
            sample_data.loc[i, 'Low'] = low_price
            sample_data.loc[i, 'Close'] = close_price
            
            # Update next open price based on close
            if i < len(sample_data) - 1:
                sample_data.loc[i + 1, 'Open'] = close_price + np.random.normal(0, 0.5)
        
        print("Sample data created. Testing with sample data...")
        X, y, dates = predictor.preprocess_data(sample_data, target_days=1)
        X_test, y_test = predictor.train_and_evaluate(X, y)
        future_predictions = predictor.predict_future(sample_data, days_ahead=5)
        
        print(f"\nSample future predictions:")
        for i, pred in enumerate(future_predictions, 1):
            print(f"  Day {i}: ${pred:.2f}")
            
        predictor.generate_report()
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()