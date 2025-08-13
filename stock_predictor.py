import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class StockMarketPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """
        Prepare features for the model including technical indicators
        """
        from utils import calculate_technical_indicators
        
        # Calculate technical indicators
        data = calculate_technical_indicators(data)
        
        # Create features
        features = [
            'Open', 'High', 'Low', 'Volume',
            'SMA_5', 'SMA_20', 'EMA_12', 'EMA_26',
            'MACD', 'Signal_Line', 'RSI',
            'BB_middle', 'BB_upper', 'BB_lower'
        ]
        
        # Drop any rows with NaN values
        data = data.dropna()
        
        # Prepare X (features) and y (target)
        X = data[features]
        y = data['Close']
        
        return X, y
        
    def train(self, X, y):
        """
        Train the model using Random Forest Regressor
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train the model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        from utils import evaluate_model
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        print("\nModel Performance:")
        print("Training Metrics:")
        train_metrics = evaluate_model(y_train, train_pred)
        for metric, value in train_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        print("\nTesting Metrics:")
        test_metrics = evaluate_model(y_test, test_pred)
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        return X_test, y_test, test_pred
    
    def predict(self, X_new):
        """
        Make predictions for new data
        """
        if self.model is None:
            raise Exception("Model not trained yet!")
            
        # Scale the features
        X_new_scaled = self.scaler.transform(X_new)
        
        # Make predictions
        predictions = self.model.predict(X_new_scaled)
        return predictions
