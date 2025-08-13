import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for stock analysis
    
    Args:
        df (pandas.DataFrame): Stock data with OHLCV columns
    
    Returns:
        pandas.DataFrame: DataFrame with additional technical indicators
    """
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
    
    return df

def plot_stock_prediction(actual, predicted, dates, title="Stock Price Prediction"):
    """
    Plot actual vs predicted stock prices
    
    Args:
        actual (array-like): Actual stock prices
        predicted (array-like): Predicted stock prices
        dates (array-like): Dates corresponding to the prices
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', color='blue')
    plt.plot(dates, predicted, label='Predicted', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('prediction_plot.png')
    plt.close()

def evaluate_model(y_true, y_pred):
    """
    Calculate various metrics to evaluate model performance
    
    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }
