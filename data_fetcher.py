import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class StockDataFetcher:
    def __init__(self):
        pass

    def fetch_stock_data(self, symbol, period='1y'):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL' for Apple)
            period (str): Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            pandas.DataFrame: Historical stock data
        """
        try:
            # Create a Ticker object
            stock = yf.Ticker(symbol)
            
            # Get historical data
            df = stock.history(period=period)
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Save to CSV for future use
            filename = f"{symbol}_stock_data.csv"
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_company_info(self, symbol):
        """
        Get company information
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Company information
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'description': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    fetcher = StockDataFetcher()
    
    # Fetch Apple stock data
    symbol = "AAPL"
    data = fetcher.fetch_stock_data(symbol)
    
    if data is not None:
        print("\nStock Data Preview:")
        print(data.head())
        
        print("\nCompany Information:")
        info = fetcher.get_company_info(symbol)
        if info:
            for key, value in info.items():
                print(f"{key}: {value}")
