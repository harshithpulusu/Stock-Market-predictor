# 🚀 AI Stock Market Predictor Web Application

A comprehensive, AI-powered stock market analysis and prediction tool built with Python, Streamlit, and advanced machine learning algorithms.

## 🌟 Features

### 📊 **Advanced Technical Analysis**
- **Moving Averages**: 5, 20, and 50-day moving averages
- **RSI (Relative Strength Index)**: Momentum oscillator for overbought/oversold conditions
- **MACD**: Moving Average Convergence Divergence with signal line and histogram
- **Bollinger Bands**: Volatility bands with position analysis
- **Volume Analysis**: Volume ratios and moving averages
- **Price Patterns**: Support and resistance levels

### 🤖 **AI-Powered Predictions**
- **Ensemble Learning**: Combines Random Forest, Gradient Boosting, and Linear Regression
- **Time Series Validation**: Proper cross-validation to prevent data leakage
- **Feature Engineering**: 17+ technical indicators and lagged features
- **Confidence Scoring**: AI model confidence assessment
- **Multi-timeframe**: 7 to 60-day prediction capability

### 📈 **Interactive Visualizations**
- **Real-time Charts**: Interactive Plotly charts with zoom and hover features
- **Technical Indicators**: Separate charts for RSI, MACD, Bollinger Bands, and Volume
- **Prediction Overlay**: Visual prediction lines on price charts
- **Feature Importance**: Bar charts showing most influential factors
- **Responsive Design**: Works on desktop and mobile devices

### 💼 **Professional Features**
- **Company Information**: Real-time company data and sector analysis
- **Performance Metrics**: Volatility, returns, and risk assessment
- **Investment Signals**: BUY/SELL/HOLD recommendations with confidence levels
- **Weekly Targets**: Detailed weekly price predictions
- **Data Export**: Download predictions and analysis as CSV
- **Progress Tracking**: Real-time progress bars for analysis steps

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Stock-Market-predictor.git
cd Stock-Market-predictor

# Install required packages
pip install -r requirements.txt

# Or install manually:
pip install streamlit plotly yfinance pandas numpy scikit-learn matplotlib seaborn
```

### Launch the Application
```bash
# Method 1: Using the launcher script
python3 launch_app.py

# Method 2: Direct streamlit command
streamlit run stock_predictor_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📱 How to Use

### 1. **Enter Stock Symbol**
- Type any valid stock symbol (e.g., AAPL, TSLA, MSFT, GOOGL)
- Use the sidebar for easy access

### 2. **Configure Analysis**
- **Data Period**: Choose from 6 months to 5 years of historical data
- **Prediction Days**: Set prediction timeframe (7-60 days)
- Click "🔮 Analyze Stock" to start

### 3. **Review Results**
- **Company Overview**: Current price, sector, and basic information
- **AI Predictions**: Target prices and expected returns
- **Interactive Charts**: Zoom, pan, and hover for detailed information
- **Technical Analysis**: Multiple indicator charts
- **Model Performance**: AI confidence and accuracy metrics

### 4. **Download Results**
- Export predictions and analysis data as CSV
- Save charts and reports for future reference

## 🧠 AI Model Architecture

### **Ensemble Learning Approach**
The application uses multiple machine learning models and selects the best performer:

1. **Random Forest Regressor**
   - 100 estimators with depth control
   - Handles non-linear relationships
   - Provides feature importance

2. **Gradient Boosting Regressor**
   - Sequential learning with error correction
   - Excellent for time series patterns
   - Robust to overfitting

3. **Linear Regression**
   - Simple baseline model
   - Good for linear trends
   - Fast computation

### **Feature Engineering**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Lagged Features**: Previous day returns and volume ratios
- **Normalized Features**: Scaled and ratio-based indicators
- **Volume Analysis**: Trading volume patterns and anomalies

### **Validation Strategy**
- **Time Series Cross-Validation**: Prevents look-ahead bias
- **Walk-Forward Analysis**: Realistic backtesting approach
- **Model Comparison**: Automatic best model selection
- **Confidence Metrics**: R² scores and standard deviations

## 📊 File Structure

```
Stock-Market-predictor/
├── stock_predictor_app.py      # Main Streamlit web application
├── launch_app.py               # Application launcher script
├── optimized_stock_predictor.py # Console-based predictor
├── improved_ai_predictor.py    # Advanced AI version
├── simple_reliable_predictor.py # Simplified reliable version
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

## 🔧 Advanced Configuration

### **Custom Model Parameters**
You can modify model parameters in `stock_predictor_app.py`:

```python
# Random Forest settings
RandomForestRegressor(
    n_estimators=100,        # Number of trees
    max_depth=10,           # Maximum tree depth
    min_samples_split=10,   # Minimum samples to split
    min_samples_leaf=5,     # Minimum samples per leaf
    random_state=42
)

# Gradient Boosting settings
GradientBoostingRegressor(
    n_estimators=100,       # Number of boosting stages
    max_depth=6,           # Maximum tree depth
    learning_rate=0.1,     # Learning rate
    random_state=42
)
```

### **Feature Selection**
Current features can be customized by modifying the `feature_columns` list:

```python
self.feature_columns = [
    'MA_Ratio_5_20',        # Moving average ratios
    'MA_Ratio_20_50',       
    'Price_to_MA20',        
    'Volatility_5',         # Volatility measures
    'Volatility_20',        
    'Volume_Ratio',         # Volume analysis
    'RSI_Normalized',       # Technical indicators
    'MACD',                 
    'MACD_Histogram',       
    'BB_Position',          # Bollinger Band position
    'Returns_Lag_1',        # Lagged returns
    'Returns_Lag_2',        
    'Returns_Lag_3',        
    'Returns_Lag_5',        
    'Volume_Ratio_Lag_1',   # Lagged volume ratios
    'Volume_Ratio_Lag_2',   
    'Volume_Ratio_Lag_3'    
]
```

## 📈 Performance Metrics

### **Model Evaluation**
- **R² Score**: Coefficient of determination (higher is better)
- **Cross-Validation**: 5-fold time series split validation
- **Standard Deviation**: Model consistency across different time periods
- **Feature Importance**: Relative importance of each technical indicator

### **Prediction Quality**
- **Confidence Level**: Based on model performance (0-100%)
- **Risk Assessment**: LOW/MEDIUM/HIGH based on volatility and confidence
- **Signal Strength**: BUY/SELL/HOLD recommendations with confidence thresholds

## ⚠️ Important Disclaimers

### **Educational Purpose**
This application is designed for educational and research purposes only. It demonstrates:
- Machine learning techniques in finance
- Technical analysis implementation
- Web application development with Streamlit
- Data visualization best practices

### **Investment Risks**
- **No Financial Advice**: This tool does not provide financial advice
- **Past Performance**: Historical performance does not guarantee future results
- **Model Limitations**: AI predictions are based on historical patterns only
- **Market Volatility**: Stock markets are inherently unpredictable
- **Professional Consultation**: Always consult with licensed financial advisors

### **Technical Limitations**
- **Data Quality**: Predictions depend on data quality and availability
- **Model Performance**: Negative R² scores indicate poor predictive power
- **Market Conditions**: Models may not perform well during unusual market conditions
- **Overfitting Risk**: Complex models may overfit to historical data

## 🔮 Future Enhancements

### **Planned Features**
- [ ] Real-time news sentiment analysis
- [ ] Cryptocurrency support
- [ ] Portfolio optimization tools
- [ ] Alert system for price targets
- [ ] Mobile application version
- [ ] Advanced charting tools
- [ ] Social media sentiment integration
- [ ] Fundamental analysis features

### **Technical Improvements**
- [ ] Deep learning models (LSTM, GRU)
- [ ] Reinforcement learning for trading strategies
- [ ] Real-time data streaming
- [ ] Cloud deployment options
- [ ] API for third-party integration
- [ ] Database integration for historical analysis
- [ ] Advanced backtesting framework

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes and improvements
- New feature implementations
- Documentation enhancements
- Performance optimizations
- User interface improvements

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **yfinance**: For providing free stock market data
- **Streamlit**: For the excellent web application framework
- **Plotly**: For interactive charting capabilities
- **scikit-learn**: For machine learning algorithms
- **pandas**: For data manipulation and analysis

## 📞 Support

If you encounter any issues or have questions:
1. Check the FAQ section in the app
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Check your internet connection for data fetching

---

**Happy Trading! 📈 Remember to always do your own research and trade responsibly! 💼**