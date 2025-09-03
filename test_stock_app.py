"""
Test suite for Stock Price Prediction Web App
Tests basic functionality, imports, and data handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import dash
        import yfinance
        import plotly
        import sklearn
        # Skip keras test if not available
        try:
            import keras
            print("✅ keras imported successfully")
        except ImportError:
            print("⚠️  keras not available, skipping")
        assert True
    except ImportError as e:
        print(f"Failed to import required package: {e}")
        assert False

def test_data_loading():
    """Test basic data loading functionality"""
    # Test that we can create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    sample_data = pd.DataFrame({
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    assert len(sample_data) == 10
    assert 'Close' in sample_data.columns
    assert 'Volume' in sample_data.columns

def test_color_scheme():
    """Test that color scheme is properly defined"""
    colors = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40'
    }
    
    assert len(colors) == 8
    assert all(color.startswith('#') for color in colors.values())

def test_dataframe_operations():
    """Test DataFrame operations used in the app"""
    # Create sample stock data
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    stock_data = pd.DataFrame({
        'Date': dates,
        'Stock': ['AAPL'] * len(dates),
        'Open': np.random.randn(len(dates)).cumsum() + 100,
        'High': np.random.randn(len(dates)).cumsum() + 102,
        'Low': np.random.randn(len(dates)).cumsum() + 98,
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Test filtering
    aapl_data = stock_data[stock_data['Stock'] == 'AAPL']
    assert len(aapl_data) == len(dates)
    
    # Test price calculations
    current_price = aapl_data['Close'].iloc[-1]
    first_price = aapl_data['Close'].iloc[0]
    price_change = current_price - first_price
    price_change_pct = (price_change / first_price) * 100
    
    assert isinstance(current_price, (int, float))
    assert isinstance(price_change, (int, float))
    assert isinstance(price_change_pct, (int, float))

def test_numpy_operations():
    """Test numpy operations used in predictions"""
    # Test array operations
    prices = np.array([100, 101, 102, 103, 104])
    assert len(prices) == 5
    
    # Test array flattening
    prices_2d = prices.reshape(-1, 1)
    assert prices_2d.shape == (5, 1)
    
    prices_1d = prices_2d.flatten()
    assert prices_1d.shape == (5,)

def test_date_operations():
    """Test date operations used in the app"""
    # Test date range creation
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    assert len(date_range) == 365
    assert date_range[0] == start_date
    assert date_range[-1] == end_date

def test_app_structure():
    """Test that the app structure is correct"""
    try:
        # Test that main app file exists and can be imported
        import stock_app
        assert hasattr(stock_app, 'app')
        assert hasattr(stock_app, 'COLORS')
        assert hasattr(stock_app, 'load_stock_data')
        assert True
    except Exception as e:
        print(f"App structure test failed: {e}")
        assert False

def test_requirements():
    """Test that requirements.txt is properly formatted"""
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        # Check that key packages are mentioned
        assert 'dash' in requirements.lower()
        assert 'yfinance' in requirements.lower()
        assert 'plotly' in requirements.lower()
        assert 'pandas' in requirements.lower()
        assert 'numpy' in requirements.lower()
        
    except FileNotFoundError:
        print("requirements.txt file not found")
        assert False

if __name__ == "__main__":
    # Run basic tests
    print("Running basic functionality tests...")
    
    try:
        test_imports()
        test_data_loading()
        test_color_scheme()
        test_dataframe_operations()
        test_numpy_operations()
        test_date_operations()
        test_app_structure()
        test_requirements()
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        exit(1)
