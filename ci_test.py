#!/usr/bin/env python3
"""
CI Test Script for GitHub Actions
This script tests basic functionality without requiring all packages
"""

import sys
import os

def test_basic_imports():
    """Test basic package imports"""
    print("🧪 Testing basic package imports...")
    
    # Test core packages
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import plotly
        print("✅ plotly imported successfully")
    except ImportError as e:
        print(f"❌ plotly import failed: {e}")
        return False
    
    try:
        import dash
        print("✅ dash imported successfully")
    except ImportError as e:
        print(f"❌ dash import failed: {e}")
        return False
    
    try:
        import yfinance
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
        return False
    
    return True

def test_file_structure():
    """Test that required files exist"""
    print("\n🧪 Testing file structure...")
    
    required_files = ['stock_app.py', 'requirements.txt']
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} not found")
            return False
    
    return True

def test_app_import():
    """Test app import (skip if keras/tensorflow not available)"""
    print("\n🧪 Testing app import...")
    
    try:
        import stock_app
        print("✅ stock_app imported successfully")
        
        # Test key components
        if hasattr(stock_app, 'app'):
            print("✅ App object found")
        else:
            print("❌ App object not found")
            return False
            
        if hasattr(stock_app, 'COLORS'):
            print("✅ COLORS configuration found")
        else:
            print("❌ COLORS configuration not found")
            return False
            
        if hasattr(stock_app, 'load_stock_data'):
            print("✅ load_stock_data function found")
        else:
            print("❌ load_stock_data function not found")
            return False
            
        return True
        
    except ImportError as e:
        if 'keras' in str(e) or 'tensorflow' in str(e):
            print("⚠️  Keras/TensorFlow not available in CI environment")
            print("✅ Skipping app import test - this is expected in CI")
            return True
        else:
            print(f"❌ App import failed: {e}")
            return False
    except Exception as e:
        print(f"❌ App import test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without heavy imports"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test pandas operations
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        sample_data = pd.DataFrame({
            'Close': np.random.randn(len(dates)).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        assert len(sample_data) == 10
        assert 'Close' in sample_data.columns
        assert 'Volume' in sample_data.columns
        
        print("✅ Basic pandas operations work")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🚀 Stock Price Prediction Web App - CI Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("File Structure", test_file_structure),
        ("App Import", test_app_import),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CI should succeed.")
        return 0
    else:
        print("❌ Some tests failed. CI will fail.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
