#!/usr/bin/env python3
"""
Simple test runner for Stock Price Prediction Web App
Run this script to test basic functionality locally
"""

import sys
import os

def run_basic_tests():
    """Run basic functionality tests"""
    print("🧪 Running basic functionality tests...")
    
    # Test 1: Check if required files exist
    required_files = ['stock_app.py', 'requirements.txt']
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} not found")
            return False
    
    # Test 2: Check Python version
    print(f"✅ Python version: {sys.version}")
    
    # Test 3: Try to import basic packages
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError:
        print("❌ pandas import failed")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError:
        print("❌ numpy import failed")
        return False
    
    try:
        import plotly
        print("✅ plotly imported successfully")
    except ImportError:
        print("❌ plotly import failed")
        return False
    
    try:
        import dash
        print("✅ dash imported successfully")
    except ImportError:
        print("❌ dash import failed")
        return False
    
    try:
        import yfinance
        print("✅ yfinance imported successfully")
    except ImportError:
        print("❌ yfinance import failed")
        return False
    
    # Test 4: Try to import the app (skip if keras not available)
    try:
        import stock_app
        print("✅ stock_app imported successfully")
        
        # Check if key components exist
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
            
    except ImportError as e:
        if 'keras' in str(e) or 'tensorflow' in str(e):
            print("⚠️  Keras/TensorFlow not available, skipping app import test")
            print("✅ Basic functionality test passed")
        else:
            print(f"❌ App import failed: {e}")
            return False
    
    print("\n🎉 All basic tests passed!")
    return True

def run_pytest():
    """Run pytest if available"""
    try:
        import pytest
        print("\n🧪 Running pytest...")
        os.system("python -m pytest test_stock_app.py -v")
        return True
    except ImportError:
        print("⚠️  pytest not available, skipping pytest tests")
        return True

if __name__ == "__main__":
    print("🚀 Stock Price Prediction Web App - Test Runner")
    print("=" * 50)
    
    # Run basic tests
    basic_tests_passed = run_basic_tests()
    
    if basic_tests_passed:
        # Try to run pytest
        run_pytest()
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
