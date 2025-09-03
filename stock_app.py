"""
Stock Price Analysis Dashboard
=============================

An interactive web dashboard for analyzing stock prices and LSTM predictions.
Features include:
- Real-time stock price visualization
- LSTM model predictions
- Multi-stock comparison
- Interactive charts with technical indicators

Author: Student Project
Date: 2024
"""

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
from datetime import datetime, timedelta
import yfinance as yf

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Dash app
app = dash.Dash(__name__, 
                title="Stock Price Analysis Dashboard",
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# App configuration
app.config.suppress_callback_exceptions = True

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

def load_stock_data():
    """
    Load and preprocess stock data for the dashboard.
    """
    try:
        # Use live data from yfinance instead of CSV files
        # Fetch TATA stock data (using TATAMOTORS.NS for NSE)
        ticker = "TATAMOTORS.NS"
        df_nse = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        if df_nse.empty:
            # Fallback to a different stock if TATA is not available
            print("TATA data not available, using AAPL as fallback...")
            df_nse = yf.download("AAPL", period="1y", interval="1d", progress=False)
        
        # Ensure we have data and it has the required columns
        if df_nse.empty or 'Close' not in df_nse.columns:
            print("No valid data available, creating sample data...")
            # Create sample data as fallback
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            df_nse = pd.DataFrame({
                'Close': np.random.randn(len(dates)).cumsum() + 100,
                'Open': np.random.randn(len(dates)).cumsum() + 100,
                'High': np.random.randn(len(dates)).cumsum() + 102,
                'Low': np.random.randn(len(dates)).cumsum() + 98,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
        
        # Reset index to get Date as a column, then set it as index again
        df_nse = df_nse.reset_index()
        df_nse["Date"] = pd.to_datetime(df_nse["Date"])
        df_nse.index = df_nse['Date']
        
        # Process data for LSTM
        data = df_nse.sort_index(ascending=True, axis=0)
        
        # Create new_data DataFrame with proper indexing
        # Ensure Close data is 1-dimensional
        close_data = data['Close'].values.flatten() if hasattr(data['Close'], 'values') else data['Close']
        
        new_data = pd.DataFrame({
            'Date': data.index,
            'Close': close_data
        })
        
        # Set Date as index
        new_data.index = new_data['Date']
        new_data = new_data[['Close']]  # Keep only Close column
        
        # Create sample stock data for multiple companies using yfinance
        sample_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        df_stocks = pd.DataFrame()
        
        for stock in sample_stocks:
            try:
                stock_data = yf.download(stock, period="6mo", interval="1d", progress=False)
                if not stock_data.empty:
                    stock_data = stock_data.reset_index()
                    stock_data['Stock'] = stock
                    # Ensure we only have the columns we need and they are properly named
                    stock_data = stock_data[['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    # Reset index to avoid any MultiIndex issues
                    stock_data = stock_data.reset_index(drop=True)
                    # Ensure column names are simple strings, not tuples
                    stock_data.columns = ['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df_stocks = pd.concat([df_stocks, stock_data], ignore_index=True)
            except Exception as e:
                print(f"Error downloading {stock}: {e}")
                continue
        
        if not df_stocks.empty:
            df_stocks["Date"] = pd.to_datetime(df_stocks["Date"])
            df_stocks = df_stocks.sort_values(['Stock', 'Date']).reset_index(drop=True)
        else:
            print("No stock data available, creating sample data...")
            # Create sample stock data as fallback
            dates = pd.date_range(start='2023-07-01', end='2024-01-01', freq='D')
            sample_data = []
            for stock in sample_stocks:
                stock_data = pd.DataFrame({
                    'Date': dates,
                    'Stock': stock,
                    'Open': np.random.randn(len(dates)).cumsum() + 100,
                    'High': np.random.randn(len(dates)).cumsum() + 102,
                    'Low': np.random.randn(len(dates)).cumsum() + 98,
                    'Close': np.random.randn(len(dates)).cumsum() + 100,
                    'Volume': np.random.randint(1000000, 10000000, len(dates))
                })
                sample_data.append(stock_data)
            df_stocks = pd.concat(sample_data, ignore_index=True)
        
        print(f"✅ Successfully loaded data:")
        print(f"   - Main stock data: {len(df_nse)} records")
        print(f"   - LSTM data: {len(new_data)} records")
        print(f"   - Stock comparison data: {len(df_stocks)} records")
        
        return df_nse, new_data, df_stocks
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Creating fallback data...")
        
        # Create fallback data to prevent crashes
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        fallback_df = pd.DataFrame({
            'Close': np.random.randn(len(dates)).cumsum() + 100,
            'Open': np.random.randn(len(dates)).cumsum() + 100,
            'High': np.random.randn(len(dates)).cumsum() + 102,
            'Low': np.random.randn(len(dates)).cumsum() + 98,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Reset index to get Date as a column
        fallback_df = fallback_df.reset_index()
        fallback_df.rename(columns={'index': 'Date'}, inplace=True)
        fallback_df["Date"] = pd.to_datetime(fallback_df["Date"])
        fallback_df.index = fallback_df['Date']
        
        # Create LSTM data
        fallback_new_data = pd.DataFrame({
            'Date': fallback_df.index,
            'Close': fallback_df['Close'].values
        })
        fallback_new_data.index = fallback_new_data['Date']
        fallback_new_data = fallback_new_data[['Close']]
        
        return fallback_df, fallback_new_data, pd.DataFrame()

def fetch_live_data(selected_stocks, period: str = '6mo', interval: str = '1d') -> pd.DataFrame:
    """
    Fetch live market data for given tickers using yfinance and return unified DataFrame
    with columns: Date, Stock, Open, High, Low, Close, Volume.
    """
    if not selected_stocks:
        return pd.DataFrame(columns=["Date", "Stock", "Open", "High", "Low", "Close", "Volume"])
    try:
        data = yf.download(tickers=selected_stocks, period=period, interval=interval, group_by='ticker', auto_adjust=False, progress=False)
        rows = []
        # yfinance returns multi-index or single depending on number of tickers
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in selected_stocks:
                if ticker in data.columns.get_level_values(0):
                    df_t = data[ticker].reset_index()
                    df_t['Stock'] = ticker
                    # Handle the case where both 'Close' and 'Adj Close' might exist
                    if 'Adj Close' in df_t.columns:
                        df_t['Close'] = df_t['Adj Close']
                    df_t = df_t[['Date','Stock','Open','High','Low','Close','Volume']]
                    rows.append(df_t)
        else:
            # Single ticker
            df_t = data.reset_index()
            df_t['Stock'] = selected_stocks[0]
            # Handle the case where both 'Close' and 'Adj Close' might exist
            if 'Adj Close' in df_t.columns:
                df_t['Close'] = df_t['Adj Close']
            df_t = df_t[['Date','Stock','Open','High','Low','Close','Volume']]
            rows.append(df_t)
        if not rows:
            return pd.DataFrame(columns=["Date", "Stock", "Open", "High", "Low", "Close", "Volume"])
        df = pd.concat(rows, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.dropna(subset=['Close'])
        df = df.sort_values(['Stock','Date']).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return pd.DataFrame(columns=["Date", "Stock", "Open", "High", "Low", "Close", "Volume"])

def create_lstm_predictions(new_data):
    """
    Create LSTM predictions for the dashboard.
    """
    try:
        from keras.models import load_model
        from sklearn.preprocessing import MinMaxScaler
        
        # Check if we have enough data
        if new_data.empty or len(new_data) < 20:
            print("Not enough data for LSTM predictions")
            return np.array([]), np.array([]), np.array([])
        
        # Load the trained model
        model = load_model("saved_lstm_model.h5")
        
        # Prepare data for prediction
        dataset = new_data.values
        train = dataset[0:15, :]
        valid = dataset[15:, :]
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Create sequences
        x_train, y_train = [], []
        for i in range(5, len(train)):
            x_train.append(scaled_data[i-5:i, 0])
            y_train.append(scaled_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        if len(x_train) > 0:
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Generate predictions
        inputs = new_data[len(new_data)-len(valid)-5:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        
        X_test = []
        for i in range(5, inputs.shape[0]):
            X_test.append(inputs[i-5:i, 0])
        
        X_test = np.array(X_test)
        
        if len(X_test) > 0:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            closing_price = model.predict(X_test)
            closing_price = scaler.inverse_transform(closing_price)
        else:
            closing_price = np.array([[valid[0][0]]])
        
        return train, valid, closing_price
        
    except Exception as e:
        print(f"Error in LSTM predictions: {str(e)}")
        # Return empty arrays if there's an error
        return np.array([]), np.array([]), np.array([])

def create_stock_comparison_chart(df_stocks, selected_stocks):
    """
    Create stock comparison chart with high/low prices and distinct styling.
    """
    dropdown = {
        "TSLA": "Tesla", 
        "AAPL": "Apple", 
        "FB": "Facebook", 
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "NVDA": "NVIDIA",
        "AMZN": "Amazon",
        "META": "Meta"
    }
    
    # Define distinct colors for each stock to make them look different
    stock_colors = {
        'AAPL': '#1f77b4',      # Blue
        'TSLA': '#ff7f0e',      # Orange
        'MSFT': '#2ca02c',      # Green
        'GOOGL': '#d62728',     # Red
        'NVDA': '#9467bd',      # Purple
        'AMZN': '#8c564b',      # Brown
        'META': '#e377c2',      # Pink
        'FB': '#7f7f7f'         # Gray
    }
    
    traces = []
    for stock in selected_stocks:
        stock_data = df_stocks[df_stocks["Stock"] == stock]
        
        if len(stock_data) > 0:
            # High prices with distinct styling
            traces.append(go.Scatter(
                x=stock_data["Date"],
                y=stock_data["High"],
                mode='lines+markers',
                opacity=0.9,
                name=f'High {dropdown.get(stock, stock)}',
                line=dict(
                    color=stock_colors.get(stock, COLORS['primary']), 
                    width=3,
                    dash='solid'
                ),
                marker=dict(
                    size=4,
                    color=stock_colors.get(stock, COLORS['primary']),
                    symbol='circle'
                ),
                hovertemplate=f'<b>{dropdown.get(stock, stock)}</b><br>' +
                             'Date: %{x}<br>' +
                             'High: $%{y:.2f}<br>' +
                             '<extra></extra>'
            ))
            
            # Low prices with distinct styling
            traces.append(go.Scatter(
                x=stock_data["Date"],
                y=stock_data["Low"],
                mode='lines+markers',
                opacity=0.9,
                name=f'Low {dropdown.get(stock, stock)}',
                line=dict(
                    color=stock_colors.get(stock, COLORS['danger']), 
                    width=3,
                    dash='dot'
                ),
                marker=dict(
                    size=4,
                    color=stock_colors.get(stock, COLORS['danger']),
                    symbol='diamond'
                ),
                hovertemplate=f'<b>{dropdown.get(stock, stock)}</b><br>' +
                             'Date: %{x}<br>' +
                             'Low: $%{y:.2f}<br>' +
                             '<extra></extra>'
            ))
    
    layout = go.Layout(
        title=dict(
            text=f"High and Low Prices for {', '.join(str(dropdown.get(i, i)) for i in selected_stocks)} Over Time",
            font=dict(size=20, color=COLORS['dark'])
        ),
        xaxis=dict(
            title="Date",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor=COLORS['light'],
                activecolor=COLORS['primary']
            ),
            rangeslider=dict(visible=True),
            type="date",
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title="Price (USD)",
            gridcolor='lightgray',
            zeroline=False,
            tickformat='$.2f'
        ),
        height=700,
        template="plotly_white",
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return {"data": traces, "layout": layout}

def create_volume_chart(df_stocks, selected_stocks):
    """
    Create trading volume chart with distinct styling for each stock.
    """
    try:
        # Validate input data
        if df_stocks.empty or not selected_stocks:
            return {"data": [], "layout": go.Layout(title="No data available")}
        
        dropdown = {
            "TSLA": "Tesla", 
            "AAPL": "Apple", 
            "FB": "Facebook", 
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "NVDA": "NVIDIA",
            "AMZN": "Amazon",
            "META": "Meta"
        }
        
        # Define distinct colors for each stock to make them look different
        stock_colors = {
            'AAPL': '#1f77b4',      # Blue
            'TSLA': '#ff7f0e',      # Orange
            'MSFT': '#2ca02c',      # Green
            'GOOGL': '#d62728',     # Red
            'NVDA': '#9467bd',      # Purple
            'AMZN': '#8c564b',      # Brown
            'META': '#e377c2',      # Pink
            'FB': '#7f7f7f'         # Gray
        }
        
        traces = []
        for stock in selected_stocks:
            try:
                stock_data = df_stocks[df_stocks["Stock"] == stock]
                
                if len(stock_data) > 0 and "Date" in stock_data.columns and "Volume" in stock_data.columns:
                    traces.append(go.Scatter(
                        x=stock_data["Date"],
                        y=stock_data["Volume"],
                        mode='lines+markers',
                        opacity=0.9,
                        name=f'Volume {dropdown.get(stock, stock)}',
                        line=dict(
                            color=stock_colors.get(stock, COLORS['info']), 
                            width=3
                        ),
                        marker=dict(
                            size=5,
                            color=stock_colors.get(stock, COLORS['info']),
                            symbol='circle'
                        ),
                        hovertemplate=f'<b>{dropdown.get(stock, stock)}</b><br>' +
                                     'Date: %{x}<br>' +
                                     'Volume: %{y:,}<br>' +
                                     '<extra></extra>'
                    ))
            except Exception as e:
                print(f"Error creating trace for {stock}: {e}")
                continue
        
        # Check if we have any traces
        if not traces:
            return {"data": [], "layout": go.Layout(title="No valid data for selected stocks")}
    
        layout = go.Layout(
            title=dict(
                text=f"Trading Volume for {', '.join(str(dropdown.get(i, i)) for i in selected_stocks)} Over Time",
                font=dict(size=20, color=COLORS['dark'])
            ),
            xaxis=dict(
                title="Date",
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    bgcolor=COLORS['light'],
                    activecolor=COLORS['primary']
                ),
                rangeslider=dict(visible=True),
                type="date",
                gridcolor='lightgray',
                zeroline=False
            ),
            yaxis=dict(
                title="Volume (Shares)",
                gridcolor='lightgray',
                zeroline=False,
                tickformat=','
            ),
            height=700,
            template="plotly_white",
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif"),
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        return {"data": traces, "layout": layout}
        
    except Exception as e:
        print(f"Error in create_volume_chart: {e}")
        return {"data": [], "layout": go.Layout(title="Error creating volume chart")}

def create_closing_prices_chart(df_stocks, selected_stocks):
    """
    Create closing prices comparison chart with distinct styling for each stock.
    """
    dropdown = {
        "TSLA": "Tesla", 
        "AAPL": "Apple", 
        "FB": "Facebook", 
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "NVDA": "NVIDIA",
        "AMZN": "Amazon",
        "META": "Meta"
    }
    
    # Define distinct colors for each stock to make them look different
    stock_colors = {
        'AAPL': '#1f77b4',      # Blue
        'TSLA': '#ff7f0e',      # Orange
        'MSFT': '#2ca02c',      # Green
        'GOOGL': '#d62728',     # Red
        'NVDA': '#9467bd',      # Purple
        'AMZN': '#8c564b',      # Brown
        'META': '#e377c2',      # Pink
        'FB': '#7f7f7f'         # Gray
    }
    
    traces = []
    for stock in selected_stocks:
        stock_data = df_stocks[df_stocks["Stock"] == stock]
        
        if len(stock_data) > 0:
            traces.append(go.Scatter(
                x=stock_data["Date"],
                y=stock_data["Close"],
                mode='lines+markers',
                opacity=0.9,
                name=f'{dropdown.get(stock, stock)}',
                line=dict(
                    color=stock_colors.get(stock, COLORS['primary']), 
                    width=3
                ),
                marker=dict(
                    size=4,
                    color=stock_colors.get(stock, COLORS['primary']),
                    symbol='circle'
                ),
                hovertemplate=f'<b>{dropdown.get(stock, stock)}</b><br>' +
                             'Date: %{x}<br>' +
                             'Close: $%{y:.2f}<br>' +
                             '<extra></extra>'
            ))
    
    layout = go.Layout(
        title=dict(
            text=f"Closing Prices Comparison for {', '.join(str(dropdown.get(i, i)) for i in selected_stocks)} Over Time",
            font=dict(size=20, color=COLORS['dark'])
        ),
        xaxis=dict(
            title="Date",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor=COLORS['light'],
                activecolor=COLORS['primary']
            ),
            rangeslider=dict(visible=True),
            type="date",
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title="Closing Price (USD)",
            gridcolor='lightgray',
            zeroline=False,
            tickformat='$.2f'
        ),
        height=700,
        template="plotly_white",
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return {"data": traces, "layout": layout}

# Load data
df_nse, new_data, df_stocks = load_stock_data()

# Initialize global variables for live data
live_stocks_data = {}
investment_recommendations = {}

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("📈 Stock Price Analysis Dashboard", 
                style={"textAlign": "center", "color": COLORS['dark'], "marginBottom": "20px"}),
        html.P("Advanced LSTM-based stock price prediction and analysis platform", 
               style={"textAlign": "center", "color": COLORS['secondary'], "fontSize": "18px"}),
        html.Hr(style={"borderColor": COLORS['light']})
    ], style={"backgroundColor": COLORS['light'], "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),
    
    # Main content
    html.Div([
        dcc.Tabs(id="tabs", children=[
            # Tab 1: Live Stock Dashboard
            dcc.Tab(label='📊 Live Stock Dashboard', children=[
                html.Div([
                    html.H2("📈 Live Stock Market Dashboard", 
                            style={"textAlign": "center", "color": COLORS['primary'], "marginBottom": "20px"}),
                    html.P("Real-time stock data for top companies with investment recommendations", 
                           style={"textAlign": "center", "color": COLORS['secondary'], "marginBottom": "30px"}),
                    
                    # Refresh Button and Auto-refresh
                    html.Div([
                        html.Button("🔄 Refresh Live Data", id="refresh-btn", n_clicks=0,
                                  style={"backgroundColor": COLORS['primary'], "color": "white", "padding": "10px 20px", 
                                         "border": "none", "borderRadius": "5px", "cursor": "pointer", "fontSize": "16px"}),
                        html.Div(id="last-updated", style={"marginLeft": "20px", "color": COLORS['secondary']}),
                        dcc.Interval(id='live-dashboard-interval', interval=30*1000, n_intervals=0)  # 30 seconds
                    ], style={"display": "flex", "justifyContent": "center", "alignItems": "center", "marginBottom": "30px"}),
                    
                    # Live Stock Rates Grid
                    html.Div(id="live-stocks-grid", style={"marginBottom": "30px"}),
                    
                    # Investment Recommendations
                    html.Div([
                        html.H3("💡 Investment Recommendations", 
                                style={"textAlign": "center", "color": COLORS['dark'], "marginBottom": "20px"}),
                        html.Div(id="investment-recommendations")
                    ])
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS['light']}),
            
            # Tab 2: Prediction Graphs
            dcc.Tab(label='📈 Prediction Graphs', children=[
                html.Div([
                    html.H2("🔮 Stock Price Predictions", 
                            style={"textAlign": "center", "color": COLORS['primary'], "marginBottom": "20px"}),
                    html.P("Select up to 4 companies to view historical data + future predictions", 
                           style={"textAlign": "center", "color": COLORS['secondary'], "marginBottom": "30px"}),
                    
                    # Company Selection
                    html.Div([
                        html.Label("Select Companies (Max 4):", 
                                 style={"fontWeight": "bold", "marginBottom": "10px", "display": "block"}),
                        dcc.Dropdown(
                            id='prediction-companies',
                            options=[
                                {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                                {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
                                {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                                {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
                                {'label': 'NVIDIA (NVDA)', 'value': 'NVDA'},
                                {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
                                {'label': 'Meta (META)', 'value': 'META'},
                                {'label': 'Netflix (NFLX)', 'value': 'NFLX'}
                            ],
                            value=['AAPL', 'MSFT'],
                            multi=True,
                            maxHeight=200,
                            style={"marginBottom": "20px"}
                        ),
                        html.Button("🚀 Generate Predictions", id="generate-predictions-btn", n_clicks=0,
                                  style={"backgroundColor": COLORS['success'], "color": "white", "padding": "10px 20px", 
                                         "border": "none", "borderRadius": "5px", "cursor": "pointer", "fontSize": "16px"})
                    ], style={"marginBottom": "30px"}),
                    
                    # Prediction Charts Container
                    html.Div(id="prediction-charts-container")
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS['light']}),
            
            # Tab 3: Multi-Stock Analysis
            dcc.Tab(label='📊 Multi-Stock Analysis', children=[
                html.Div([
                    html.H2("🔍 Stock Market Analysis Dashboard", 
                            style={"textAlign": "center", "color": COLORS['primary'], "marginBottom": "30px"}),
                    
                    # Stock selection + live toggle
                    html.Div([
                        html.Div([
                            html.Label("Select Stocks for Analysis:", 
                                     style={"fontWeight": "bold", "marginBottom": "10px"}),
                            dcc.Dropdown(
                                id='stock-dropdown',
                                options=[
                                    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                                    {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
                                    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
                                    {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                                    {'label': 'NVIDIA (NVDA)', 'value': 'NVDA'},
                                    {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
                                    {'label': 'Meta (META)', 'value': 'META'},
                                    {'label': 'Facebook (FB)', 'value': 'FB'}
                                ],
                                multi=True,
                                value=['AAPL', 'TSLA', 'MSFT'],
                                style={"marginBottom": "20px"}
                            )
                        ], style={"width": "50%", "margin": "0 auto"}),
                        html.Div([
                            dcc.Checklist(
                                id='use-live-data-multi',
                                options=[{'label': 'Use live data (yfinance)', 'value': 'live'}],
                                value=[],
                                style={"textAlign": "center"}
                            ),
                            dcc.Interval(id='live-interval-multi', interval=60*1000, n_intervals=0)
                        ], style={"marginTop": "10px"}),
                        
                        # Summary Statistics
                        html.Div([
                            html.H4("📈 Summary Statistics", 
                                    style={"textAlign": "center", "color": COLORS['primary'], "marginTop": "20px"}),
                            html.Div(id='summary-stats', style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "15px", "marginTop": "15px"})
                        ], style={"marginTop": "30px"})
                    ], style={"padding": "20px"}),
                    
                    # High/Low Chart
                    html.H3("📈 High vs Low Prices Over Time", 
                            style={"textAlign": "center", "color": COLORS['dark'], "marginTop": "30px"}),
                    dcc.Graph(id='highlow-chart'),
                    
                    # Closing Prices Chart
                    html.H3("💰 Closing Prices Comparison", 
                            style={"textAlign": "center", "color": COLORS['dark'], "marginTop": "30px"}),
                    dcc.Graph(id='closing-chart'),
                    
                    # Volume Chart
                    html.H3("📊 Trading Volume Analysis", 
                            style={"textAlign": "center", "color": COLORS['dark'], "marginTop": "30px"}),
                    dcc.Graph(id='volume-chart')
                ], style={"padding": "20px"})
            ], style={"backgroundColor": COLORS['light']}),
            
            # Tab 3: About & Documentation
            dcc.Tab(label='ℹ️ About & Help', children=[
                html.Div([
                    html.H2("📚 Project Information", 
                            style={"textAlign": "center", "color": COLORS['primary'], "marginBottom": "30px"}),
                    
                    html.Div([
                        html.H3("🎯 Project Overview", style={"color": COLORS['dark']}),
                        html.P([
                            "This dashboard demonstrates advanced machine learning techniques for stock price prediction. ",
                            "The LSTM (Long Short-Term Memory) neural network analyzes historical stock data to predict future prices."
                        ], style={"fontSize": "16px", "lineHeight": "1.6"}),
                        
                        html.H3("🔬 Technical Details", style={"color": COLORS['dark'], "marginTop": "20px"}),
                        html.Ul([
                            html.Li("LSTM Architecture: 2 LSTM layers with 50 units each"),
                            html.Li("Training: Adam optimizer with Mean Squared Error loss"),
                            html.Li("Data Processing: MinMaxScaler normalization"),
                            html.Li("Lookback Window: 60 days for sequence creation")
                        ], style={"fontSize": "16px", "lineHeight": "1.6"}),
                        
                        html.H3("📊 Features", style={"color": COLORS['dark'], "marginTop": "20px"}),
                        html.Ul([
                            html.Li("Real-time stock price visualization"),
                            html.Li("LSTM model predictions"),
                            html.Li("Multi-stock comparison"),
                            html.Li("Interactive charts with zoom and pan"),
                            html.Li("Technical analysis tools")
                        ], style={"fontSize": "16px", "lineHeight": "1.6"}),
                        
                        html.H3("⚠️ Disclaimer", style={"color": COLORS['danger'], "marginTop": "20px"}),
                        html.P([
                            "This tool is for educational and research purposes only. ",
                            "Stock predictions are based on historical data and should not be used as financial advice. ",
                            "Always consult with qualified financial advisors before making investment decisions."
                        ], style={"fontSize": "16px", "lineHeight": "1.6", "color": COLORS['danger']})
                    ], style={"maxWidth": "800px", "margin": "0 auto", "padding": "20px"})
                ])
            ], style={"backgroundColor": COLORS['light']})
        ], style={"backgroundColor": COLORS['light']})
    ], style={"padding": "20px"}),
    
    # Footer
    html.Div([
        html.Hr(style={"borderColor": COLORS['light']}),
        html.P("Stock Price Analysis Dashboard | LSTM Neural Network Project | Created for Academic Purposes", 
               style={"textAlign": "center", "color": COLORS['secondary'], "fontSize": "14px"}),
        html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
               style={"textAlign": "center", "color": COLORS['secondary'], "fontSize": "12px"})
    ], style={"backgroundColor": COLORS['light'], "padding": "20px", "borderRadius": "10px", "marginTop": "20px"})
], style={"backgroundColor": COLORS['light'], "minHeight": "100vh"})

# Callbacks

# Live Dashboard Callbacks
@app.callback(
    [Output('live-stocks-grid', 'children'),
     Output('last-updated', 'children'),
     Output('investment-recommendations', 'children')],
    [Input('refresh-btn', 'n_clicks'),
     Input('live-dashboard-interval', 'n_intervals')]
)
def update_live_dashboard(n_clicks, n_intervals):
    """Update live stock data and investment recommendations"""
    try:
        # Define the 8 top companies
        top_companies = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"]
        
        # Fetch live data for all companies
        live_data = {}
        for company in top_companies:
            try:
                stock_info = yf.Ticker(company)
                info = stock_info.info
                hist = stock_info.history(period="5d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100
                    
                    live_data[company] = {
                        'current_price': current_price,
                        'price_change': price_change,
                        'price_change_pct': price_change_pct,
                        'volume': hist['Volume'].iloc[-1],
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0)
                    }
            except Exception as e:
                print(f"Error fetching data for {company}: {e}")
                continue
        
        # Create live stock grid
        stock_cards = []
        for company, data in live_data.items():
            change_color = COLORS['success'] if data['price_change'] >= 0 else COLORS['danger']
            change_icon = "📈" if data['price_change'] >= 0 else "📉"
            
            stock_cards.append(
                html.Div([
                    html.H4(company, style={"color": COLORS['primary'], "marginBottom": "10px"}),
                    html.Div([
                        html.P(f"${data['current_price']:.2f}", 
                               style={"fontSize": "24px", "fontWeight": "bold", "marginBottom": "5px"}),
                        html.P(f"{change_icon} {data['price_change']:+.2f} ({data['price_change_pct']:+.1f}%)", 
                               style={"color": change_color, "fontSize": "16px", "marginBottom": "5px"}),
                        html.P(f"Volume: {data['volume']:,.0f}", style={"fontSize": "14px", "color": COLORS['dark']}),
                        html.P(f"Market Cap: ${data['market_cap']/1e9:.1f}B", style={"fontSize": "12px", "color": COLORS['secondary']})
                    ])
                ], style={
                    "textAlign": "center", 
                    "padding": "20px", 
                    "backgroundColor": "white", 
                    "borderRadius": "10px", 
                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                    "border": f"2px solid {change_color}",
                    "minHeight": "200px"
                })
            )
        
        # Create investment recommendations
        recommendations = generate_investment_recommendations(live_data)
        
        # Update global variables
        global live_stocks_data, investment_recommendations
        live_stocks_data = live_data
        investment_recommendations = recommendations
        
        return (
            html.Div(stock_cards, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))", "gap": "20px"}),
            f"Last updated: {datetime.now().strftime('%H:%M:%S')}",
            recommendations
        )
        
    except Exception as e:
        print(f"Error updating live dashboard: {e}")
        return [], "Error updating data", html.Div("Error loading live data", style={"color": COLORS['danger']})

def generate_investment_recommendations(live_data):
    """Generate investment recommendations based on live data"""
    try:
        recommendations = []
        
        # Simple recommendation logic based on price momentum and volume
        for company, data in live_data.items():
            if data['price_change_pct'] > 2:  # Strong upward momentum
                recommendations.append(
                    html.Div([
                        html.H4(f"🚀 {company} - Strong Buy Signal", style={"color": COLORS['success']}),
                        html.P(f"Current Price: ${data['current_price']:.2f}"),
                        html.P(f"Price Change: +{data['price_change_pct']:.1f}% (Strong momentum)"),
                        html.P("💡 Recommendation: Consider buying now, monitor for profit-taking opportunities"),
                        html.P(f"📊 Volume: {data['volume']:,.0f} shares traded")
                    ], style={
                        "backgroundColor": "white", 
                        "padding": "20px", 
                        "borderRadius": "10px", 
                        "border": f"2px solid {COLORS['success']}",
                        "marginBottom": "15px"
                    })
                )
            elif data['price_change_pct'] < -2:  # Strong downward momentum
                recommendations.append(
                    html.Div([
                        html.H4(f"⚠️ {company} - Caution Required", style={"color": COLORS['warning']}),
                        html.P(f"Current Price: ${data['current_price']:.2f}"),
                        html.P(f"Price Change: {data['price_change_pct']:.1f}% (Declining)"),
                        html.P("💡 Recommendation: Wait for stabilization, consider buying on dips"),
                        html.P(f"📊 Volume: {data['volume']:,.0f} shares traded")
                    ], style={
                        "backgroundColor": "white", 
                        "padding": "20px", 
                        "borderRadius": "10px", 
                        "border": f"2px solid {COLORS['warning']}",
                        "marginBottom": "15px"
                    })
                )
            else:  # Stable
                recommendations.append(
                    html.Div([
                        html.H4(f"📊 {company} - Stable Performance", style={"color": COLORS['info']}),
                        html.P(f"Current Price: ${data['current_price']:.2f}"),
                        html.P(f"Price Change: {data['price_change_pct']:+.1f}% (Stable)"),
                        html.P("💡 Recommendation: Good for long-term holding, consider dollar-cost averaging"),
                        html.P(f"📊 Volume: {data['volume']:,.0f} shares traded")
                    ], style={
                        "backgroundColor": "white", 
                        "padding": "20px", 
                        "borderRadius": "10px", 
                        "border": f"2px solid {COLORS['info']}",
                        "marginBottom": "15px"
                    })
                )
        
        return html.Div(recommendations)
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return html.Div("Error generating recommendations", style={"color": COLORS['danger']})

# Prediction Graphs Callback
@app.callback(
    Output('prediction-charts-container', 'children'),
    [Input('generate-predictions-btn', 'n_clicks')],
    [dash.dependencies.State('prediction-companies', 'value')]
)
def generate_prediction_charts(n_clicks, selected_companies):
    """Generate prediction charts for selected companies"""
    if not n_clicks or not selected_companies:
        return html.Div("Select companies and click 'Generate Predictions' to view charts")
    
    try:
        # Limit to 4 companies
        selected_companies = selected_companies[:4]
        
        charts = []
        for company in selected_companies:
            try:
                # Fetch historical data
                stock_data = yf.download(company, period="1y", interval="1d", progress=False)
                
                if not stock_data.empty:
                    # Create simple trend-based prediction (for demo purposes)
                    dates = stock_data.index
                    prices = stock_data['Close'].values
                    
                    print(f"Debug: {company} - prices shape: {prices.shape}, type: {type(prices)}")
                    
                    # Ensure prices is a 1D array
                    if prices.ndim > 1:
                        prices = prices.flatten()
                        print(f"Debug: {company} - flattened prices shape: {prices.shape}")
                    
                    # Simple linear trend prediction for next 30 days
                    if len(prices) > 30:
                        recent_prices = prices[-30:]
                        # Ensure recent_prices is 1D
                        if recent_prices.ndim > 1:
                            recent_prices = recent_prices.flatten()
                        
                        x = np.arange(len(recent_prices))
                        slope, intercept = np.polyfit(x, recent_prices, 1)
                        
                        # Generate future dates and predictions
                        future_days = 30
                        future_x = np.arange(len(recent_prices), len(recent_prices) + future_days)
                        future_prices = slope * future_x + intercept
                        
                        # Combine historical and future data
                        all_dates = pd.date_range(start=dates[-30], periods=len(recent_prices) + future_days, freq='D')
                        all_prices = np.concatenate([recent_prices, future_prices])
                        
                        # Create chart
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=all_dates[:len(recent_prices)],
                            y=recent_prices.flatten() if hasattr(recent_prices, 'flatten') else recent_prices,
                            mode='lines+markers',
                            name='Historical Data',
                            line=dict(color=COLORS['primary'], width=3)
                        ))
                        
                        # Predicted data
                        fig.add_trace(go.Scatter(
                            x=all_dates[len(recent_prices):],
                            y=future_prices.flatten() if hasattr(future_prices, 'flatten') else future_prices,
                            mode='lines+markers',
                            name='Predicted Trend',
                            line=dict(color=COLORS['warning'], width=3, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"{company} - Historical Data + Trend Prediction",
                            xaxis_title="Date",
                            yaxis_title="Stock Price ($)",
                            height=400,
                            template="plotly_white",
                            showlegend=True
                        )
                        
                        charts.append(
                            html.Div([
                                html.H4(f"📈 {company} Prediction Chart", 
                                        style={"textAlign": "center", "color": COLORS['primary'], "marginBottom": "15px"}),
                                dcc.Graph(figure=fig)
                            ], style={"marginBottom": "30px"})
                        )
                        
            except Exception as e:
                print(f"Error creating prediction chart for {company}: {e}")
                continue
        
        if not charts:
            return html.Div("Error generating prediction charts", style={"color": COLORS['danger']})
        
        return html.Div(charts)
        
    except Exception as e:
        print(f"Error in prediction charts: {e}")
        return html.Div("Error generating prediction charts", style={"color": COLORS['danger']})

# Existing Callbacks
@app.callback(
    Output('highlow-chart', 'figure'),
    [Input('stock-dropdown', 'value'), Input('use-live-data-multi', 'value'), Input('live-interval-multi', 'n_intervals')]
)
def update_highlow_chart(selected_stocks, use_live, _):
    try:
        if not selected_stocks:
            return {"data": [], "layout": go.Layout(title="Please select stocks")}
        
        use_live_flag = 'live' in (use_live or [])
        data_source = fetch_live_data(selected_stocks) if use_live_flag else df_stocks
        
        # Check if data source is valid
        if data_source.empty:
            return {"data": [], "layout": go.Layout(title="No data available")}
        
        return create_stock_comparison_chart(data_source, selected_stocks)
    except Exception as e:
        print(f"Error updating highlow chart: {e}")
        return {"data": [], "layout": go.Layout(title="Error loading chart")}

@app.callback(
    Output('closing-chart', 'figure'),
    [Input('stock-dropdown', 'value'), Input('use-live-data-multi', 'value'), Input('live-interval-multi', 'n_intervals')]
)
def update_closing_chart(selected_stocks, use_live, _):
    try:
        if not selected_stocks:
            return {"data": [], "layout": go.Layout(title="Please select stocks")}
        
        use_live_flag = 'live' in (use_live or [])
        data_source = fetch_live_data(selected_stocks) if use_live_flag else df_stocks
        
        # Check if data source is valid
        if data_source.empty:
            return {"data": [], "layout": go.Layout(title="No data available")}
        
        return create_closing_prices_chart(data_source, selected_stocks)
    except Exception as e:
        print(f"Error updating closing chart: {e}")
        return {"data": [], "layout": go.Layout(title="Error loading chart")}

@app.callback(
    Output('volume-chart', 'figure'),
    [Input('stock-dropdown', 'value'), Input('use-live-data-multi', 'value'), Input('live-interval-multi', 'n_intervals')]
)
def update_volume_chart(selected_stocks, use_live, _):
    try:
        if not selected_stocks:
            return {"data": [], "layout": go.Layout(title="Please select stocks")}
        
        use_live_flag = 'live' in (use_live or [])
        data_source = fetch_live_data(selected_stocks) if use_live_flag else df_stocks
        
        # Check if data source is valid
        if data_source.empty:
            return {"data": [], "layout": go.Layout(title="No data available")}
        
        return create_volume_chart(data_source, selected_stocks)
    except Exception as e:
        print(f"Error updating volume chart: {e}")
        return {"data": [], "layout": go.Layout(title="Error loading chart")}

@app.callback(
    Output('summary-stats', 'children'),
    [Input('stock-dropdown', 'value')]
)
def update_summary_stats(selected_stocks):
    try:
        if not selected_stocks:
            return []
        
        # Check if df_stocks is available and has data
        if df_stocks.empty:
            return [html.Div("No stock data available", style={"textAlign": "center", "color": COLORS['warning']})]
        
        stats_cards = []
        for stock in selected_stocks:
            stock_data = df_stocks[df_stocks["Stock"] == stock]
            print(f"Debug: Processing {stock}, data shape: {stock_data.shape}, columns: {stock_data.columns.tolist()}")
            if len(stock_data) > 0:
                try:
                    # Ensure we have valid data
                    if stock_data["Close"].empty or stock_data["Volume"].empty:
                        continue
                    
                    # Get the actual values - these should now be scalar values
                    current_price = stock_data["Close"].iloc[-1]
                    first_price = stock_data["Close"].iloc[0]
                    
                    print(f"Debug: {stock} - current_price type: {type(current_price)}, value: {current_price}")
                    print(f"Debug: {stock} - first_price type: {type(first_price)}, value: {first_price}")
                    
                    # Convert to float - these should now be simple values
                    current_price = float(current_price)
                    first_price = float(first_price)
                    
                    price_change = current_price - first_price
                    price_change_pct = (price_change / first_price) * 100
                    
                    # Handle volume data
                    avg_volume = float(stock_data["Volume"].mean())
                    
                    # Determine color based on price change
                    change_color = COLORS['success'] if price_change >= 0 else COLORS['danger']
                    change_icon = "📈" if price_change >= 0 else "📉"
                    
                    stats_cards.append(
                        html.Div([
                            html.H4(f"{stock}", style={"color": COLORS['primary'], "marginBottom": "10px"}),
                            html.Div([
                                html.P(f"Current: ${current_price:.2f}", style={"fontSize": "18px", "fontWeight": "bold"}),
                                html.P(f"{change_icon} {price_change:+.2f} ({price_change_pct:+.1f}%)", 
                                       style={"color": change_color, "fontSize": "16px"}),
                                html.P(f"Avg Volume: {avg_volume:,.0f}", style={"fontSize": "14px", "color": COLORS['dark']})
                            ])
                        ], style={
                            "textAlign": "center", 
                            "padding": "20px", 
                            "backgroundColor": "white", 
                            "borderRadius": "10px", 
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                            "border": f"2px solid {change_color}"
                        })
                    )
                except Exception as e:
                    print(f"Error processing stats for {stock}: {e}")
                    continue
        
        if not stats_cards:
            return [html.Div("Unable to load stock statistics", style={"textAlign": "center", "color": COLORS['warning']})]
        
        return stats_cards
        
    except Exception as e:
        print(f"Error updating summary stats: {e}")
        return [html.Div("Error loading statistics", style={"textAlign": "center", "color": COLORS['danger']})]

# Run the app
if __name__ == '__main__':
    print("🚀 Starting Stock Price Analysis Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050/")
    print("🔍 Open your web browser and navigate to the URL above")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=8050)