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
        # Load NSE-TATA data
        df_nse = pd.read_csv("./NSE-TATA.csv")
        df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
        df_nse.index = df_nse['Date']
        
        # Process data for LSTM
        data = df_nse.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])
        
        for i in range(0, len(data)):
            new_data.loc[i, "Date"] = data['Date'].iloc[i]
            new_data.loc[i, "Close"] = data["Close"].iloc[i]
        
        new_data.index = new_data.Date
        new_data.drop("Date", axis=1, inplace=True)
        
        # Load stock comparison data (combine multiple datasets)
        df_stocks_old = pd.read_csv("./stock_data.csv")
        df_stocks_old["Date"] = pd.to_datetime(df_stocks_old["Date"])
        
        # Load recent 2024 data
        df_stocks_recent = pd.read_csv("./recent_stocks_2024.csv")
        df_stocks_recent["Date"] = pd.to_datetime(df_stocks_recent["Date"])
        
        # Combine datasets
        df_stocks = pd.concat([df_stocks_old, df_stocks_recent], ignore_index=True)
        
        return df_nse, new_data, df_stocks
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        # Return empty dataframes if loading fails
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def create_lstm_predictions(new_data):
    """
    Create LSTM predictions for the dashboard.
    """
    try:
        from keras.models import load_model
        from sklearn.preprocessing import MinMaxScaler
        
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
        # Return fallback data
        dataset = new_data.values
        train = dataset[0:15, :]
        valid = dataset[15:, :]
        closing_price = np.array([[valid[0][0]]])
        return train, valid, closing_price

def create_stock_comparison_chart(df_stocks, selected_stocks):
    """
    Create stock comparison chart with high/low prices.
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
    
    traces = []
    for stock in selected_stocks:
        stock_data = df_stocks[df_stocks["Stock"] == stock]
        
        # High prices
        traces.append(go.Scatter(
            x=stock_data["Date"],
            y=stock_data["High"],
            mode='lines',
            opacity=0.8,
            name=f'High {dropdown.get(stock, stock)}',
            line=dict(color=COLORS['primary'], width=2)
        ))
        
        # Low prices
        traces.append(go.Scatter(
            x=stock_data["Date"],
            y=stock_data["Low"],
            mode='lines',
            opacity=0.8,
            name=f'Low {dropdown.get(stock, stock)}',
            line=dict(color=COLORS['danger'], width=2)
        ))
    
    layout = go.Layout(
        title=f"High and Low Prices for {', '.join(str(dropdown.get(i, i)) for i in selected_stocks)} Over Time",
        xaxis=dict(
            title="Date",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(title="Price (USD)"),
        height=600,
        template="plotly_white",
        showlegend=True,
        hovermode='x unified'
    )
    
    return {"data": traces, "layout": layout}

def create_volume_chart(df_stocks, selected_stocks):
    """
    Create trading volume chart.
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
    
    traces = []
    for stock in selected_stocks:
        stock_data = df_stocks[df_stocks["Stock"] == stock]
        
        traces.append(go.Scatter(
            x=stock_data["Date"],
            y=stock_data["Volume"],
            mode='lines+markers',
            opacity=0.8,
            name=f'Volume {dropdown.get(stock, stock)}',
            line=dict(color=COLORS['info'], width=2),
            marker=dict(size=6)
        ))
    
    layout = go.Layout(
        title=f"Trading Volume for {', '.join(str(dropdown.get(i, i)) for i in selected_stocks)} Over Time",
        xaxis=dict(
            title="Date",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(title="Volume"),
        height=600,
        template="plotly_white",
        showlegend=True,
        hovermode='x unified'
    )
    
    return {"data": traces, "layout": layout}

# Load data
df_nse, new_data, df_stocks = load_stock_data()
train, valid, closing_price = create_lstm_predictions(new_data)

# Prepare data for predictions
if len(valid) > 0 and len(closing_price) > 0:
    train_df = pd.DataFrame(train, columns=['Close'])
    valid_df = pd.DataFrame(valid, columns=['Close'])
    valid_df['Predictions'] = closing_price.flatten()
else:
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()

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
            # Tab 1: LSTM Predictions
            dcc.Tab(label='🤖 LSTM Stock Predictions', children=[
                html.Div([
                    html.Div([
                        html.H2("📊 Actual vs Predicted Stock Prices", 
                                style={"textAlign": "center", "color": COLORS['primary']}),
                        html.P("LSTM neural network predictions for NSE-TATA stock", 
                               style={"textAlign": "center", "color": COLORS['secondary']}),
                        
                        # Metrics cards
                        html.Div([
                            html.Div([
                                html.H4("Training Data", style={"color": COLORS['primary']}),
                                html.H3(f"{len(train_df)} samples", style={"color": COLORS['dark']})
                            ], style={"textAlign": "center", "padding": "20px", "backgroundColor": "white", 
                                    "borderRadius": "10px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                            
                            html.Div([
                                html.H4("Validation Data", style={"color": COLORS['success']}),
                                html.H3(f"{len(valid_df)} samples", style={"color": COLORS['dark']})
                            ], style={"textAlign": "center", "padding": "20px", "backgroundColor": "white", 
                                    "borderRadius": "10px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
                            
                            html.Div([
                                html.H4("Predictions", style={"color": COLORS['warning']}),
                                html.H3(f"{len(closing_price)} generated", style={"color": COLORS['dark']})
                            ], style={"textAlign": "center", "padding": "20px", "backgroundColor": "white", 
                                    "borderRadius": "10px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})
                        ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "20px", "marginBottom": "30px"}),
                        
                        # Actual vs Predicted Chart
                        dcc.Graph(
                            id="prediction-chart",
                            figure={
                                "data": [
                                    go.Scatter(
                                        x=list(range(len(train_df))),
                                        y=train_df['Close'] if len(train_df) > 0 else [],
                                        mode='lines',
                                        name='Training Data',
                                        line=dict(color=COLORS['primary'], width=3)
                                    ),
                                    go.Scatter(
                                        x=list(range(len(train_df), len(train_df) + len(valid_df))),
                                        y=valid_df['Close'] if len(valid_df) > 0 else [],
                                        mode='lines',
                                        name='Actual Prices',
                                        line=dict(color=COLORS['success'], width=3)
                                    ),
                                    go.Scatter(
                                        x=list(range(len(train_df), len(train_df) + len(valid_df))),
                                        y=valid_df['Predictions'] if len(valid_df) > 0 else [],
                                        mode='lines',
                                        name='LSTM Predictions',
                                        line=dict(color=COLORS['warning'], width=3, dash='dash')
                                    )
                                ],
                                "layout": go.Layout(
                                    title="LSTM Model: Training Data, Actual Prices, and Predictions",
                                    xaxis=dict(title="Time Period"),
                                    yaxis=dict(title="Stock Price"),
                                    template="plotly_white",
                                    height=500,
                                    showlegend=True,
                                    hovermode='x unified'
                                )
                            }
                        )
                    ], style={"padding": "20px"})
                ])
            ], style={"backgroundColor": COLORS['light']}),
            
            # Tab 2: Multi-Stock Analysis
            dcc.Tab(label='📊 Multi-Stock Analysis', children=[
                html.Div([
                    html.H2("🔍 Stock Market Analysis Dashboard", 
                            style={"textAlign": "center", "color": COLORS['primary'], "marginBottom": "30px"}),
                    
                    # Stock selection
                    html.Div([
                        html.Div([
                            html.Label("Select Stocks for High/Low Analysis:", 
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
                                value=['AAPL', 'TSLA'],
                                style={"marginBottom": "20px"}
                            )
                        ], style={"width": "50%", "margin": "0 auto"}),
                        
                        # High/Low Chart
                        html.H3("📈 High vs Low Prices Over Time", 
                                style={"textAlign": "center", "color": COLORS['dark'], "marginTop": "30px"}),
                        dcc.Graph(id='highlow-chart'),
                        
                        # Volume Chart
                        html.H3("📊 Trading Volume Analysis", 
                                style={"textAlign": "center", "color": COLORS['dark'], "marginTop": "30px"}),
                        dcc.Graph(id='volume-chart')
                    ], style={"padding": "20px"})
                ])
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
@app.callback(
    Output('highlow-chart', 'figure'),
    [Input('stock-dropdown', 'value')]
)
def update_highlow_chart(selected_stocks):
    if not selected_stocks:
        return {"data": [], "layout": go.Layout(title="Please select stocks")}
    return create_stock_comparison_chart(df_stocks, selected_stocks)

@app.callback(
    Output('volume-chart', 'figure'),
    [Input('stock-dropdown', 'value')]
)
def update_volume_chart(selected_stocks):
    if not selected_stocks:
        return {"data": [], "layout": go.Layout(title="Please select stocks")}
    return create_volume_chart(df_stocks, selected_stocks)

# Run the app
if __name__ == '__main__':
    print("🚀 Starting Stock Price Analysis Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050/")
    print("🔍 Open your web browser and navigate to the URL above")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=8050)