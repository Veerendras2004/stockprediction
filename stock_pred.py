"""
Stock Price Prediction using LSTM Neural Network
==============================================

This script implements a Long Short-Term Memory (LSTM) neural network
to predict stock prices based on historical data.

Features:
- Configurable LSTM architecture
- Data preprocessing and normalization
- Model training and evaluation
- Prediction generation
- Model persistence

Author: Student Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'lookback_days': 60,        # Number of days to look back for prediction
    'lstm_units': 50,           # Number of LSTM units per layer
    'epochs': 1,                # Number of training epochs
    'batch_size': 1,            # Batch size for training
    'train_split': 0.8,         # Training data split ratio
    'random_state': 42,         # Random seed for reproducibility
    'model_save_path': 'saved_lstm_model.h5'
}

def load_and_preprocess_data(file_path):
    """
    Load and preprocess stock data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (processed_dataframe, scaler, scaled_data)
    """
    try:
        # Load data
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Basic data validation
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Convert date and set index
        df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
        df.index = df['Date']
        
        # Sort by date
        df = df.sort_index(ascending=True, axis=0)
        
        # Create new dataset with only Date and Close
        new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
        
        # Fill data using proper pandas methods
        for i in range(0, len(df)):
            new_dataset.loc[i, "Date"] = df['Date'].iloc[i]
            new_dataset.loc[i, "Close"] = df["Close"].iloc[i]
        
        new_dataset.index = new_dataset.Date
        new_dataset.drop("Date", axis=1, inplace=True)
        
        # Convert to numpy array
        final_dataset = new_dataset.values
        
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(final_dataset)
        
        print("Data preprocessing completed successfully.")
        return new_dataset, scaler, scaled_data
        
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise

def create_sequences(data, lookback_days):
    """
    Create sequences for LSTM training.
    
    Args:
        data (array): Scaled data
        lookback_days (int): Number of days to look back
        
    Returns:
        tuple: (X, y) sequences
    """
    X, y = [], []
    
    for i in range(lookback_days, len(data)):
        X.append(data[i-lookback_days:i, 0])
        y.append(data[i, 0])
    
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, lstm_units):
    """
    Build and compile LSTM model.
    
    Args:
        input_shape (tuple): Input shape for the model
        lstm_units (int): Number of LSTM units
        
    Returns:
        keras.Model: Compiled LSTM model
    """
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        
        model = Sequential([
            LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),  # Add dropout for regularization
            LSTM(units=lstm_units),
            Dropout(0.2),  # Add dropout for regularization
            Dense(1)
        ])
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        print("LSTM model built successfully.")
        print(f"Model summary:")
        model.summary()
        
        return model
        
    except Exception as e:
        print(f"Error building model: {str(e)}")
        raise

def train_model(model, X_train, y_train, epochs, batch_size):
    """
    Train the LSTM model.
    
    Args:
        model: LSTM model to train
        X_train: Training features
        y_train: Training labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        keras.History: Training history
    """
    try:
        print(f"Starting model training...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            validation_split=0.1
        )
        
        print("Model training completed successfully.")
        return history
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def generate_predictions(model, data, scaler, lookback_days):
    """
    Generate predictions using trained model.
    
    Args:
        model: Trained LSTM model
        data: Input data for prediction
        scaler: Fitted scaler for inverse transformation
        lookback_days: Number of lookback days
        
    Returns:
        array: Predicted prices
    """
    try:
        # Prepare input data
        inputs = data[len(data)-lookback_days:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)
        
        # Create test sequences
        X_test = []
        for i in range(lookback_days, inputs.shape[0]):
            X_test.append(inputs[i-lookback_days:i, 0])
        
        if len(X_test) > 0:
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # Generate predictions
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            
            print(f"Generated {len(predictions)} predictions.")
            return predictions
        else:
            print("Warning: No test sequences created. Using fallback prediction.")
            return np.array([[data.iloc[-1]['Close']]])
            
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        raise

def plot_results(train_data, valid_data, predictions, title="Stock Price Prediction"):
    """
    Plot training data, validation data, and predictions.
    
    Args:
        train_data: Training dataset
        valid_data: Validation dataset
        predictions: Model predictions
        title: Plot title
    """
    try:
        plt.figure(figsize=(16, 8))
        plt.plot(train_data.index, train_data["Close"], label='Training Data', color='blue')
        plt.plot(valid_data.index, valid_data["Close"], label='Actual Prices', color='green')
        plt.plot(valid_data.index, predictions, label='Predicted Prices', color='red')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Stock Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"stock_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error plotting results: {str(e)}")

def main():
    """
    Main function to run the complete stock prediction pipeline.
    """
    try:
        print("=" * 60)
        print("STOCK PRICE PREDICTION USING LSTM NEURAL NETWORK")
        print("=" * 60)
        print(f"Configuration: {CONFIG}")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        new_dataset, scaler, scaled_data = load_and_preprocess_data("NSE-TATA.csv")
        
        # Step 2: Split data into training and validation sets
        train_size = int(len(new_dataset) * CONFIG['train_split'])
        train_data = new_dataset[:train_size]
        valid_data = new_dataset[train_size:]
        
        print(f"Training data: {len(train_data)} samples")
        print(f"Validation data: {len(valid_data)} samples")
        
        # Step 3: Create sequences for LSTM
        X_train, y_train = create_sequences(scaled_data[:train_size], CONFIG['lookback_days'])
        
        if len(X_train) == 0:
            print("Warning: No training sequences created. Adjusting lookback days...")
            CONFIG['lookback_days'] = min(5, len(scaled_data[:train_size]) // 2)
            X_train, y_train = create_sequences(scaled_data[:train_size], CONFIG['lookback_days'])
        
        print(f"Training sequences created: {X_train.shape}")
        
        # Step 4: Build LSTM model
        input_shape = (X_train.shape[1], 1)
        lstm_model = build_lstm_model(input_shape, CONFIG['lstm_units'])
        
        # Step 5: Train the model
        history = train_model(lstm_model, X_train, y_train, CONFIG['epochs'], CONFIG['batch_size'])
        
        # Step 6: Save the model
        lstm_model.save(CONFIG['model_save_path'])
        print(f"Model saved successfully to: {CONFIG['model_save_path']}")
        
        # Step 7: Generate predictions
        predictions = generate_predictions(lstm_model, new_dataset, scaler, CONFIG['lookback_days'])
        
        # Step 8: Plot results
        plot_results(train_data, valid_data, predictions)
        
        # Step 9: Calculate basic metrics
        if len(predictions) > 0 and len(valid_data) > 0:
            actual_prices = valid_data['Close'].values[:len(predictions)]
            mse = np.mean((predictions.flatten() - actual_prices) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions.flatten() - actual_prices))
            
            print("\n" + "=" * 40)
            print("PREDICTION METRICS")
            print("=" * 40)
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print("=" * 40)
        
        print("\nStock prediction pipeline completed successfully!")
        print("You can now run 'python stock_app.py' to launch the web dashboard.")
        
    except Exception as e:
        print(f"\nError in main pipeline: {str(e)}")
        print("Please check your data and configuration.")
        raise

if __name__ == "__main__":
    main()