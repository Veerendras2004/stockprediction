import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

print("🚀 Training LSTM Model...")

# Load live data from yfinance instead of CSV
import yfinance as yf

print("📊 Fetching live stock data...")
# Try to get TATA stock data (using TATAMOTORS.NS for NSE)
ticker = "TATAMOTORS.NS"
df = yf.download(ticker, period="2y", interval="1d", progress=False)

if df.empty:
    # Fallback to a different stock if TATA is not available
    print("TATA data not available, using AAPL as fallback...")
    df = yf.download("AAPL", period="2y", interval="1d", progress=False)

df["Date"] = df.index
df = df.sort_values('Date')
print(f"✅ Loaded {len(df)} data points for {ticker if not df.empty else 'AAPL'}")

# Prepare data
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
X, y = [], []
lookback = 60

for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train model
print("Training...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Save model
model.save("saved_lstm_model.h5")
print("✅ Model saved as saved_lstm_model.h5")

# Evaluate
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Training Loss: {train_loss:.6f}")
print(f"Testing Loss: {test_loss:.6f}")

print("🎯 LSTM Model Training Complete!")



