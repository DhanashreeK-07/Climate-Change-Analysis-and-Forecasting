import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

print("--- Preparing Multivariate Data for LSTM ---")

features = [
    "Dew Point Temp_C", 
    "Rel Hum_%", 
    "Wind Speed_km/h", 
    "Press_kPa", 
    "Visibility_km"
]

# FIX: Added the .csv extension here!
try:
    df = pd.read_csv('Project 1 - Weather Dataset.csv') 
except FileNotFoundError:
    print("❌ ERROR: Could not find the CSV file. Please check the spelling.")
    exit()

X = df[features].values
y = df['Temp_C'].values

X_lstm = X.reshape((X.shape[0], 1, len(features)))

print("--- Training Multivariate LSTM ---")
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, len(features))))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.fit(X_lstm, y, epochs=15, batch_size=32, verbose=1)

lstm_model.save('lstm_model.h5')
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)
print("\n✅ MODEL AND FEATURES SAVED SUCCESSFULLY")