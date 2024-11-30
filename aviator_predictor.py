import requests
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tkinter as tk
from tkinter import ttk
import threading

# Fetch historical data from the 1Win Aviator API
def fetch_history():
    url = "https://aviatorengine.1win.com/Aviator/GetGameHistory"  # Replace with actual endpoint
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()  # Expected: [{'multiplier': 1.5}, {'multiplier': 2.3}, ...]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching history: {e}")
        return None  # Return None if an error occurs

# Fetch the current multiplier
def fetch_current_state():
    url = "https://aviatorengine.1win.com/Aviator/GetActualBalloon"  # Replace with actual endpoint
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()  # Expected: {'multiplier': 2.3}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching current state: {e}")
        return None

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

# Build LSTM model
def build_lstm_model(seq_length):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train LSTM model using historical data from the API
def train_model(seq_length=10, epochs=50, batch_size=32):
    print("Fetching historical data...")
    history = fetch_history()  # Fetch directly from API
    if not history:
        print("No history data available. Exiting.")
        return None, []  # Return None and an empty list for history

    # Convert history to a NumPy array
    data = np.array([entry['multiplier'] for entry in history])
    
    # Create sequences for LSTM
    X, y = create_sequences(data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM input

    # Build and train the model
    print("Training model...")
    model = build_lstm_model(seq_length)
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    print("Model training complete.")
    return model, data.tolist()  # Return the model and the history list

# Real-time prediction
def real_time_prediction(model, seq_length, history, update_callback):
    print("Starting real-time prediction...")
    while True:
        current_state = fetch_current_state()
        if current_state:
            current_multiplier = current_state['multiplier']
            history.append(current_multiplier)
            if len(history) >= seq_length:
                input_sequence = np.array(history[-seq_length:]).reshape(1, seq_length, 1)
                prediction = model.predict(input_sequence)
                update_callback(current_multiplier, prediction[0][0])
            else:
                update_callback(current_multiplier, "Collecting data...")
        time.sleep(5)

# GUI Implementation
def start_gui():
    # Splash Screen
    splash = tk.Tk()
    splash.title("Aviator Predictor")
    splash.geometry("400x300")
    splash.resizable(False, False)
    ttk.Label(splash, text="Aviator Predictor", font=("Arial", 24)).pack(expand=True)
    ttk.Label(splash, text="by CyberJay", font=("Arial", 16)).pack()
    splash.after(3000, splash.destroy)  # Close splash screen after 3 seconds
    splash.mainloop()

    # Main App
    app = tk.Tk()
    app.title("Aviator Predictor by CyberJay")
    app.geometry("600x400")
    app.resizable(False, False)

    # GUI Elements
    ttk.Label(app, text="Aviator Predictor", font=("Arial", 20)).pack(pady=10)
    current_label = ttk.Label(app, text="Current Multiplier: --", font=("Arial", 16))
    current_label.pack(pady=10)
    predicted_label = ttk.Label(app, text="Predicted Next Multiplier: --", font=("Arial", 16))
    predicted_label.pack(pady=10)

    # Callback to update GUI
    def update_gui(current, predicted):
        current_label.config(text=f"Current Multiplier: {current}")
        predicted_label.config(text=f"Predicted Next Multiplier: {predicted:.2f}" if isinstance(predicted, float) else f"Predicted Next Multiplier: {predicted}")

    # Train Model and Start Predictions
    seq_length = 10
    model, history = train_model(seq_length=seq_length)
    if model is None:
        print("Model training failed. Exiting.")
        app.destroy()
        return

    # Start Real-Time Prediction in a Separate Thread
    threading.Thread(target=real_time_prediction, args=(model, seq_length, history, update_gui), daemon=True).start()

    app.mainloop()

# Run the GUI
if __name__ == "__main__":
    start_gui()
