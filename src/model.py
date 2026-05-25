import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

def construir_modelo_lstm(input_shape):
    model = Sequential([
        # Definición explícita de la forma de entrada
        Input(shape=input_shape),
        
        # Capas LSTM sin el parámetro input_shape
        LSTM(units=64, return_sequences=True),
        Dropout(0.2),
        
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        
        Dense(units=16, activation='relu'),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model