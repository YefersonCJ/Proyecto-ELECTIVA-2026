import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

def construir_modelo_lstm(input_shape, output_steps=7):
    modelo = Sequential()
    modelo.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    modelo.add(Dropout(0.2))
    modelo.add(LSTM(50))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(output_steps))
    modelo.compile(optimizer='adam', loss='mse')
    return modelo