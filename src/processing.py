import numpy as np
import pandas as pd

def limpiar_y_normalizar(df):
    # Relleno estricto de vacíos residuales tras interpolación
    df = df.replace(-999, np.nan).interpolate(method='linear', limit_direction='both').fillna(0)
    
    # Prevención de división por cero
    rango = df.max() - df.min()
    rango = rango.replace(0, 1) 
    
    df_norm = (df - df.min()) / rango
    return df_norm.fillna(0)

def crear_secuencias(data, pasos=30):
    X, y = [], []
    for i in range(len(data) - pasos):
        X.append(data[i : i + pasos, :])
        y.append(data[i + pasos, 0]) 
        
    # Conversión explícita a float32 para el motor de TensorFlow
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
    