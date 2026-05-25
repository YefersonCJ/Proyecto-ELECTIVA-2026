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

def crear_secuencias(datos, pasos=30, horizonte=7):
    X, y = [], []
    for i in range(len(datos) - pasos - horizonte + 1):
        X.append(datos[i:(i + pasos), :])
        # Índice 0 asume que GHI es la primera columna. Retorna 7 días futuros.
        y.append(datos[(i + pasos):(i + pasos + horizonte), 0])
    return np.array(X), np.array(y)
    