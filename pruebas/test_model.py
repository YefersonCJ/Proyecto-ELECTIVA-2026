import numpy as np
import pandas as pd
import pytest
from src.processing import limpiar_y_normalizar, crear_secuencias

def test_limpiar_y_normalizar():
    # Configuración de matriz de datos con valores de error y vacíos
    datos_prueba = pd.DataFrame({
        'GHI': [100.0, -999.0, 300.0, 400.0],
        'Temp': [20.0, 25.0, 30.0, 35.0],
        'Humedad': [50.0, 55.0, 60.0, 65.0],
        'Viento': [2.0, 2.0, 2.0, 2.0]
    })
    
    resultado = limpiar_y_normalizar(datos_prueba)
    
    # Verificación de eliminación de valores de error satelital
    assert not (resultado == -999.0).any().any()
    # Verificación de los límites de la normalización lineal
    assert resultado.max().max() <= 1.0
    assert resultado.min().min() >= 0.0

def test_crear_secuencias():
    # Configuración de matriz numérica simulada (40 registros, 4 variables)
    datos_simulados = np.random.rand(40, 4)
    ventanas_tiempo = 30
    proyeccion = 7
    
    X, y = crear_secuencias(datos_simulados, pasos=ventanas_tiempo, horizonte=proyeccion)
    
    # Validación del volumen de muestras generado (40 - 30 - 7 + 1 = 4)
    assert len(X) == 4
    assert len(y) == 4
    # Validación de las dimensiones algebraicas de los tensores
    assert X.shape == (4, ventanas_tiempo, 4)
    assert y.shape == (4, proyeccion)