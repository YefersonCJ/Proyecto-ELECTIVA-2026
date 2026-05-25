from src.extraction import extraer_datos_nasa
from src.processing import limpiar_y_normalizar, crear_secuencias

def iniciar_proyecto():
    print("--- INICIANDO PIPELINE ETL (NASA POWER) ---")
    
    # Paso 1: Extracción de datos crudos
    df_raw = extraer_datos_nasa()
    
    if df_raw is not None:
        # Paso 2: Transformación (Limpieza y Normalización)
        df_norm = limpiar_y_normalizar(df_raw)
        
        # Paso 3: Carga (Creación de secuencias para la red LSTM)
        X, y = crear_secuencias(df_norm.values)
        
        print(f"\n=== DATASET GENERADO EXITOSAMENTE ===")
        print(f"Total de muestras: {len(X)}")
        print(f"Forma de entrada (X): {X.shape} (Muestras, Días, Variables)")
        print(f"Forma de salida (y): {y.shape}")
        
        return X, y
    else:
        print("No se pudo completar el flujo de datos.")

if __name__ == "__main__":
    iniciar_proyecto()