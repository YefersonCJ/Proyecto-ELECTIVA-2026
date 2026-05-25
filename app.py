from flask import Flask, request, jsonify
from flask_cors import CORS
from src.extraction import extraer_datos_nasa
from src.processing import limpiar_y_normalizar, crear_secuencias
import time

app = Flask(__name__)
CORS(app)

@app.route('/procesar-completo', methods=['GET'])
def procesar_completo():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    # Extraer 20 años
    df_raw = extraer_datos_nasa(lat, lon, dias_atras=7200)
    if df_raw is None: return jsonify({"status": "error"}), 500
    
    # ETL
    df_norm = limpiar_y_normalizar(df_raw)
    
    # Preparar muestras para las tablas (últimos 10 registros)
    # Convertimos el índice (fecha) a una columna llamada 'fecha'
    crudos_web = df_raw.tail(10).reset_index()
    crudos_web.columns = ['fecha', 'GHI', 'Viento', 'Temp', 'Humedad']
    
    etl_web = df_norm.tail(10).reset_index()
    etl_web.columns = ['fecha', 'GHI', 'Viento', 'Temp', 'Humedad']

    # Dataset para LSTM (Ventana 30)
    X, y = crear_secuencias(df_norm.values, pasos=30)

    return jsonify({
        "status": "success",
        "crudos": crudos_web.astype(str).to_dict(orient='records'),
        "etl": etl_web.astype(str).to_dict(orient='records'),
        "dataset": {
            "total_registros": len(df_raw),
            "total_secuencias": X.shape[0],
            "shape": list(X.shape)
        }
    })

from src.model import construir_modelo_lstm

@app.route('/entrenar-ia', methods=['GET'])
def entrenar_ia():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    # 1. Pipeline de datos (20 años)
    df_raw = extraer_datos_nasa(lat, lon, dias_atras=7200)
    df_norm = limpiar_y_normalizar(df_raw)
    X, y = crear_secuencias(df_norm.values, pasos=30)
    
    # 2. Modelo
    modelo = construir_modelo_lstm(input_shape=(X.shape[1], X.shape[2]))
    
    # 3. Entrenamiento
    history = modelo.fit(X, y, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    
    # 4. PREDICCIÓN: Tomamos la última ventana de 30 días para predecir "mañana"
    ultima_ventana = X[-1] # Último bloque de 30 días
    ultima_ventana = ultima_ventana.reshape(1, 30, 4) # Ajustar para el modelo
    prediccion_norm = modelo.predict(ultima_ventana)
    
    # El resultado está normalizado (0-1), lo ideal es des-normalizarlo
    # Para efectos del ejercicio, enviamos el valor escalado
    
    return jsonify({
        "status": "success",
        "mse": history.history['loss'],
        "prediccion": float(prediccion_norm[0][0]),
        "valor_real_ultimo": float(y[-1])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)