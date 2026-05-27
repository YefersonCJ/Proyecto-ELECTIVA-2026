from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.extraction import extraer_datos_nasa
from src.processing import limpiar_y_normalizar, crear_secuencias
import traceback
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
# Importaciones de la arquitectura de Inteligencia Artificial (CORRECCIÓN INTEGRADA)
from src.model import construir_modelo_lstm
from tensorflow.keras.callbacks import EarlyStopping


app = Flask(__name__)
CORS(app)

@app.route('/entrenar-ia', methods=['GET'])
def entrenar_ia():
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        # 1. Extracción y Limpieza idéntica al ETL/EDA
        df = extraer_datos_nasa(lat, lon, dias_atras=7200)
        df.replace(-999.0, np.nan, inplace=True)
        df.interpolate(method='linear', inplace=True)
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        
        # Selección de características alineadas con el EDA
        columnas_features = ['GHI', 'Temp', 'Humedad', 'Viento']
        df_ia = df[columnas_features].dropna()
        
        # 2. Conversión a matriz numérica
        data_matrix = df_ia.values
        
        # 3. Escalado de Datos Min-Max
        min_val = data_matrix.min(axis=0)
        max_val = data_matrix.max(axis=0)
        range_val = np.where(max_val - min_val == 0, 1, max_val - min_val)
        data_scaled = (data_matrix - min_val) / range_val
        
        # 4. Construcción de Tensores Objetivo Independientes (GHI y Viento)
        lookback = 30
        horizonte = 7
        X, Y_ghi, Y_viento = [], [], []
        
        for i in range(len(data_scaled) - lookback - horizonte + 1):
            X.append(data_scaled[i:(i + lookback), :]) 
            Y_ghi.append(data_scaled[(i + lookback):(i + lookback + horizonte), 0])      # Target: GHI (Índice 0)
            Y_viento.append(data_scaled[(i + lookback):(i + lookback + horizonte), 3])   # Target: Viento (Índice 3)
            
        X = np.array(X)
        Y_ghi = np.array(Y_ghi)
        Y_viento = np.array(Y_viento)
        
        # --- PARTICIÓN DE DATOS ---
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        Y_train_ghi, Y_val_ghi = Y_ghi[:split], Y_ghi[split:]
        Y_train_viento, Y_val_viento = Y_viento[:split], Y_viento[split:]
        
        # --- ENTRENAMIENTO PARALELO (INSTANCIAS INDEPENDIENTES) ---
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        modelo_ghi = construir_modelo_lstm(input_shape=(X.shape[1], X.shape[2]))
        historial_ghi = modelo_ghi.fit(
            X_train, Y_train_ghi, epochs=50, batch_size=32,
            validation_data=(X_val, Y_val_ghi), callbacks=[early_stop], verbose=0
        )
        
        modelo_viento = construir_modelo_lstm(input_shape=(X.shape[1], X.shape[2]))
        historial_viento = modelo_viento.fit(
            X_train, Y_train_viento, epochs=50, batch_size=32,
            validation_data=(X_val, Y_val_viento), callbacks=[early_stop], verbose=0
        )
        
        # --- EXTRACCIÓN DE MÉTRICAS (GHI Y VIENTO) ---
        loss_history_ghi = [float(x) for x in historial_ghi.history['loss']]
        val_loss_history_ghi = [float(x) for x in historial_ghi.history['val_loss']]
        
        loss_history_viento = [float(x) for x in historial_viento.history['loss']]
        val_loss_history_viento = [float(x) for x in historial_viento.history['val_loss']]
        
        ultimos_7_dias_norm = data_scaled[-7:, 0].tolist()

        # Extraer y desnormalizar el histórico de viento de los últimos 7 días (Índice 3)
        viento_pasado_norm = data_scaled[-7:, 3].tolist()
        viento_pasado_real = [(float(x) * range_val[3]) + min_val[3] for x in viento_pasado_norm]
        
        # --- INFERENCIA PROBABILÍSTICA (MONTE CARLO DROPOUT) ---
        ultima_secuencia = data_scaled[-lookback:].reshape(1, lookback, X.shape[2])
        iteraciones_mc = 100
        
        # Ejecución estocástica: training=True mantiene los nodos apagados aleatoriamente
        preds_mc_ghi = np.array([modelo_ghi(ultima_secuencia, training=True)[0] for _ in range(iteraciones_mc)])
        preds_mc_viento = np.array([modelo_viento(ultima_secuencia, training=True)[0] for _ in range(iteraciones_mc)])
        
        # Cálculo de medias (Pronóstico determinista)
        media_ghi = np.mean(preds_mc_ghi, axis=0)
        media_viento = np.mean(preds_mc_viento, axis=0)
        
        # Cálculo de desviación estándar (Márgenes de error dinámicos progresivos)
        std_ghi = np.std(preds_mc_ghi, axis=0)
        std_viento = np.std(preds_mc_viento, axis=0)
        
        # Conversión a listas planas
        preds_futuras_ghi = [float(x) for x in media_ghi]
        margen_error_ghi = [float(x) for x in std_ghi]
        preds_futuras_viento = [float(x) for x in media_viento]
        
        # Desnormalización de las proyecciones y varianzas del viento a metros por segundo
        viento_real = [(float(x) * range_val[3]) + min_val[3] for x in preds_futuras_viento]
        margen_error_viento_real = [float(x) * range_val[3] for x in std_viento]
        
        # --- MODELOS GPR (RADIACIÓN GHI Y VELOCIDAD DEL VIENTO) ---
        # Submuestreo estricto: Últimos 365 días para evitar colapso de memoria O(N^3)
        X_gpr = np.arange(365).reshape(-1, 1)
        y_gpr_ghi = data_scaled[-365:, 0]  # Índice 0: GHI
        y_gpr_viento = data_scaled[-365:, 3] # Índice 3: Viento

        # Configuración del Kernel GPR (Compartido para ambas variables)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        
        # Entrenamiento GPR - GHI
        gpr_modelo_ghi = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        gpr_modelo_ghi.fit(X_gpr, y_gpr_ghi)
        
        # Entrenamiento GPR - Viento
        gpr_modelo_viento = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        gpr_modelo_viento.fit(X_gpr, y_gpr_viento)

        # Inferencia predictiva GPR: Próximos 7 días
        X_futuro_gpr = np.arange(365, 365 + 7).reshape(-1, 1)
        preds_gpr_ghi_norm, std_gpr_ghi_norm = gpr_modelo_ghi.predict(X_futuro_gpr, return_std=True)
        preds_gpr_viento_norm, std_gpr_viento_norm = gpr_modelo_viento.predict(X_futuro_gpr, return_std=True)
        
        # Desnormalización de GPR GHI a valores físicos
        preds_gpr_ghi_real = [(float(x) * range_val[0]) + min_val[0] for x in preds_gpr_ghi_norm]
        std_gpr_ghi_real = [float(x) * range_val[0] for x in std_gpr_ghi_norm]

        # Desnormalización de GPR Viento a m/s
        preds_gpr_viento_real = [(float(x) * range_val[3]) + min_val[3] for x in preds_gpr_viento_norm]
        std_gpr_viento_real = [float(x) * range_val[3] for x in std_gpr_viento_norm]

        # Generar predicciones de entrenamiento para evaluar el R²
        preds_train_ghi = modelo_ghi.predict(X_train, verbose=0)
        r2_ghi_train = float(r2_score(Y_train_ghi, preds_train_ghi))

        # Generar predicciones de entrenamiento para evaluar el R² de viento
        preds_train_viento = modelo_viento.predict(X_train, verbose=0)
        r2_viento_train = float(r2_score(Y_train_viento, preds_train_viento))

        # 6. Estructura JSON final
        return jsonify({
            "status": "success",
            "escalado": {
                "min": min_val.tolist(),
                "range": range_val.tolist()
            },
            "pasado_reciente": ultimos_7_dias_norm,
            "lstm": {
                "mse_train": float(loss_history_ghi[-1]),
                "r2_train": r2_ghi_train,
                "loss_history": loss_history_ghi,
                "val_loss_history": val_loss_history_ghi,
                "forecast": preds_futuras_ghi,
                "uncertainty": margen_error_ghi
            },
            "lstm_viento": {
                "mse_train": float(loss_history_viento[-1]),
                "r2_train": r2_viento_train,
                "loss_history": loss_history_viento,
                "val_loss_history": val_loss_history_viento,
                "forecast_norm": preds_futuras_viento,
                "forecast_real": viento_real,
                "uncertainty_real": margen_error_viento_real,
                "pasado_real": viento_pasado_real
            },
            "gpr": {
                "forecast": preds_gpr_ghi_real,
                "uncertainty": std_gpr_ghi_real,
                "score": float(gpr_modelo_ghi.score(X_gpr, y_gpr_ghi))
            },
            "gpr_viento": {
                "forecast_real": preds_gpr_viento_real,
                "uncertainty_real": std_gpr_viento_real,
                "score": float(gpr_modelo_viento.score(X_gpr, y_gpr_viento))
            }
        })
        
    except Exception as e:
        import traceback
        print("FALLO CRÍTICO EN PIPELINE IA:\n", traceback.format_exc())
        return jsonify({
            "status": "error",
            "msg": f"Excepción en la arquitectura neuronal o GPR: {str(e)}",
            "trace": traceback.format_exc()
        }), 500


@app.route('/procesar-completo', methods=['GET'])
def procesar_completo():
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        df = extraer_datos_nasa(lat, lon, dias_atras=7200)
        
        df_crudos_muestra = df.head(5).fillna(0)
        crudos_list = []
        for idx, row in df_crudos_muestra.iterrows():
            crudos_list.append({
                "fecha": idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                "GHI": float(row.get('GHI', 0)),
                "Viento": float(row.get('WS10M', row.get('Viento', 0))),
                "Temp": float(row.get('T2M', row.get('Temp', 0))),
                "Humedad": float(row.get('RH2M', row.get('Humedad', 0)))
            })
            
        df.replace(-999.0, np.nan, inplace=True)
        df.interpolate(method='linear', inplace=True)
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        
        columnas_objetivo = ['GHI', 'Temp', 'Humedad', 'Viento']
        df_limpio = df[columnas_objetivo].dropna()
        
        df_etl_muestra = df_limpio.head(5)
        etl_list = []
        for idx, row in df_etl_muestra.iterrows():
            etl_list.append({
                "fecha": idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                "GHI": float(row['GHI']),
                "Viento": float(row['Viento']),
                "Temp": float(row['Temp']),
                "Humedad": float(row['Humedad'])
            })
            
        lookback = 30
        total_registros = len(df_limpio)
        total_secuencias = total_registros - lookback if total_registros > lookback else 0
        features = len(columnas_objetivo)
        
        return jsonify({
            "status": "success",
            "crudos": crudos_list,
            "etl": etl_list,
            "dataset": {
                "total_registros": int(total_registros),
                "total_secuencias": int(total_secuencias),
                "shape": f"{total_secuencias}, {lookback}, {features}"
            }
        })
    except Exception as e:
        import traceback
        print("FALLO EN PROCESAR COMPLETO:\n", traceback.format_exc())
        return jsonify({"status": "error", "trace": traceback.format_exc()}), 500


@app.route('/ejecutar-eda', methods=['GET'])
def ejecutar_eda():
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        df = extraer_datos_nasa(lat, lon, dias_atras=7200)
        
        df.replace(-999.0, np.nan, inplace=True)
        df.interpolate(method='linear', inplace=True)
        df.bfill(inplace=True)
        df.ffill(inplace=True)
        
        columnas_objetivo = ['GHI', 'Temp', 'Humedad', 'Viento']
        df_limpio = df[columnas_objetivo].dropna()
        
        descriptivo = df_limpio.describe().round(4).to_dict()
        varianzas = df_limpio.var().round(4).to_dict()
        for col in descriptivo:
            descriptivo[col]['var'] = varianzas[col]
            descriptivo[col]['iqr'] = round(descriptivo[col]['75%'] - descriptivo[col]['25%'], 4)
            
        conteo_ghi, limites_ghi = np.histogram(df_limpio['GHI'], bins=30)
        
        corr = df_limpio.corr().round(4).to_dict()
        df_muestra = df_limpio.tail(365)
        
        scatter_temp = [{'x': row['Temp'], 'y': row['GHI']} for _, row in df_muestra.iterrows()]
        scatter_hum = [{'x': row['Humedad'], 'y': row['GHI']} for _, row in df_muestra.iterrows()]
        
        max_ghi = df_muestra['GHI'].max()
        bubble_data = [
            {
                'x': row['Temp'], 
                'y': row['Humedad'], 
                'r': float((row['GHI'] / max_ghi) * 20 + 2) if max_ghi > 0 else 2
            } 
            for _, row in df_muestra.iterrows()
        ]
        
        if isinstance(df_muestra.index, pd.DatetimeIndex):
            fechas_str = df_muestra.index.strftime('%Y-%m-%d').tolist()
        else:
            fechas_str = [f"Día {i}" for i in range(len(df_muestra))]
            
        return jsonify({
            "status": "success",
            "descriptivo": descriptivo,
            "hist_ghi": {"conteo": conteo_ghi.tolist(), "limites": limites_ghi.tolist()},
            "box_data_ghi": df_limpio['GHI'].tolist(),
            "correlacion": corr,
            "scatter_temp": scatter_temp,
            "scatter_hum": scatter_hum,
            "bubble_multi": bubble_data,
            "serie_tiempo": {"fechas": fechas_str, "ghi": df_muestra['GHI'].tolist(), "temp": df_muestra['Temp'].tolist()}
        })
    except Exception as e:
        import traceback
        return jsonify({"status": "error", "trace": traceback.format_exc()}), 500

    app = Flask(__name__)
    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')

    @app.route('/resultados', methods=['GET'])
    def resultados():
        return render_template('resultados.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)