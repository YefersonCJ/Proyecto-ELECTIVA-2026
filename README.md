# Proyecto-ELECTIVA-2026
Este proyecto implementa un pipeline analítico y una arquitectura predictiva multimodelo para la estimación de radiación solar (GHI) y velocidad del viento (WS10M), conectando un backend de Flask con modelos avanzados de Machine Learning y un frontend interactivo con Chart.js.

Arquitectura del Sistema
Backend (Flask): Extracción automatizada desde NASA POWER API, limpieza de datos, escalamiento Min-Max, entrenamiento en paralelo de redes LSTM (TensorFlow/Keras) e inferencia no paramétrica mediante Procesos Gaussianos (GPR via Scikit-Learn).

Frontend (HTML5/CSS3/JavaScript): Dashboard analítico, visualización probabilística continua, benchmarking de divergencia algorítmica y estimación de generación energética final (kWh/día).

Requisitos Previos
Python 3.9 o superior

Git instalado en el sistema

Configuración del Entorno Virtual (venv)
Clonar el repositorio y acceder al directorio del proyecto:
git clone 
cd

Crear el entorno virtual de Python:
python -m venv venv

Activar el entorno virtual:

En Linux/macOS:
source venv/bin/activate

En Windows (CMD):
venv\Scripts\activate.bat

En Windows (PowerShell):
venv\Scripts\Activate.ps1

Instalación de Dependencias
Ejecutar la instalación de las librerías base requeridas por el pipeline analítico:

pip install --upgrade pip
pip install flask numpy pandas requests tensorflow scikit-learn

Resumen de Librerías Core Utilizadas
flask: Suministro de endpoints de API RESTful (/entrenar_ia).

Ejecución de la Aplicación
Iniciar el servidor de Flask:
python app.py

Acceder al aplicativo local mediante el navegador web:
http://127.0.0.1:5000

Flujo de operación:

Ingrese coordenadas geográficas en el formulario base para disparar el pipeline de extracción.

El backend ejecutará el ETL de 7200 días de registro y entrenará de forma paralela los estimadores LSTM y GPR.

Los resultados se serializan en un objeto JSON estructurado que incluye los parámetros de la escala física original (data.escalado).

El frontend (resultados.html) procesará el payload, desnormalizará las predicciones centrales del LSTM mediante la inversión del Min-Max en JavaScript, y proyectará las métricas físicas reales en las tablas de benchmarking y consolidación energética.

numpy: Operaciones matriciales, cálculo estocástico y manipulación de tensores.

tensorflow: Construcción y entrenamiento de la arquitectura recurrente LSTM.

scikit-learn: Ajuste del regresor por procesos gaussianos (GaussianProcessRegressor) y optimización de kernels paramétricos (RBF, WhiteKernel).

requests: Consumo asíncrono de la API externa de la NASA.
