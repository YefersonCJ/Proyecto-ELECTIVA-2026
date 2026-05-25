import requests
import pandas as pd
from datetime import datetime, timedelta

def extraer_datos_nasa(lat, lon, dias_atras=7200):
    fecha_fin = datetime.now()
    fecha_ini = fecha_fin - timedelta(days=dias_atras)
    
    fmt = "%Y%m%d"
    url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
           f"parameters=ALLSKY_SFC_SW_DWN,WS10M,T2M,RH2M&community=RE&"
           f"longitude={lon}&latitude={lat}&"
           f"start={fecha_ini.strftime(fmt)}&end={fecha_fin.strftime(fmt)}&format=JSON")
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        df = pd.DataFrame(data['properties']['parameter'])
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df.columns = ['GHI', 'Viento', 'Temp', 'Humedad']
        return df
    except Exception as e:
        print(f"Error en extracción: {e}")
        return None