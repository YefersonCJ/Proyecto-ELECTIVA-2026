import pandas as pd
import requests
from datetime import datetime, timedelta

def extraer_datos_nasa(lat, lon, dias_atras=7200):
    # Truncación sistémica: T-7 días para evadir latencia de asimilación (-999)
    fecha_fin = datetime.now() - timedelta(days=7)
    fecha_inicio = fecha_fin - timedelta(days=dias_atras)

    inicio_str = fecha_inicio.strftime('%Y%m%d')
    fin_str = fecha_fin.strftime('%Y%m%d')

    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,WS10M,T2M,RH2M&community=RE&longitude={lon}&latitude={lat}&start={inicio_str}&end={fin_str}&format=JSON"

    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    if 'properties' not in data:
        return None

    df = pd.DataFrame(data['properties']['parameter'])
    
    df = df.rename(columns={
        'ALLSKY_SFC_SW_DWN': 'GHI',
        'WS10M': 'Viento',
        'T2M': 'Temp',
        'RH2M': 'Humedad'
    })
    
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    return df