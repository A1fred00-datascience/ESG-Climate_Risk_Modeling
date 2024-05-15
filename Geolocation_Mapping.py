import pandas as pd
from geopy.geocoders import Bing
from tqdm import tqdm
import time

# Clave de API de Bing
bing_api_key = 'AlQL0PZhRxsM4ohl_t24dNNwfCHaMBPOwBIr1g40Pc0J-D695c3mVsOHU1ha3PfP'

# Ruta del archivo Excel con datos de Pais
file_path = r'C:\Users\alfredo.serrano.fig1\Desktop\BAC\Datos\Creacion de Coordenadas\CP_El_Salvador.xlsx'

# Leer el archivo Excel
df = pd.read_excel(file_path)

# Inicializar el geocodificador de Bing
geolocator = Bing(api_key=bing_api_key)

# Función para obtener coordenadas con reintento
def get_coordinates(municipio, departamento, pais):
    for attempt in range(5):  # Intentar hasta 5 veces
        try:
            location = geolocator.geocode(f"{municipio}, {departamento}, {pais}")
            if location:
                return location.latitude, location.longitude
            else:
                return None, None
        except Exception as e:
            print(f"Intento {attempt+1}: Error obteniendo coordenadas para {municipio}, {departamento}, {pais}: {e}")
            time.sleep(5)  # Esperar 5 segundos antes de reintentar
    return None, None

# Inicializar listas para almacenar coordenadas
latitudes = []
longitudes = []

# Iterar sobre las filas del DataFrame con tqdm para mostrar una barra de progreso
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    lat, lon = get_coordinates(row['MUNICIPIO'], row['DEPARTAMENTO'], row['PAIS'])
    latitudes.append(lat)
    longitudes.append(lon)

# Añadir las coordenadas al DataFrame
df['Latitud'] = latitudes
df['Longitud'] = longitudes

# Guardar los resultados en un nuevo archivo Excel
output_file_path = r'C:\Users\alfredo.serrano.fig1\Desktop\BAC\Datos\Creacion de Coordenadas\CP_El_Salvador_with_coordinates.xlsx'
df.to_excel(output_file_path, index=False)

print(f"Archivo guardado en: {output_file_path}")

#---------------------------------------------------- ENGLISH VERSION ----------------------------------------#
# import pandas as pd
# from geopy.geocoders import Bing
# from tqdm import tqdm
# import time

# # Bing API key
# bing_api_key = 'AlQL0PZhRxsM4ohl_t24dNNwfCHaMBPOwBIr1g40Pc0J-D695c3mVsOHU1ha3PfP'

# # Path to the Excel file with data
# file_path = r'C:\Users\alfredo.serrano.fig1\Desktop\BAC\Datos\Creacion de Coordenadas\CP_Honduras.xlsx'

# # Read the Excel file
# df = pd.read_excel(file_path)

# # Initialize the Bing geocoder
# geolocator = Bing(api_key=bing_api_key)

# # Function to get coordinates with retries
# def get_coordinates(zipcode, county, state):
#     for attempt in range(5):  # Try up to 5 times
#         try:
#             location = geolocator.geocode(f"{county}, {state}")
#             if location:
#                 return location.latitude, location.longitude
#             else:
#                 return None, None
#         except Exception as e:
#             print(f"Attempt {attempt+1}: Error getting coordinates for {zipcode}, {county}, {state}: {e}")
#             time.sleep(5)  # Wait 5 seconds before retrying
#     return None, None

# # Initialize lists to store coordinates
# latitudes = []
# longitudes = []

# # Iterate over the rows of the DataFrame with tqdm to show a progress bar
# for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#     lat, lon = get_coordinates(row['zipcode'], row['county'], row['state'])
#     latitudes.append(lat)
#     longitudes.append(lon)

# # Add the coordinates to the DataFrame
# df['Latitude'] = latitudes
# df['Longitude'] = longitudes

# # Save the results to a new Excel file
# output_file_path = r'C:\Users\alfredo.serrano.fig1\Desktop\BAC\Datos\Creacion de Coordenadas\US_with_coordinates.xlsx'
# df.to_excel(output_file_path, index=False)

# print(f"File saved at: {output_file_path}")
