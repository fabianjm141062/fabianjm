import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Fetch Earthquake Data from USGS
@st.cache
def fetch_earthquake_data(start_year, end_year, latitude, longitude, max_radius_km):
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={start_year}-01-01&endtime={end_year}-12-31&latitude={latitude}&longitude={longitude}&maxradiuskm={max_radius_km}"
    response = requests.get(url)
    data = response.json()
    features = data['features']
    earthquakes = []
    for feature in features:
        properties = feature['properties']
        geometry = feature['geometry']
        magnitude = properties['mag']
        depth = geometry['coordinates'][2]
        location = properties['place']
        earthquakes.append({
            'time': properties['time'],
            'magnitude': magnitude,
            'depth': depth,
            'location': location,
            'latitude': geometry['coordinates'][1],
            'longitude': geometry['coordinates'][0]
        })
    return pd.DataFrame(earthquakes)

# Step 2: Calculate PGA using an attenuation model
def calculate_pga(magnitude, distance):
    # Example of Boore-Atkinson 2008 model (simplified)
    c1 = -0.5
    c2 = 1.2
    c3 = -1.3
    c4 = -0.005
    pga_ln = c1 + c2 * magnitude + c3 * np.log(distance) + c4 * distance
    pga = np.exp(pga_ln)
    return pga

# Step 3: Calculate Sa with Fa and Fv for different soil types
def calculate_sa(Fa, Fv, Ss, S1, T):
    SMS = Ss * Fa
    SDS = (2/3) * SMS
    SML = S1 * Fv
    SDL = (2/3) * SML
    
    Sa = []
    for t in T:
        if t <= 0.2 * (SDL / SDS):
            Sa_value = SDS * (0.4 + 0.6 * (t / (0.2 * (SDL / SDS))))
        elif 0.2 * (SDL / SDS) < t <= SDL / SDS:
            Sa_value = SDS
        else:
            Sa_value = SDL / t
        Sa.append(Sa_value)
    
    return Sa

# Streamlit App UI
st.title("Respons Spektra Gempa Kota Manado (1800-2024)oleh Prof.Dr. Fabian J Manoppo")
st.write("""
Aplikasi ini menghitung dan menampilkan grafik respons spektra berdasarkan data gempa dari USGS untuk kota Manado, dengan mempertimbangkan berbagai jenis tanah.
""")

# Input parameters
start_year = st.sidebar.number_input("Start Year", min_value=1800, max_value=2024, value=1800)
end_year = st.sidebar.number_input("End Year", min_value=1800, max_value=2024, value=2024)
latitude = st.sidebar.number_input("Latitude", value=1.4748)
longitude = st.sidebar.number_input("Longitude", value=124.8421)
max_radius_km = st.sidebar.number_input("Max Radius (km)", value=100)

# Fetch earthquake data
earthquakes = fetch_earthquake_data(start_year, end_year, latitude, longitude, max_radius_km)
if earthquakes.empty:
    st.write("No earthquake data found for the specified parameters.")
else:
    st.write(f"Total earthquakes found: {len(earthquakes)}")

    # Calculate PGA for each earthquake
    earthquakes['distance'] = np.random.uniform(10, 300, size=len(earthquakes))  # Placeholder for actual distance calculation
    earthquakes['PGA'] = earthquakes.apply(lambda row: calculate_pga(row['magnitude'], row['distance']), axis=1)

    # Define soil types with Fa and Fv values
    soil_types = {
        'Soft Soil': {'Fa': 2.5, 'Fv': 3.5},
        'Stiff Soil': {'Fa': 1.6, 'Fv': 1.7},
        'Rock': {'Fa': 1.0, 'Fv': 1.0},
        'Hard Rock': {'Fa': 0.8, 'Fv': 0.8}
    }

    # Example spectral acceleration values for Manado
    Ss = 0.8  # Example short-period spectral acceleration
    S1 = 0.3  # Example long-period spectral acceleration
    T_values = np.linspace(0.1, 4.0, 100)

    # Plot Sa for different soil types
    plt.figure(figsize=(12, 8))
    for soil, factors in soil_types.items():
        Sa_values = calculate_sa(factors['Fa'], factors['Fv'], Ss, S1, T_values)
        plt.plot(T_values, Sa_values, label=f'{soil} (Fa={factors["Fa"]}, Fv={factors["Fv"]})')

    plt.xlabel('Periode (T) [detik]')
    plt.ylabel('Sa (g)')
    plt.title('Grafik Respons Spektra untuk Berbagai Jenis Tanah (Manado, 1800-2024)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Display the data
    st.write("Tabel Data Gempa:")
    st.dataframe(earthquakes)

    # Optional: Download the processed data as CSV
    csv = earthquakes.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Data as CSV", data=csv, file_name='earthquake_data_manado_with_sa.csv', mime='text/csv')
