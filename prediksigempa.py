import requests
import pandas as pd
import streamlit as st

# Function to get earthquake data from the USGS API
def get_earthquake_data(starttime, endtime, minlatitude, maxlatitude, minlongitude, maxlongitude):
    url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
    params = {
        'format': 'geojson',
        'starttime': starttime,
        'endtime': endtime,
        'minlatitude': minlatitude,
        'maxlatitude': maxlatitude,
        'minlongitude': minlongitude,
        'maxlongitude': maxlongitude,
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

# Extract earthquake data from the USGS response
def extract_earthquake_data(data):
    features = data['features']
    earthquakes = []
    for feature in features:
        props = feature['properties']
        geom = feature['geometry']['coordinates']
        earthquakes.append({
            'Time': props['time'],
            'Place': props['place'],
            'Magnitude': props['mag'],
            'Latitude': geom[1],
            'Longitude': geom[0],
            'Depth': geom[2]
        })
    return pd.DataFrame(earthquakes)

# Define coordinates for Manado and a region of interest
minlatitude = 0.0  # Adjust as needed
maxlatitude = 2.0
minlongitude = 124.0
maxlongitude = 126.0

# Get earthquake data for Manado from 1800 to 2024
starttime = '1800-01-01'
endtime = '2024-12-31'

# Streamlit App
st.title("Prediksi Gempa Kota Manado Berdasarkan Earthquake Data in Manado (1800-2024)oleh Fabian J Manoppo")

# Fetch the earthquake data
st.write("Fetching earthquake data from USGS...")
data = get_earthquake_data(starttime, endtime, minlatitude, maxlatitude, minlongitude, maxlongitude)
df = extract_earthquake_data(data)

# Rename the columns to match what Streamlit expects
df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)

# Display the data in a table
st.write("Displaying earthquake data in the Manado region:")
st.dataframe(df)

# Show the earthquake data on a map
st.map(df[['lat', 'lon']])

# Filter data by magnitude
min_magnitude = st.slider("Minimum Magnitude", 0.0, 10.0, 4.0)
filtered_df = df[df['Magnitude'] >= min_magnitude]

# Display filtered data
st.write(f"Displaying earthquakes with magnitude greater than {min_magnitude}:")
st.dataframe(filtered_df)

# Show filtered earthquakes on a map
st.map(filtered_df[['lat', 'lon']])
