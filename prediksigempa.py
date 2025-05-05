import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
import random
from datetime import datetime

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
            'Time': datetime.utcfromtimestamp(props['time'] / 1000.0).strftime('%Y-%m-%d %H:%M:%S'),
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

# Streamlit App
st.title("Prediksi Gempa Kota Manado Menggunakan Earthquake Data & Prediction in Manado (1800-2024)oleh Prof.Dr. Fabian J Manoppo")

# Get earthquake data for Manado from 1800 to 2024
starttime = '1800-01-01'
endtime = '2025-05-05'

st.write("Fetching earthquake data from USGS...")
data = get_earthquake_data(starttime, endtime, minlatitude, maxlatitude, minlongitude, maxlongitude)
df = extract_earthquake_data(data)

# Rename the columns to match what Streamlit expects
df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)

# Display the data in a table
st.write("Displaying earthquake data in the Manado region:")
st.dataframe(df)

# Map with historical earthquake data
st.write("Map with historical earthquake data:")
st.map(df[['lat', 'lon']])

# Filter data by magnitude
min_magnitude = st.slider("Minimum Magnitude", 0.0, 10.0, 4.0)
filtered_df = df[df['Magnitude'] >= min_magnitude]

# Display filtered data
st.write(f"Displaying earthquakes with magnitude greater than {min_magnitude}:")
st.dataframe(filtered_df)

# Map with filtered earthquake data
st.map(filtered_df[['lat', 'lon']])

# Earthquake prediction based on input date
st.write("Enter a date for earthquake prediction:")
input_date = st.date_input("Select date")

# Mock Prediction (using random location and magnitude for now)
# You can replace this with a real prediction model if available
if st.button("Predict Earthquake"):
    predicted_lat = random.uniform(minlatitude, maxlatitude)
    predicted_lon = random.uniform(minlongitude, maxlongitude)
    predicted_magnitude = random.uniform(4.0, 7.0)
    
    st.write(f"Predicted earthquake on {input_date}:")
    st.write(f"Latitude: {predicted_lat}, Longitude: {predicted_lon}, Magnitude: {predicted_magnitude}")
    
    # Data for prediction point
    predicted_point = pd.DataFrame({
        'lat': [predicted_lat],
        'lon': [predicted_lon],
        'Magnitude': [predicted_magnitude]
    })

    # Create a PyDeck map with a star marker for the predicted point
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=predicted_point,
        get_position='[lon, lat]',
        get_color='[255, 0, 0, 160]',
        get_radius=10000,
        radius_scale=20,
        pickable=True,
        stroked=True,
        filled=True,
        line_width_min_pixels=2,
    )

    # Create the deck.gl map
    view_state = pdk.ViewState(
        latitude=(minlatitude + maxlatitude) / 2,
        longitude=(minlongitude + maxlongitude) / 2,
        zoom=7,
        pitch=50,
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Predicted Earthquake\nLatitude: {lat}\nLongitude: {lon}\nMagnitude: {Magnitude}"}
    )

    st.pydeck_chart(r)
