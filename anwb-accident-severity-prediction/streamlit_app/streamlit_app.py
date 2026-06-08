import streamlit as st
import pandas as pd
import geocoder
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import psycopg2
import joblib
from sklearn.preprocessing import LabelEncoder

def get_coordinates(address):
    location = geocoder.osm(address)
    if location.ok:
        return location.latlng
    else:
        return None


def display_map_and_zoom(map_placeholder, street_name):
    if street_name:
        no = 0
        if street_name in streets:
            no = streets[street_name]

        coordinates = get_coordinates(f'{street_name}, Breda, Netherlands')
        if coordinates:
            m = folium.Map(location=coordinates, zoom_start=street_zoom)
            popup = f'{street_name} - Accidents: {no}'
            folium.Marker(location=coordinates, popup=popup).add_to(m)
            folium_static(m, width = map_dimensions['width'], height = map_dimensions['height'])
        else:
            st.error("Error: Geocoder server does not respond. It's not my fault, I swear! (also make sure you didn't do any typo)")
            display_default_map(map_placeholder)
    else:
        display_default_map(map_placeholder)


def display_default_map(map_placeholder):
    m = folium.Map(location=[51.5868, 4.7759], zoom_start=default_zoom)
    folium_static(m, width = map_dimensions['width'], height = map_dimensions['height'])


def display_map_and_markers(coordinates):
    m = folium.Map(location=[51.5868, 4.7759], zoom_start=default_zoom)
    for lat, lon, count, street in zip(coordinates['latitude'], coordinates['longitude'], coordinates['no_of_accidents'], coordinates['street']):
        folium.Marker(
            location=[lat, lon],
            popup=f'Street: {street} \nAccidents: {count}',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    folium_static(m, width = map_dimensions['width'], height = map_dimensions['height'])


def main_page():
    st.write("<h1 style='text-align: center; color: red; '>Street Finder Breda</h1>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    if col2.button('Severity'):
        st.session_state['page'] = 'severity'
    if col3.button('Visuals'):
        st.session_state['page'] = 'visuals'
    if col4.button('Dangerous Areas'):
        st.session_state['page'] = 'dangerous_areas'

    
    search_col, reset_col = st.columns([16, 1])

    with search_col:
        search_address = st.text_input('', value=st.session_state.get('search_address', ''), placeholder='Enter a street name in Breda')


    with reset_col:
        for k in range(2): st.write('')
        if st.button('Reset'):
            search_address = ''
            st.session_state['search_address'] = search_address
            st.session_state['page'] = 'main'

    map_placeholder = st.empty()

    display_map_and_zoom(map_placeholder, search_address)

    
def severity_page():
    st.write("<h1 style='text-align: center; color: red;'>Severity</h1>", unsafe_allow_html=True)
    st.button('Back', on_click=go_back)

    selected_values = {}

    predict = False
    
    cols1 = st.columns(6)
    cols2 = st.columns(6)
        
    for i, (key, options) in enumerate(list(dropdown_options.items())[:6]):
        with cols1[i]:
            selected_values[key] = st.selectbox(key, options)
            
    for i, (key, options) in enumerate(list(dropdown_options.items())[6:]):
        with cols2[i]:
            selected_values[key] = st.selectbox(key, options)
            
    with cols2[4]:
        selected_values['Street'] = st.text_input('Street')
    
    with cols2[5]:
        for k in range(2): st.write('')
        
        if st.button('Predict'):
            if selected_values['Street'] != '':
                predict = True
            else: st.write('Enter the street name')

    accident_values = pd.DataFrame([selected_values])
    
    if predict:
        for k in range(3): st.write('')
        #st.image(fatal_image, caption='Your Image Caption', use_column_width=True)
        column_order = ['Area type', 'Light condition', 'Road location', 'Road condition',
       'Road surface', 'Road situation', 'Speed limit', 'Street', 'Weather',
       'Initial transport', 'Secondary transport']
        accident_values = accident_values.reindex(columns=column_order)
        
        le = LabelEncoder()
        
        for column in accident_values.columns:
            accident_values[column] = le.fit_transform(accident_values[column])
        
        predicted_severity = model.predict(accident_values)
        # 0 - fatal, 1 - injured, 2 - material damage only
        if predicted_severity == 0: predicted_severity = 'fatal'
        elif predicted_severity == 1: predicted_severity = 'injured'
        else: predicted_severity = 'material damage only'
        
        st.write(f'Predicted severity: {predicted_severity}')



def dangerous_areas():
    st.button('Back', on_click=go_back)
    st.write("<h1 style='text-align: center; color: red;'>Most dangerous areas in Breda</h1>", unsafe_allow_html=True)
    display_map_and_markers(dangerous_areas_coordinates)

def visuals():
    st.button('Back', on_click=go_back)
    st.write("<h1 style='text-align: center; color: red;'>Visuals</h1>", unsafe_allow_html=True)

    # Calculating the speeding for each row
    df_speeding['speeding'] = df_speeding['maxwaarde'] - df_speeding['speed_limit']
    
    # Grouping by road_name
    average_speeding = df_speeding.groupby(['road_name']).agg(average_speeding=('speeding', 'mean'))
    average_speeding.reset_index(inplace=True)
    
    top_10_speeding_streets = average_speeding.sort_values(by='average_speeding', ascending=False).head(10)
    
    # creating the streamlit plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_10_speeding_streets['road_name'], top_10_speeding_streets['average_speeding'])
    plt.xlabel('Average Speeding (km/h)')
    plt.ylabel('Street')
    plt.title('Top 10 Streets in Breda with Highest Average Speeding')
    plt.gca().invert_yaxis()
    
    st.pyplot(plt)

def go_back():
    st.session_state['page'] = 'main'

st.set_page_config(layout='wide')

map_dimensions = {'width': 1510, 'height': 800}

default_zoom = 13
street_zoom = 16

breda_coordinates = pd.DataFrame({'lat': [51.5868], 'lon': [4.7759]})

fatal_picture = ''

db_params = {
    'host': '194.171.191.226',
    'port': '6379',
    'database': 'postgres',
    'user': 'group15',
    'password': 'blockd_2024group15_73'
}

con = psycopg2.connect(**db_params)

sql = '''
SELECT *
FROM group15_warehouse.accidents_on_each_street;
'''

cur = con.cursor()
cur.execute(sql)

data = cur.fetchall()

columns_names = [col.name for col in cur.description]
df = pd.DataFrame(data, columns = columns_names)

streets = dict(df[['street', 'no_of_accidents']].to_dict('split')['data'])

sql = '''
SELECT *
FROM group15_warehouse.dangerous_coordinates;
'''

cur = con.cursor()
cur.execute(sql)

data = cur.fetchall()

columns_names = [col.name for col in cur.description]
df = pd.DataFrame(data, columns = columns_names)

dangerous_areas_coordinates = df.to_dict(orient='list')

dropdown_options = {
    'Initial transport': ['Car', 'Lorry', 'Delivery van', 'Bicycle', 'Light-moped', 'Motorcycle', 'Bus', 'Pedestrian', 'Moped', 'Other'],
    'Secondary transport': ['Car', 'Lorry', 'Delivery van', 'Bicycle', 'Light-moped', 'Motorcycle', 'Bus', 'Pedestrian', 'Moped', 'Other'],
    'Area type': ['Urban area', 'Rural area', 'Unknown'],
    'Light condition': ['Daylight', 'Darkness', 'Twilight'],
    'Road location': ['Road section', 'Intersection'],
    'Road condition': ['Dry', 'Wet/damp', 'Snow/black ice', 'Unknown'],
    'Road surface': ['Asphalt (other)', 'Porous asphalt', 'Brick', 'Concrete', 'Unknown'],
    'Road situation': ['Straight road', 'Intersection - 4 arms', 'Intersection - 3 arms', 'Bend', 'Roundabout', 'Straight Road - separated carriageway', 'Unknown'],
    'Weather': ['Dry', 'Rain', 'Snow/hale', 'Fog', 'Hard gust of wind', 'Unknown'],
    'Speed limit': ['15 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h', '90 km/h', '100 km/h', '120 km/h', '130 km/h', 'Unknown']
}

sql = '''
SELECT *
FROM group15_warehouse.accident_data_17_23;
'''

cur = con.cursor()
cur.execute(sql)

data = cur.fetchall()

columns_names = [col.name for col in cur.description]
accident_data = pd.DataFrame(data, columns = columns_names)

sql = '''
SELECT *
FROM group15_warehouse.speeding;
'''

cur = con.cursor()
cur.execute(sql)

data = cur.fetchall()

columns_names = [col.name for col in cur.description]
df_speeding = pd.DataFrame(data, columns = columns_names)

cur.close()

model = joblib.load('model.joblib')


if 'page' not in st.session_state:
    st.session_state['page'] = 'main'

if st.session_state['page'] == 'main':
    main_page()
elif st.session_state['page'] == 'severity':
    severity_page()
elif st.session_state['page'] == 'dangerous_areas':
    dangerous_areas()
elif st.session_state['page'] == 'visuals':
    visuals()