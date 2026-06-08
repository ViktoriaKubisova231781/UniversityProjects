import pandas as pd 
import folium 
from io import StringIO
import re

def initialize_data():
    # URL of the raw CSV file
    url = "https://raw.githubusercontent.com/KeesKlijs/dataset/main/accidents_data_breda_2013-2022.csv"

    # Read the CSV file into a DataFrame
    df = pd.read_csv(url) 

    data = {
        "Kruispunt": [
            "Keizerstraat x Oude Vest",
            "Weerijssingel x Vincent van Goghstraat",
            "Academiesingel x Delpratsingel",
            "Backer x Ruebweg",
            "Ettensebaan x Breda-West",
            "Terheijdenseweg x Crogtdijk",
            "Westerparklaan x Ettensebaan",
            "Noordelijke rondweg x Veldsteen",
            "Willemstraat x Academiesingel",
            "Beverweg x Hooghout",
            "Adriaan van Bergenstraat x Nieuwe prinsenkade",
            "Van Coothplein",
            "Graaf Hendrik III Laan x Brederostraat"
        ],
        "Latitude": [
            51.588119, 51.568394, 51.589066, 51.570245, 51.585713, 51.596624,
            51.587601, 51.584876, 51.589029, 51.579836, 51.594620, 51.588244,
            51.576906
        ],
        "Longitude": [
            4.775935, 4.774556, 4.778311, 4.738678, 4.717468, 4.786052,
            4.712868, 4.725985, 4.777835, 4.808470, 4.772819, 4.775569,
            4.795405
        ]
    }

    # Create DataFrame
    df_dangerous_intersections = pd.DataFrame(data) 

    import psycopg2

    db_params = {
        'host': '194.171.191.226',
        'port': '6379',
        'database': 'postgres',
        'user': 'group15',
        'password': 'blockd_2024group15_73'
    }

    con = psycopg2.connect(**db_params)
    cur = con.cursor()

    sql = '''
    SELECT *
    FROM data_lake.safe_driving
    '''
    
    cur = con.cursor()
    cur.execute(sql)
    
    data = cur.fetchall()
    
    columns_names = [col.name for col in cur.description]

    # Create a pandas DataFrame from the fetched data
    df_savedriving = pd.DataFrame(data, columns=columns_names)

    # Filter and clean the junctions DataFrame
    junctions_df = df[df["Area_Code"].str.startswith("JTE")]
    accidents_df = df[df["Area_Code"].str.startswith("WVK")]

    from sklearn.cluster import DBSCAN

    # Create a numpy array of junction coordinates
    junction_coords = junctions_df[['Longitude', 'Latitude']].values

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=0.0003, min_samples=1).fit(junction_coords)

    # Get the cluster labels
    cluster_labels = clustering.labels_

    # Add the cluster labels to the junctions_df
    junctions_df['junction_id'] = cluster_labels

    single_junctions_df = junctions_df.drop_duplicates(subset='junction_id') 

    junctions_df['center_lat'] = junctions_df.groupby('junction_id')['Latitude'].transform('mean') 
    junctions_df['center_lon'] = junctions_df.groupby('junction_id')['Longitude'].transform('mean') 

    from scipy.spatial import cKDTree

    # Assuming your DataFrames are named 'accidents_df' and 'junctions_df'
    # Assuming your accident coordinates are in 'X' and 'Y' columns

    # Calculate average latitude and longitude for each junction cluster (assuming 'cluster_id' exists)
    junctions_df['center_lat'] = junctions_df.groupby('junction_id')['Latitude'].transform('mean')
    junctions_df['center_lon'] = junctions_df.groupby('junction_id')['Longitude'].transform('mean')

    # Optional: Drop duplicates if 'junction_id' uniquely identifies junctions
    # If 'junction_id' might have duplicates within clusters, keep all rows in 'junctions_df'
    single_junctions_df = junctions_df.drop_duplicates(subset='junction_id') 

    # Create GeoDataFrames from your DataFrames
    accidents_gdf = gpd.GeoDataFrame(
        accidents_df, 
        geometry=gpd.points_from_xy(accidents_df['Longitude'], accidents_df['Latitude']), 
        crs="EPSG:3857"
    )
    single_junctions_gdf = gpd.GeoDataFrame(
        single_junctions_df,  # Pass the actual data
        geometry=gpd.points_from_xy(single_junctions_df['center_lon'], single_junctions_df['center_lat']), 
        crs="EPSG:3857"
    ) 

    # perform nearest-neighbor join
    nearest_junctions = gpd.sjoin_nearest(accidents_gdf, single_junctions_gdf, max_distance=50)

    accidents_df['junction_id'] = nearest_junctions['junction_id']

    # Calculate average latitude and longitude for each junction cluster (assuming 'cluster_id' exists)
    junctions_df['center_lat'] = junctions_df.groupby('junction_id')['Latitude'].transform('mean')
    junctions_df['center_lon'] = junctions_df.groupby('junction_id')['Longitude'].transform('mean')

    # Optional: Drop duplicates if 'junction_id' uniquely identifies junctions
    # If 'junction_id' might have duplicates within clusters, keep all rows in 'junctions_df'
    single_junctions_df = junctions_df.drop_duplicates(subset='junction_id') 

    # Create GeoDataFrames from your DataFrames
    savedriving_gdf = gpd.GeoDataFrame(
        df_savedriving, 
        geometry=gpd.points_from_xy(df_savedriving['longitude'], df_savedriving['latitude']), 
        crs="EPSG:3857"
    )
    single_junctions_gdf = gpd.GeoDataFrame(
        single_junctions_df,  # Pass the actual data
        geometry=gpd.points_from_xy(single_junctions_df['center_lon'], single_junctions_df['center_lat']), 
        crs="EPSG:3857"
    ) 

    # perform nearest-neighbor join
    nearest_junctions = gpd.sjoin_nearest(savedriving_gdf, single_junctions_gdf, max_distance=50)

    df_savedriving['junction_id'] = nearest_junctions['junction_id']

    # Calculate average latitude and longitude for each junction cluster (assuming 'cluster_id' exists)
    junctions_df['center_lat'] = junctions_df.groupby('junction_id')['Latitude'].transform('mean')
    junctions_df['center_lon'] = junctions_df.groupby('junction_id')['Longitude'].transform('mean')

    # Optional: Drop duplicates if 'junction_id' uniquely identifies junctions
    # If 'junction_id' might have duplicates within clusters, keep all rows in 'junctions_df'
    single_junctions_df = junctions_df.drop_duplicates(subset='junction_id') 

    # Create GeoDataFrames from your DataFrames
    dangerous_intersections_gdf = gpd.GeoDataFrame(
        df_dangerous_intersections, 
        geometry=gpd.points_from_xy(df_dangerous_intersections['Longitude'], df_dangerous_intersections['Latitude']), 
        crs="EPSG:3857"
    )
    single_junctions_gdf = gpd.GeoDataFrame(
        single_junctions_df,  # Pass the actual data
        geometry=gpd.points_from_xy(single_junctions_df['center_lon'], single_junctions_df['center_lat']), 
        crs="EPSG:3857"
    ) 

    # perform nearest-neighbor join
    nearest_junctions = gpd.sjoin_nearest(dangerous_intersections_gdf, single_junctions_gdf, max_distance=50)

    df_dangerous_intersections['junction_id'] = nearest_junctions['junction_id']

    # Concatenate row-wise
    combined_df = pd.concat([junctions_df, accidents_df], axis=0)

    # Resetting index after concatenation
    combined_df.reset_index(drop=True, inplace=True)

    # Display combined DataFrame
    combined_df = combined_df[['Accident_Severity', 'Parties_Involved', 'junction_id']] 

    # Calculate severity index based on Accident_Severity
    severity_index_map = {
        'Vehicle Damage': 0.5,
        'Wounded': 1,
        'Fatal': 3
    }

    # Calculate the count of accidents, average number of parties involved,
    # severity index, and total amount of accidents for each junction_id
    junction_id_df = combined_df.groupby('junction_id').agg(
        Total_Accidents=('junction_id', 'count'),
        Avg_Parties_Involved=('Parties_Involved', 'mean'),
        Severity_Index=('Accident_Severity', lambda x: sum(severity_index_map[s] for s in x) / len(x))
    )

    # Resetting index to make junction_id a column
    junction_id_df.reset_index(inplace=True)

    df_savedriving['category'] = df_savedriving['category'].str.strip()

    # Selecting specific columns
    savedriving_subset = df_savedriving[['junction_id', 'maxwaarde', 'category']]

    # Filter for HARSH CORNERING, SPEED, BRAKING, and ACCELERATING incidents
    harsh_cornering = savedriving_subset[savedriving_subset['category'] == 'HARSH CORNERING']
    speed = savedriving_subset[savedriving_subset['category'] == 'SPEED']
    braking = savedriving_subset[savedriving_subset['category'] == 'BRAKING']
    accelerating = savedriving_subset[savedriving_subset['category'] == 'ACCELERATING']

    # Group by junction_id and category, and calculate average g force for each category
    harsh_cornering_summary = harsh_cornering.groupby('junction_id').agg(
        Avg_GForce_Harsh_Cornering=('maxwaarde', 'mean'),
        Count_Harsh_Cornering=('category', 'count')
    )

    speed_summary = speed.groupby('junction_id').agg(
        Count_Speed=('category', 'count'),
        Avg_GForce_Speed=('maxwaarde', 'mean')
    )

    braking_summary = braking.groupby('junction_id').agg(
        Count_Braking=('category', 'count'),
        Avg_GForce_Braking=('maxwaarde', 'mean')
    )

    accelerating_summary = accelerating.groupby('junction_id').agg(
        Count_Accelerating=('category', 'count'),
        Avg_GForce_Accelerating=('maxwaarde', 'mean')
    )

    # Merge the summaries on junction_id
    savedriving_junctionid_df = harsh_cornering_summary.merge(speed_summary, on='junction_id', how='outer')
    savedriving_junctionid_df = savedriving_junctionid_df.merge(braking_summary, on='junction_id', how='outer')
    savedriving_junctionid_df = savedriving_junctionid_df.merge(accelerating_summary, on='junction_id', how='outer')

    # Fill NaN values with 0 (for junctions with no incidents of certain type)
    savedriving_junctionid_df.fillna(0, inplace=True)

    # Calculate total incidents for each junction_id
    total_incidents = savedriving_junctionid_df.sum(axis=1).reset_index(name='Total_incidents')

    # Merge total_incidents with savedriving_junctionid_df
    savedriving_junctionid_df = pd.merge(savedriving_junctionid_df, total_incidents, on='junction_id', how='left')

    # Left join savedriving_junctionid_df to df_savedriving on junction_id
    final_df = pd.merge(junction_id_df, savedriving_junctionid_df, on='junction_id', how='left')

    dangerous_junction_ids = set(df_dangerous_intersections['junction_id'])

    # Add a new column 'dangerous' to final_df
    final_df['dangerous'] = final_df['junction_id'].apply(lambda x: 1 if x in dangerous_junction_ids else 0)

    # Drop junction_id 0 from final_df
    final_df = final_df[final_df['junction_id'] != 0]

    # Filter out junctions with less than 5 accidents
    final_df = final_df[final_df['Total_Accidents'] >= 5] 
    final_df = final_df[final_df['Total_incidents'] >= 10]
    return final_df 

