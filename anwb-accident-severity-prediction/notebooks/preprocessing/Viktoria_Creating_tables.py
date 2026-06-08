import psycopg2
import pandas as pd

# Database connection parameters
db_params = {
    'host': '194.171.191.226',
    'port': '5432',  # PostgreSQL's default port is 5432, not 6379
    'database': 'postgres',
    'user': 'group15',
    'password': 'blockd_2024group15_73'
}

def create_table_and_fetch_data(table_name, source_table):
    try:
        # Establish the connection and create a cursor
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cursor:
                # Create table query
                create_table_query = f'''
                    CREATE TABLE IF NOT EXISTS group15_warehouse.{table_name} AS
                    SELECT * FROM data_lake.{source_table};
                '''
                cursor.execute(create_table_query)
                conn.commit()
                
                # Fetch data query
                query = f'''
                    SELECT * 
                    FROM group15_warehouse.{table_name};
                '''
                cursor.execute(query)
                
                # Fetch all rows and column names
                rows = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                
                # Convert to DataFrame
                data_df = pd.DataFrame(rows, columns=column_names)
                return data_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Creating and fetching data from the tables
new_accidents = create_table_and_fetch_data('accident_data_17_23', 'accident_data_17_23')
safe_driving = create_table_and_fetch_data('safe_driving', 'safe_driving')
temperature = create_table_and_fetch_data('temperature', 'temperature')
wind = create_table_and_fetch_data('wind', 'wind')
precipitation = create_table_and_fetch_data('precipitation', 'precipitation')
greenery = create_table_and_fetch_data('greenery', 'greenery')
breda_road = create_table_and_fetch_data('breda_road', 'breda_road')
accidents = create_table_and_fetch_data('accidents', 'accidents')
gridcodes = create_table_and_fetch_data('gridcodes', 'gridcodes')

# Display the DataFrames
if new_accidents is not None:
    display(new_accidents)
if safe_driving is not None:
    display(safe_driving)
if temperature is not None:
    display(temperature)
if wind is not None:
    display(wind)
if precipitation is not None:
    display(precipitation)
if greenery is not None:
    display(greenery)
if breda_road is not None:
    display(breda_road)
if accidents is not None:
    display(accidents)
if gridcodes is not None:
    display(gridcodes)
