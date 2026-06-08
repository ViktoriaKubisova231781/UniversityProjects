def load(
            # Defining database parameters
    db_params = {
        'host': '194.171.191.226',
        'port': '6379',
        'database': 'postgres',
        'user': 'group15',
        'password': 'blockd_2024group15_73'
    }

    # Establishing the connection
    conn_psycopg2 = psycopg2.connect(**db_params)
    cursor = conn_psycopg2.cursor()

    # Execute SQL query to select the whole dataset
    query = '''
        SELECT * 
        FROM group15_warehouse.accident_data_17_23;
    '''

    cursor.execute(query)

    # Fetch all rows
    new_accident_rows = cursor.fetchall()

    # Fetch column names
    column_names = [desc[0] for desc in cursor.description]

    # Convert the fetched data into a pandas DataFrame
    new_accidents = pd.DataFrame(new_accident_rows, columns=column_names)

    # Display the dataset
    display(new_accidents)

    # Convert all column names to capitalized format
    new_accidents = new_accidents.rename(columns=lambda x: x.capitalize())

    # Display the updated DataFrame
    print("DataFrame with capitalized column names:")
    display(new_accidents)

    # Handling missing values
    missing_values = new_accidents.isnull().sum()
    print("Missing values:")
    print(missing_values)

    # Dropping unnecessary columns
    new_accidents = new_accidents.drop(columns=['Municipality'])

    # Make changes in variable values
    # Replace 'Other vehicle' and '-' with 'Other' in 'First Mode of Transport'
    new_accidents['First mode of transport'] = new_accidents['First mode of transport'].replace({'Other vehicle': 'Other', '-': 'Other'})

    # Replace 'Other vehicle' with 'Other' in 'Second mode of Transport'
    new_accidents['Second mode of transport'] = new_accidents['Second mode of transport'].replace({'Other vehicle': 'Other'})

    # Handling textual data inconsistencies
    new_accidents['Town'] = new_accidents['Town'].str.capitalize() 

    # Converting categorical variables to category type
    categorical_cols = ['Accident severity', 'Town', 'First mode of transport', 'Second mode of transport',
                        'Area type', 'Light condition', 'Road location', 'Road condition', 'Road surface', 'Road situation', 'Speed limit', 'Street', 'Weather']

    for col in categorical_cols:
        new_accidents[col] = new_accidents[col].astype('category') 

        # Drop outliers from the DataFrame
    new_accidents = new_accidents.drop(outlier_indices)

    # Drop the "Accidents" column as it's no longer needed
    new_accidents = new_accidents.drop('Accidents', axis=1)

    # Display the cleaned DataFrame
    print("Cleaned DataFrame after removing outliers and 'Accidents' column:")
    display(new_accidents.head())

    #  Define the updated mapping dictionary
    speed_to_road_type = {
        '130 km/h': 'Highway',
        '120 km/h': 'Highway',
        '100 km/h': 'Rural Road',
        '90 km/h': 'Rural Road',
        '80 km/h': 'Rural Road',
        '70 km/h': 'Build-up Road',
        '50 km/h': 'Urban Road',
        '30 km/h': 'Town Center Road',
        '60 km/h': 'Build-up Road',
        '15 km/h': 'Residential Road',
        'Footpace / homezone': 'Honezone Road',
        'Unknown': 'Unknown'
    }

    # Create the new column 'Road Type' based on 'Speed limit'
    new_accidents['Road type'] = new_accidents['Speed limit'].map(speed_to_road_type)

    # Check the distribution of the new column
    print(new_accidents['Road type'].value_counts())

    # Display the first few rows to verify the changes
    display(new_accidents.head()) 

    # Label encode categorical variables
    from sklearn.preprocessing import LabelEncoder

    categorical_cols = ['Accident severity', 'Town', 'First mode of transport', 'Second mode of transport',
                        'Area type', 'Light condition', 'Road location', 'Road condition', 'Road surface', 'Road situation', 'Speed limit', 'Street', 'Weather', 'Road type']

    label_encoders = {}
    for column in categorical_cols:
        le = LabelEncoder()
        new_accidents[column] = le.fit_transform(new_accidents[column])
        label_encoders[column] = le  # Store the label encoder for inverse transformation if needed

    # Check the resulting DataFrame
    display(new_accidents.head())
    return new_accidents, label_encoders
)