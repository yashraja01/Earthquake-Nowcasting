import pandas as pd
import numpy as np

#Defining Taiwan's cities with their latitudes and longitudes
cities = {
    'Taipei': (25.033, 121.5654),
    'Kaohsiung': (22.6273, 120.3014),
    'Taitung': (22.7613, 121.1438),
    'Tainan': (22.9999, 120.2269),
    'Chiayi': (23.4801, 120.4491),
    'Hsinchu': (24.8138, 120.9675),
    'Keelung': (25.1276, 121.7392),
    'Hualien': (23.9872, 121.6016),
    'Nantou': (23.918, 120.6775),
    'Pingtung': (22.552, 120.5488)
}

file_path = '/content/Taiwan Datasheet 3.5.xlsx'  # Change this to your actual file path
df = pd.read_excel(file_path)

#Just a check
required_columns = {'Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Mw'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"Missing one of the required columns: {required_columns}")

#Looping through cities and updating dataframes
for city, (city_lat, city_lon) in cities.items():
    #Calculate distance
    distance_km = np.sqrt((df['Latitude'] - city_lat)**2 + (df['Longitude'] - city_lon)**2) * 101.5

    #Check if distance is within 150 km, returns true or false
    within_radius = distance_km <= 150

    # Creating new columns
    city_distance_col = f'{city} (km)'
    city_date_col = f'{city} EQ Date'
    city_mag_col = f'{city} Magnitude'

    df[city_distance_col] = np.where(within_radius, distance_km.round(2), np.nan)
    df[city_date_col] = np.where(within_radius, df['Date'], np.nan)
    df[city_mag_col] = np.where(within_radius, df['Mw'], np.nan)

# Save back to the same Excel file
df.to_excel(file_path, index=False)

print(f"Updated file saved to {file_path}")
