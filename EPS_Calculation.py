import pandas as pd
import numpy as np
from openpyxl import load_workbook

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
df['Date'] = pd.to_datetime(df['Date'])




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

    df[city_distance_col] = np.where(within_radius, distance_km.round(4), np.nan)
    df[city_date_col] = np.where(within_radius, df['Date'], pd.NaT)
    df[city_mag_col] = np.where(within_radius, df['Mw'], np.nan)

# Save back to the same Excel file
df.to_excel(file_path, index=False)

print(f"Updated file saved to {file_path}")

# Ensure 'Date' column is datetime


# Prepare output list for summary sheet
summary_data = []

# Process each city
for city, (city_lat, city_lon) in cities.items():
    # Column names generated in previous script
    distance_col = f"{city} (km)"

    # Filter for earthquakes within 150 km and Mw >= 6
    valid_quakes = df[(~df[distance_col].isna()) & (df['Mw'] >= 6)]

    if not valid_quakes.empty:
        # Get latest (most recent) earthquake
        latest_quake = valid_quakes.sort_values(by='Date', ascending=False).iloc[0]

        # Get info from that quake
        latest_date = latest_quake['Date']
        latest_info = {
            'City': city,
            'Latitude': city_lat,
            'Longitude': city_lon,
            '': '',  # Empty placeholder column
            '  ': '',
            '   ': '',
            'EQ Date': latest_quake['Date'],
            'EQ Time': latest_quake['Time'],
            'EQ Latitude': latest_quake['Latitude'],
            'EQ Longitude': latest_quake['Longitude'],
            'Depth': latest_quake['Depth'],
            'Mw': latest_quake['Mw'],
            'Distance (km)': round(latest_quake[distance_col], 4),
        }

        # Step 1: Combine Date + Time into a new 'Datetime' column (once at the top, outside the loop)
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')

        # Step 2: Get the full datetime of the latest big quake
        latest_datetime = pd.to_datetime(str(latest_quake['Date']) + ' ' + str(latest_quake['Time']), errors='coerce')

        # Step 3: Filter small quakes after the big one, within 150km, and Mw < 6
        small_quakes = df[
            (df['Datetime'] > latest_datetime) &
            (~df[distance_col].isna()) &
            (df['Mw'] < 6)
        ]

        # Step 4: Count and store
        latest_info['Small Event Count'] = len(small_quakes)

    else:
        # No big earthquake found
        latest_info = {
            'City': city,
            'Latitude': city_lat,
            'Longitude': city_lon,
            '': '',  # Empty columns
            '  ': '',
            '   ': '',
            'EQ Date': None,
            'EQ Time': None,
            'EQ Latitude': None,
            'EQ Longitude': None,
            'Depth': None,
            'Mw': None,
            'Distance (km)': None,
            'Small Event Count': 0
        }

    # Append to summary list
    summary_data.append(latest_info)

# Convert to DataFrame
summary_df = pd.DataFrame(summary_data)

# Write to new sheet in same Excel file
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    summary_df.to_excel(writer, sheet_name='City EPS Summary', index=False)

print(f"'City EPS Summary' sheet added to {file_path}")

