import pandas as pd
import numpy as np
import random
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.io as pio

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Function to compute the average of infection data (from 'Samples')
def compute_average(row):
    numbers = [int(num) for num in row.split(':') if num.strip().isdigit()]
    if numbers:
        return sum(numbers) / len(numbers)
    return None

# Function to clean and convert coordinates to float
def clean_coordinates(coord):
    coord = ''.join(c for c in coord if c.isdigit() or c in ['.', '-'])
    try:
        return float(coord)
    except ValueError:
        return None

# Load bat database
bat_data = pd.read_csv('Bat Database.csv')
bat_data['Average'] = bat_data['Samples'].apply(compute_average)
bat_data['GPS lat'] = bat_data['GPS lat'].apply(clean_coordinates)
bat_data['GPS long'] = bat_data['GPS long'].apply(clean_coordinates)

# Drop rows with missing lat/long coordinates
bat_data = bat_data.dropna(subset=['GPS lat', 'GPS long'])

# Bat migration algorithm
def simulate_bat_migration(lat, lon, month):
    """
    Simulate bat migration using an algorithm based on proximity to favorable regions, 
    seasonal variations, and random variability.
    """
    # Favorable region is centered roughly at equatorial Africa
    seasonal_factor = np.sin((month - 1) * np.pi / 6)  # Sinusoidal seasonal variation
    random_lat_shift = np.random.uniform(-5.0, 5.0)  # Larger latitude shift
    random_lon_shift = np.random.uniform(-5.0, 5.0)  # Larger longitude shift

    # Adjust latitude and longitude based on migration factors
    migration_lat = lat + random_lat_shift * seasonal_factor
    migration_lon = lon + random_lon_shift * seasonal_factor

    # Migration Factor (less impactful on Risk Score)
    migration_factor = np.clip(1 + seasonal_factor * np.random.uniform(-0.1, 0.1), 0.9, 1.1)

    return migration_lat, migration_lon, migration_factor

# Pre-simulate migrations for all months
monthly_migrations = {}
for month in range(1, 13):
    migration_results = bat_data.apply(
        lambda row: simulate_bat_migration(row['GPS lat'], row['GPS long'], month), axis=1
    )
    migrated_data = bat_data.copy()
    migrated_data[['GPS lat', 'GPS long', 'Migration Factor']] = pd.DataFrame(
        migration_results.tolist(), index=bat_data.index
    )
    migrated_data['Risk Score'] = migrated_data['Average'] * migrated_data['Migration Factor']
    migrated_data['Risk Score'] = migrated_data['Risk Score'].fillna(migrated_data['Risk Score'].mean())
    monthly_migrations[month] = migrated_data

# Function to input the month manually
def get_user_month():
    while True:
        try:
            month = int(input("Enter the month (1-12): "))
            if 1 <= month <= 12:
                return month
            else:
                print("Please enter a valid month number between 1 and 12.")
        except ValueError:
            print("Invalid input! Please enter a number between 1 and 12.")

# Get user-selected month
selected_month = get_user_month()

# Use the pre-simulated data for the selected month
bat_data = monthly_migrations[selected_month]

# Normalize data for clustering
scaler = StandardScaler()
features = ['GPS lat', 'GPS long', 'Risk Score']

# Perform normalization
normalized_data = scaler.fit_transform(bat_data[features])

# Impute missing values in the normalized data
imputer = SimpleImputer(strategy='mean')
normalized_data = imputer.fit_transform(normalized_data)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)

# Perform clustering on the imputed normalized data
bat_data['Cluster'] = kmeans.fit_predict(normalized_data)

# Generate a CSV file for high-risk zones based on bat migrations and risk scores
high_risk_zones = bat_data[['GPS lat', 'GPS long', 'Risk Score']]
high_risk_zones.to_csv(f'ebola_risk_zones_month_{selected_month}.csv', index=False)

# Load population data CSV
population_data = pd.read_csv('africa_population_density.csv')

# Merge high-risk zones with population data based on geographical proximity
def find_nearest_country(lat, lon):
    distances = np.sqrt((population_data['Latitude'] - lat) ** 2 + (population_data['Longitude'] - lon) ** 2)
    nearest_country_index = distances.idxmin()
    return population_data.iloc[nearest_country_index]['Country']

high_risk_zones['Country'] = high_risk_zones.apply(
    lambda row: find_nearest_country(row['GPS lat'], row['GPS long']), axis=1
)

# Aggregate risk scores by country
country_risk = high_risk_zones.groupby('Country')['Risk Score'].max().reset_index()

# Debugging: Print some data to check
print(high_risk_zones.head())
print(country_risk.head())

# Plot the heatmap using Plotly
fig = px.density_mapbox(
    high_risk_zones, 
    lat='GPS lat', 
    lon='GPS long', 
    z='Risk Score', 
    radius=20,
    center=dict(lat=0, lon=20),  # Center the map around equatorial Africa
    zoom=2,
    mapbox_style="open-street-map",
    title=f"Ebola Risk Heatmap for Month {selected_month}"
)

# Show the map
fig.show()
