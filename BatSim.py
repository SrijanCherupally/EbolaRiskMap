import pandas as pd
import numpy as np
import random
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.io as pio
from datetime import datetime, timedelta

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

# Function to determine the season based on the date
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'fall'

# Function to get productivity factor based on the season
def get_productivity_factor(season):
    if season == 'winter':
        return 0.1
    elif season == 'spring':
        return 0.5
    elif season == 'summer':
        return 1.0
    elif season == 'fall':
        return 0.7

# Bat migration algorithm
def simulate_bat_migration(lat, lon, date):
    """
    Simulate bat migration using an algorithm based on proximity to favorable regions, 
    seasonal variations, and random variability.
    """
    season = get_season(date)
    seasonal_factor = np.sin((date.month - 1) * np.pi / 6)  # Sinusoidal seasonal variation
    daily_variability = np.random.uniform(-1.0, 1.0)  # Increased daily variability factor
    random_lat_shift = np.random.uniform(-0.5, 0.5)  # Larger latitude shift for more noticeable changes
    random_lon_shift = np.random.uniform(-0.5, 0.5)  # Larger longitude shift for more noticeable changes

    # Adjust latitude and longitude based on migration factors
    if season == 'winter':
        migration_lat = lat - abs(random_lat_shift) + daily_variability  # Move to lower latitudes
    else:
        migration_lat = lat + random_lat_shift * seasonal_factor + daily_variability
    migration_lon = lon + random_lon_shift * seasonal_factor + daily_variability

    # Migration Factor (more impactful on Risk Score)
    migration_factor = np.clip(1 + seasonal_factor * np.random.uniform(-0.2, 0.2), 0.8, 1.2)

    return migration_lat, migration_lon, migration_factor, season

# Function to input the date manually
def get_user_date():
    while True:
        try:
            date_str = input("Enter the date (YYYY-MM-DD): ")
            date = datetime.strptime(date_str, "%Y-%m-%d")
            return date
        except ValueError:
            print("Invalid input! Please enter a date in the format YYYY-MM-DD.")

# Get user-selected date
selected_date = get_user_date()

# Simulate migrations for the selected date
migration_results = bat_data.apply(
    lambda row: simulate_bat_migration(row['GPS lat'], row['GPS long'], selected_date), axis=1
)
migrated_data = bat_data.copy()
migrated_data[['GPS lat', 'GPS long', 'Migration Factor', 'Season']] = pd.DataFrame(
    migration_results.tolist(), index=bat_data.index
)

# Adjust risk score based on productivity factor
migrated_data['Productivity Factor'] = migrated_data['Season'].apply(get_productivity_factor)
migrated_data['Risk Score'] = migrated_data['Average'] * migrated_data['Migration Factor'] * migrated_data['Productivity Factor']
migrated_data['Risk Score'] = migrated_data['Risk Score'].fillna(migrated_data['Risk Score'].mean())

# Normalize data for clustering
scaler = StandardScaler()
features = ['GPS lat', 'GPS long', 'Risk Score']

# Perform normalization
normalized_data = scaler.fit_transform(migrated_data[features])

# Impute missing values in the normalized data
imputer = SimpleImputer(strategy='mean')
normalized_data = imputer.fit_transform(normalized_data)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)

# Perform clustering on the imputed normalized data
migrated_data['Cluster'] = kmeans.fit_predict(normalized_data)

# Normalize the risk scores to a fixed range for consistent color scaling
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
migrated_data['Normalized Risk Score'] = min_max_scaler.fit_transform(migrated_data[['Risk Score']])

# Sample data based on the season to adjust the density of the heatmap
season = get_season(selected_date)
if season == 'winter':
    sampled_data = migrated_data.sample(frac=0.2, random_state=42)
elif season in ['spring', 'fall']:
    sampled_data = migrated_data.sample(frac=0.7, random_state=42)
else:  # summer
    sampled_data = migrated_data

# Generate a CSV file for high-risk zones based on bat migrations and risk scores
high_risk_zones = sampled_data[['GPS lat', 'GPS long', 'Normalized Risk Score']]
high_risk_zones.to_csv(f'ebola_risk_zones_{selected_date.strftime("%Y_%m_%d")}.csv', index=False)

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
country_risk = high_risk_zones.groupby('Country')['Normalized Risk Score'].max().reset_index()

# Plot the heatmap using Plotly
fig = px.density_mapbox(
    high_risk_zones, 
    lat='GPS lat', 
    lon='GPS long', 
    z='Normalized Risk Score', 
    radius=20,
    center=dict(lat=0, lon=20),  # Center the map around equatorial Africa
    zoom=2,
    mapbox_style="open-street-map",
    title=f"Ebola Risk Heatmap for {selected_date.strftime('%Y-%m-%d')}",
    color_continuous_scale=px.colors.sequential.Viridis  # Change color scale to keep map consistent
)

# Show the map
fig.show()
