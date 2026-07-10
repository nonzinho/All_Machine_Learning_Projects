import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler, RobustScaler # --> Testing 2 types of scaled data

# ----- Import Geographical Tools -----
import geopandas as gpd # --> Pandas dataframe-like geodataframes for geographical data
import contextily as ctx
from shapely.geometry import Point

# ----- Import Warnings -----
import warnings
warnings.filterwarnings('ignore')

# ----- Import Other Libraries -----
import requests
import zipfile
import io
import os

# ----- Fetch URL of Zipfile on the cloud server -----
zip_file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip' # --> Canada map for reference
output_dir = './' # --> Directory to save extracted TIF file
os.makedirs(output_dir, exist_ok=True)

# ----- Download Zip File -----
response = requests.get(zip_file_url)
response.raise_for_status() # --> Ensure the request was successful

# ---- Open Zip File in Memory -----
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    # --> Iterate over files in ZIP
    for file_name in zip_ref.namelist():
        if file_name.endswith('.tif'):
            zip_ref.extract(file_name, output_dir)
            print(f"Downloaded and extracted: {file_name}")

# ----- Write a Function That Plots Clustered Locations and Overlays Them On A Base Map -----
def plot_clustered_locations(df, title="Museums Clustered by Proximity"):
    # --> df : Dataframe containing parameters latitude, longtitude, and cluster columns, title (str) : Title of the plot.
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longtitude'], df['Latitude'], crs='EPSG:4326')) # --> Transform data into geographical coordinate reference system (standard is epsg:4326)
    gdf = gdf.to_crs(epsg=3857) 
    # --> 1) Transform normal Pandas Dataframe into a geographical coordinate reference system (EPSG:4326) : Define a position in the 3D sphere (Latitude, Longtitude).
    # --> 2) Project the position in the 3D sphere into 2D coordinates X and Y (EPSG=3857).
    
    # ----- Create The Plot ----
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # ----- Separate noise, non-noise from the data and plot clusters -----
    non_noise = gdf[gdf['Cluster'] != 1]
    noise = gdf[gdf['Cluster'] == 1]
    noise.plot(ax=ax, color='k', markersize=20, ec='r', alpha=1, label='Noise')
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.5)

    # ----- Add Basemap of Canada -----
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)

    # ----- Format The Plot -----
    plt.title(title)
    plt.xlabel('Longtitude')
    plt.ylabel('Latitude')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    # ---- Show The Plot -----
    plt.show() 

# ----- Load Dataset -----
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv'
df = pd.read_csv(url, encoding='ISO-8859-1') # --> encoding='ISO-8859-1' ensures no UnicodeDecodeError, but still, using UTF-8 is totally acceptable!

# ----- Exploring Data -----
print(df.head())
print(df.info())
print(df.isnull().sum()) # --> Pd doesn't count '..' as null value, but we can totally see it!

print(df['Source_Facility_Type'].value_counts()) # --> 2533 entries for '..' !!
print(df['ODCAF_Facility_Type'].value_counts())

df = df[df['ODCAF_Facility_Type'] == 'museum']
print(df['ODCAF_Facility_Type'].value_counts())

df = df[['Latitude','Longitude']]
print(df.info())
df = df[df['Latitude'] != '..']
df = df[df['Longitude'] != '..']
df[['Latitude', 'Longitude']] = df[['Latitude', 'Longitude']].astype('float')




