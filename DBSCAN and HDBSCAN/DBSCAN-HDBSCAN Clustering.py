import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan

# ----- Geographical Tools -----
import geopandas as gpd
import contextily as ctx

# ----- Suppress Warnings -----
import warnings
warnings.filterwarnings("ignore")

# ----- Other Libraries -----
import requests
import zipfile
import io
import os

# ==========================================================
# Download Canada Basemap
# ==========================================================

CANADA_MAP_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "YcUk-ytgrPkmvZAh5bf7zA/Canada.zip"
)

OUTPUT_DIR = "./"

os.makedirs(OUTPUT_DIR, exist_ok=True)

response = requests.get(CANADA_MAP_URL)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:

    for file_name in zip_ref.namelist():

        if file_name.endswith(".tif"):

            zip_ref.extract(file_name, OUTPUT_DIR)
            print(f"Downloaded and extracted: {file_name}")


# ==========================================================
# Plot Function
# ==========================================================

def plot_clustered_locations(df, title="Museums Clustered by Proximity"):
    """
    Plot clustered museum locations on a map of Canada.

    Parameters
    ----------
    df : pandas.DataFrame

        Must contain:
            Latitude
            Longitude
            Cluster

    title : str

        Plot title.
    """

    # Convert DataFrame into GeoDataFrame
    # EPSG:4326 = GPS latitude/longitude coordinates
    geo_df = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["Longitude"],
            df["Latitude"],
            crs="EPSG:4326"
        )
    )

    # Convert to Web Mercator so it aligns correctly
    # with the Canada basemap.
    geo_df = geo_df.to_crs(epsg=3857)

    # Create Figure
    fig, ax = plt.subplots(figsize=(15, 10))

    # Separate clustered points from noise
    non_noise = geo_df[geo_df["Cluster"] != -1]
    noise = geo_df[geo_df["Cluster"] == -1]

    # Plot noise
    noise.plot(
        ax=ax,
        color="black",
        edgecolor="red",
        markersize=20,
        alpha=1,
        label="Noise",
    )

    # Plot clusters
    non_noise.plot(
        ax=ax,
        column="Cluster",
        cmap="tab10",
        markersize=30,
        edgecolor="black",
        alpha=0.5,
        legend=False,
    )

    # Add Canada basemap
    ctx.add_basemap(
        ax,
        source="./Canada.tif",
        zoom=4,
    )

    # Formatting
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# ==========================================================
# Load Dataset
# ==========================================================

DATASET_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv"
)

museum_df = pd.read_csv(
    DATASET_URL,
    encoding="ISO-8859-1",
)

# ==========================================================
# Explore Dataset
# ==========================================================

print(museum_df.head())
print(museum_df.info())
print(museum_df.isnull().sum())

print(museum_df["Source_Facility_Type"].value_counts())
print(museum_df["ODCAF_Facility_Type"].value_counts())

# ==========================================================
# Keep Only Museums
# ==========================================================

museum_df = museum_df[
    museum_df["ODCAF_Facility_Type"] == "museum"
]

print(museum_df["ODCAF_Facility_Type"].value_counts())

# Keep only coordinates
museum_df = museum_df[
    ["Latitude", "Longitude"]
]

print(museum_df.info())

# Remove invalid coordinates
museum_df = museum_df[
    (museum_df["Latitude"] != "..")
    &
    (museum_df["Longitude"] != "..")
]

# Convert coordinates to float
museum_df[["Latitude", "Longitude"]] = museum_df[
    ["Latitude", "Longitude"]
].astype(float)

# ==========================================================
# Prepare Coordinates for DBSCAN
# ==========================================================

coords_scaled = museum_df.copy()

# Latitude ranges approximately from -90° to 90°
# Longitude ranges approximately from -180° to 180°.
#
# Multiplying latitude by 2 gives both axes
# a more comparable numerical scale before
# Euclidean distance is calculated.

coords_scaled["Latitude"] *= 2

# ==========================================================
# DBSCAN
# ==========================================================

DBSCAN_EPS = 1.0
DBSCAN_MIN_SAMPLES = 3
METRIC = "euclidean"

dbscan = DBSCAN(
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES,
    metric=METRIC,
)

museum_df["Cluster"] = dbscan.fit_predict(coords_scaled)

print("\nDBSCAN Cluster Counts")
print(museum_df["Cluster"].value_counts().sort_index())

plot_clustered_locations(
    museum_df,
    title="Museums Clustered by Proximity (DBSCAN)",
)

# ==========================================================
# HDBSCAN
# ==========================================================

HDBSCAN_MIN_CLUSTER_SIZE = 8

hdb = hdbscan.HDBSCAN(
    min_samples=None,
    min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
    metric=METRIC,
)

museum_df["Cluster"] = hdb.fit_predict(coords_scaled)

print("\nHDBSCAN Cluster Counts")
print(museum_df["Cluster"].value_counts().sort_index())

plot_clustered_locations(
    museum_df,
    title="Museums Clustered by Proximity (HDBSCAN)",
)