# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
#
# # Initialize Spark
# spark = SparkSession.builder \
#     .appName("Valid MMSI Filter") \
#     .getOrCreate()
#
# # Load your AIS dataset (adjust the path as needed)
# df = spark.read.csv("/home/martin/p4/deployment/aisdk-2024-06-04_fishing_labeled.csv", header=True, inferSchema=True)
#
# # Columns to check for nulls
# required_columns = [
#     "Latitude", "Longitude", "ROT", "SOG", "COG",
#     "Heading", "Width", "Length", "Draught"
# ]
#
# # Filter out rows with nulls in any of the required columns
# non_null_df = df
# for col_name in required_columns:
#     non_null_df = non_null_df.filter(col(col_name).isNotNull())
#
# # Select MMSI, Latitude, and Longitude, and drop duplicates by MMSI
# unique_mmsi_df = non_null_df.select("MMSI", "Latitude", "Longitude").dropDuplicates(["MMSI"])
#
# # Show the result
# unique_mmsi_df.show(100,truncate=False)
#
# # Optionally save to file:
# # unique_mmsi_df.write.csv("/path/to/valid_mmsi_output.csv", header=True)



import pandas as pd
from shapely.geometry import Point
from zone_check import load_cable_lines, build_buffered_zone
import os

# === CONFIG ===
AIS_CSV = "/home/martin/p4/deployment/aisdk-2025-02-22_fishing_labeled.csv"
CABLE_CSV = "/home/martin/p4/deployment/data/cable_coordinates.csv"
BUFFER_METERS = 2200  # ≈ 0.87 NM
OUTPUT_PATH = "zone_crossers.txt"

# === Required columns ===
required_columns = [
    "Latitude", "Longitude", "ROT", "SOG", "COG",
    "Heading", "Width", "Length", "Draught"
]

print(f"🔍 Loading AIS data from {AIS_CSV}")
df = pd.read_csv(AIS_CSV)

# === Drop rows with missing required columns ===
df_clean = df.dropna(subset=required_columns)
valid_mmsis = df_clean["MMSI"].unique().tolist()
print(f"✅ Found {len(valid_mmsis)} valid MMSIs with complete data")

# === Load cables and buffer zone ===
print(f"📡 Loading cable lines from {CABLE_CSV}")
cable_lines = load_cable_lines(CABLE_CSV)
buffered_zone = build_buffered_zone(cable_lines, buffer_meters=BUFFER_METERS)

# === Check zone crossing ===
violators = []

print("🚨 Checking zone violations...")
for mmsi in valid_mmsis:
    vessel_df = df_clean[df_clean["MMSI"] == mmsi]

    for _, row in vessel_df.iterrows():
        point = Point(row["Longitude"], row["Latitude"])
        if buffered_zone.contains(point):
            print(f"⚠️  MMSI {mmsi} entered the critical zone at {row['# Timestamp']}")
            violators.append(mmsi)
            break  # no need to check further once vessel is flagged

# === Save violators ===
if violators:
    with open(OUTPUT_PATH, "w") as f:
        for m in sorted(set(violators)):
            f.write(f"{m}\n")
    print(f"✅ Saved {len(set(violators))} zone-crossing MMSIs to {OUTPUT_PATH}")
else:
    print("✅ No vessels crossed into critical zone.")

