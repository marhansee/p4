from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os

# Create Spark session
spark = SparkSession.builder \
    .appName("EnhanceAISCSVWithFutureCoordinates") \
    .config("spark.driver.memory", "100g") \
    .getOrCreate()

# Define folders
input_folder = "/ceph/project/gatehousep4/data/train"
output_folder = "/ceph/project/gatehousep4/data/train_labeled"

os.makedirs(output_folder, exist_ok=True)

# Loop over CSV files
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)
        print(f"Processing {input_path}")

        # Read CSV
        df = spark.read.option("header", True).option("inferSchema", True).csv(input_path)

        # Fix Timestamp format
        df = df.withColumnRenamed("# Timestamp", "Timestamp")
        df = df.withColumn("Timestamp", F.to_timestamp("Timestamp"))

        # Sort by MMSI and Timestamp
        window = Window.partitionBy("MMSI").orderBy("Timestamp")

        # Add future shifted coordinates
        df = df.withColumn("future_lat_10", F.lead("Latitude", 10).over(window))
        df = df.withColumn("future_lon_10", F.lead("Longitude", 10).over(window))
        df = df.withColumn("future_lat_20", F.lead("Latitude", 20).over(window))
        df = df.withColumn("future_lon_20", F.lead("Longitude", 20).over(window))

        # Write enriched CSV back
        output_path = os.path.join(output_folder, filename)
        df.write.option("header", True).csv(output_path, mode="overwrite")

print("✅ All CSVs enriched and saved.")
