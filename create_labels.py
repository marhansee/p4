import os
import shutil
import glob
import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


# Argument parsing

parser = argparse.ArgumentParser(description="Add lagged features to AIS data.")
parser.add_argument('--mode', choices=['train', 'test'], required=True,
                    help="Mode: 'train' or 'test'")
args = parser.parse_args()


# Set folders by mode
if args.mode == 'train':
    input_folder = "/ceph/project/gatehousep4/data/train"
    output_folder = "/ceph/project/gatehousep4/data/train_labeled"
elif args.mode == 'test':
    input_folder = "/ceph/project/gatehousep4/data/test"
    output_folder = "/ceph/project/gatehousep4/data/test_labeled"
else:
    raise ValueError("Invalid mode")

os.makedirs(output_folder, exist_ok=True)

# Initialize Spark
spark = SparkSession.builder \
    .appName(f"AIS Labeling ({args.mode})") \
    .config("spark.sql.shuffle.partitions", "15") \
    .config("spark.local.dir", "/ceph/project/gatehousep4/data/petastorm_cache") \
    .config("spark.driver.memory", "100g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Process each CSV
def add_lagged_features(input_folder, output_folder):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f"No CSVs found in {input_folder}")
        return []

    output_files = []
    for path in csv_files:
        print(f"Processing {path}...")
        df = spark.read.option("header", True).option("inferSchema", True).csv(path)

        # Rename and parse timestamp (format dd/MM/yyyy HH:mm:ss)
        df = df.withColumnRenamed("# Timestamp", "ts_raw")
        df = df.withColumn(
            "ts_parsed",
            F.to_timestamp("ts_raw", "dd/MM/yyyy HH:mm:ss")
        )
        # Drop rows that failed parsing
        df = df.filter(F.col("ts_parsed").isNotNull())
        # Add epoch seconds as numeric
        df = df.withColumn(
            "timestamp_epoch",
            F.unix_timestamp("ts_parsed").cast("long")
        )
        # Drop raw columns
        df = df.drop("ts_raw", "ts_parsed")

        # Sort by MMSI + epoch for windowing
        window = Window.partitionBy("MMSI").orderBy("timestamp_epoch")

        # Add future‐lag features for lat/lon
        for i in range(1, 21):
            df = df.withColumn(f"future_lat_{i}", F.lead("Latitude", i).over(window))
            df = df.withColumn(f"future_lon_{i}", F.lead("Longitude", i).over(window))

        # Write out a single "prod_ready.csv"
        base = os.path.basename(path).replace("_fishing_labeled.csv", "_prod_ready.csv")
        temp_dir = os.path.join(output_folder, base + "_tmp")
        (df.coalesce(1)
           .write.option("header", True)
           .mode("overwrite")
           .csv(temp_dir)
        )
        # Move and clean up
        tmp_csv = glob.glob(os.path.join(temp_dir, "*.csv"))[0]
        dest = os.path.join(output_folder, base)
        shutil.move(tmp_csv, dest)
        shutil.rmtree(temp_dir)

        print(f"Saved labeled file to {dest} (rows: {df.count()})")
        output_files.append(dest)

    return output_files

# Run
labeled = add_lagged_features(input_folder, output_folder)
print(f"Completed processing {len(labeled)} files.")

spark.stop()
