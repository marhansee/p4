import os
import shutil
import glob
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def add_lagged_feature(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return []

    print(f"Found {len(csv_files)} CSV files to process")
    output_files = []

    for data_path in csv_files:
        print(f"Processing {data_path}...")

        # Read CSV
        df = spark.read.option("header", True).option("inferSchema", True).csv(data_path)

        # Fix Timestamp format
        df = df.withColumnRenamed("# Timestamp", "Timestamp")
        df = df.withColumn("Timestamp", F.to_timestamp("Timestamp"))

        # Sort by MMSI and Timestamp
        window = Window.partitionBy("MMSI").orderBy("Timestamp")

        # Add future shifted coordinates
        for i in range(1, 21):
            df = df.withColumn(f"future_lat_{i}", F.lead("Latitude", i).over(window))
            df = df.withColumn(f"future_lon_{i}", F.lead("Longitude", i).over(window))

        base_filename = os.path.basename(data_path)
        new_file_name = base_filename.replace("_fishing_labeled.csv", "prod_ready.csv")
        output_path = os.path.join(output_folder, new_file_name)

        temp_dir = output_path + "_temp"

        (df.coalesce(1)
         .write
         .option("header", "true")
         .mode("overwrite")
         .csv(temp_dir))

        # Move the single part-xxx.csv to final output_path
        temp_csv_file = glob.glob(os.path.join(temp_dir, "*.csv"))[0]
        shutil.move(temp_csv_file, output_path)

        shutil.rmtree(temp_dir)

        count = df.count()
        print(f"Saved {count} records to {output_path}")
        output_files.append(output_path)

    return output_files

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("Fishing Vessel Data Processor") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.memory", "100g") \
        .getOrCreate()

    input_folder = "/ceph/project/gatehousep4/data/train"
    output_folder = "/ceph/project/gatehousep4/data/train_labeled"

    output_files = add_lagged_feature(input_folder, output_folder)

    print(f"Completed processing {len(output_files)} files.")

    spark.stop()
