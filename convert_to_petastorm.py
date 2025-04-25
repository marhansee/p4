from pyspark.sql import SparkSession
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField
from petastorm.codecs import ScalarCodec
import json
from datetime import datetime
import numpy as np
from pyspark.sql.types import FloatType, IntegerType, StringType
import os

AISSchema = Unischema('AISSchema', [
    UnischemaField('Timestamp', np.string_, (), ScalarCodec(StringType()), False),
    UnischemaField('MMSI', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('Latitude', np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Longitude', np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('ROT', np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('SOG', np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('COG', np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Heading', np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Width', np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Length', np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Draught', np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Gear_Type', np.string_, (), ScalarCodec(StringType()), False),
    UnischemaField('trawling', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('future_lat_10', np.float32, (), ScalarCodec(FloatType()), True),
    UnischemaField('future_lon_10', np.float32, (), ScalarCodec(FloatType()), True),
    UnischemaField('future_lat_20', np.float32, (), ScalarCodec(FloatType()), True),
    UnischemaField('future_lon_20', np.float32, (), ScalarCodec(FloatType()), True),
])

# Auto-versioning logic
base_path = "/ceph/project/gatehousep4/data/petastorm/train"
os.makedirs(base_path, exist_ok=True)
existing_versions = []

for folder_name in os.listdir(base_path):
    if folder_name.startswith("v") and folder_name[1:].isdigit():
        version_number = int(folder_name[1:])
        existing_versions.append(version_number)

if existing_versions:
    next_version = max(existing_versions) + 1
else:
    next_version = 1

output_path = f"{base_path}/v{next_version}"

print(f"Saving new Petastorm dataset to: {output_path}")

# Start Spark
spark = SparkSession.builder \
    .appName("ConvertAISDataToPetastorm") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "100g") \
    .config(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///ceph/project/gatehousep4/data/petastorm_cache") \
    .getOrCreate()

# Load labeled CSVs
input_folder = "/ceph/project/gatehousep4/data/train_labeled"
csv_path = f"{input_folder}/*.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(csv_path)

# Rename columns
df = df.withColumnRenamed("# Timestamp", "Timestamp") \
       .withColumnRenamed("Gear Type", "Gear_Type")

# Merge partitions
df = df.coalesce(8)

# Materialize dataset
with materialize_dataset(spark, f"file://{output_path}", AISSchema, parquet_row_group_size_mb=256):
    df.write.mode('overwrite').parquet(f"file://{output_path}")

# Count total rows (this triggers a small Spark action)
num_rows = df.count()

# Build metadata dictionary
metadata = {
    "version": next_version,
    "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "source_path": csv_path,
    "output_path": output_path,
    "schema": [field.name for field in AISSchema.fields],
    "num_partitions": df.rdd.getNumPartitions(),
    "num_rows": num_rows
}

# Write metadata.json
metadata_path = os.path.join(output_path.replace("file://", ""), "metadata.json")

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"✅ Metadata saved at {metadata_path}")
print(f"✅ Finished saving Petastorm dataset at: {output_path}")
