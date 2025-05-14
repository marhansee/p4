import json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import os
from pyspark.sql import SparkSession, functions as F
import argparse

parser = argparse.ArgumentParser(description="Process AIS data with resampling, interpolation, and lag generation.")
parser.add_argument('--version', type=str, required=True, help="Version name, e.g. v1, v2")

args = parser.parse_args()

base_input = f"/ceph/project/gatehousep4/data/train_labeled/{args.version}"
base_output = f"/ceph/project/gatehousep4/data/norm_stats/{args.version}"

os.makedirs(base_output, exist_ok=True)

output_stats_file = os.path.join(base_output, "train_norm_stats.json")
# Continuous columns to normalize
continuous_cols = ["Latitude", "Longitude", "ROT", "SOG", "COG", "Heading", "Draught","Width", "Length"]

# Initialize Spark
spark = SparkSession.builder \
    .appName(f"Normalization stats") \
    .config("spark.sql.shuffle.partitions", "128") \
    .config("spark.local.dir", "/ceph/project/gatehousep4/data/petastorm_cache") \
    .config("spark.driver.memory", "180g") \
    .getOrCreate()


schema = StructType([
    StructField("MMSI", IntegerType(), True),
    StructField("# Timestamp", StringType(), True),
    StructField("Type of mobile", StringType(), True),
    StructField("Latitude", DoubleType(), True),
    StructField("Longitude", DoubleType(), True),
    StructField("Navigational status", StringType(), True),
    StructField("ROT", DoubleType(), True),
    StructField("SOG", DoubleType(), True),
    StructField("COG", DoubleType(), True),
    StructField("Heading", DoubleType(), True),
    StructField("IMO", StringType(), True),
    StructField("Callsign", StringType(), True),
    StructField("Name", StringType(), True),
    StructField("Ship type", StringType(), True),
    StructField("Cargo type", StringType(), True),
    StructField("Width", IntegerType(), True),
    StructField("Length", IntegerType(), True),
    StructField("Type of position fixing device", StringType(), True),
    StructField("Draught", DoubleType(), True),
    StructField("Destination", StringType(), True),
    StructField("ETA", StringType(), True),
    StructField("Data source type", StringType(), True),
    StructField("A", IntegerType(), True),
    StructField("B", IntegerType(), True),
    StructField("C", IntegerType(), True),
    StructField("D", IntegerType(), True),
    StructField("Gear Type", StringType(), True),
    StructField("trawling", IntegerType(), True),
])

# Read and union all training CSVs
df = spark.read \
    .option("recursiveFileLookup", "true") \
    .schema(schema) \
    .parquet(base_input)

# Compute mean and stddev for continuous columns
stats_exprs = []
for col in continuous_cols:
    stats_exprs.append(F.mean(col).alias(f"{col}_mean"))
    stats_exprs.append(F.stddev(col).alias(f"{col}_std"))

stats_row = df.select(*stats_exprs).collect()[0]

# Format stats into a dictionary
norm_stats = {
    col: {
        "mean": stats_row[f"{col}_mean"],
        "std": stats_row[f"{col}_std"]
    }
    for col in continuous_cols
}

# Save to JSON file
with open(output_stats_file, "w") as f:
    json.dump(norm_stats, f, indent=2)

print(f"Normalization stats saved to {output_stats_file}")
spark.stop()
