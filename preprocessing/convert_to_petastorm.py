import os
import json
import argparse
from datetime import datetime
import numpy as np
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from petastorm.spark import SparkDatasetConverter
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField
from petastorm.codecs import ScalarCodec
from pyspark.sql.types import FloatType, IntegerType, LongType

# Command line arguments

parser = argparse.ArgumentParser(description="Convert prod-ready Parquet folders to a single Petastorm dataset.")
parser.add_argument('--mode', choices=['train', 'test','val'], required=True,
                    help="Mode: 'train', 'val' or 'test'")
parser.add_argument('--version', type=str, required=True,
                    help="Target version (e.g., v1, v2, v3). Must start with 'v' followed by a number.")
args = parser.parse_args()

if not (args.version.startswith('v') and args.version[1:].isdigit()):
    raise ValueError("Invalid --version format. Must be like 'v1', 'v2', etc.")

# Paths based on arguments

if args.mode == 'train':
    input_folder = f"/ceph/project/gatehousep4/data/train_labeled/{args.version}"
    output_path = f"/ceph/project/gatehousep4/data/petastorm/train/{args.version}"
elif args.mode == 'test':
    input_folder = f"/ceph/project/gatehousep4/data/test_labeled/{args.version}"
    output_path = f"/ceph/project/gatehousep4/data/petastorm/test/{args.version}"
elif args.mode == 'val':
    input_folder = f"/ceph/project/gatehousep4/data/val_labeled/{args.version}"
    output_path = f"/ceph/project/gatehousep4/data/petastorm/val/{args.version}"
else:
    raise ValueError("Invalid --mode")

os.makedirs(output_path, exist_ok=True)

# Define Unischema

fields = [
    UnischemaField('timestamp_epoch', np.int64, (), ScalarCodec(LongType()), False),
    UnischemaField('MMSI',            np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('Latitude',        np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Longitude',       np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('ROT',             np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('SOG',             np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('COG',             np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Heading',         np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Width',           np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Length',          np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('Draught',         np.float32, (), ScalarCodec(FloatType()), False),
    UnischemaField('trawling',        np.int32,   (), ScalarCodec(IntegerType()), False),
]
for i in range(1, 121):
    fields.append(UnischemaField(f'future_lat_{i}', np.float32, (), ScalarCodec(FloatType()), True))
    fields.append(UnischemaField(f'future_lon_{i}', np.float32, (), ScalarCodec(FloatType()), True))

AISSchema = Unischema('AISSchema', fields)

# Init Spark Session

spark = SparkSession.builder \
    .appName(f"Convert to Petastorm ({args.mode}, {args.version})") \
    .config("spark.sql.shuffle.partitions", "128") \
    .config("spark.driver.memory", "100g") \
    .getOrCreate()

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
               "file:///ceph/project/gatehousep4/data/petastorm_cache")
spark.sparkContext.setLogLevel("ERROR")

# Load Input

subdirs = [os.path.join(input_folder, d) for d in os.listdir(input_folder)
           if d.endswith("_prod_ready") and os.path.isdir(os.path.join(input_folder, d))]

if not subdirs:
    raise FileNotFoundError(f"No *_prod_ready folders found in {input_folder}")

print(f"Reading from {len(subdirs)} folders...")
df = spark.read.parquet(*subdirs)

# ----------------- Safe Column Selection -----------------

expected_cols = list(AISSchema.fields.keys())
actual_cols = df.columns
safe_cols = [col for col in expected_cols if col in actual_cols]
missing = set(expected_cols) - set(actual_cols)
if missing:
    print(f"Warning: Missing {len(missing)} expected columns: {sorted(missing)[:5]}...")

df = df.select(safe_cols)

# Optional: Repartition

df = df.repartition("MMSI")

# Write Petastorm Dataset

with materialize_dataset(spark, f"file://{output_path}", AISSchema):
    df.write.mode('overwrite').parquet(f"file://{output_path}")

# Write _metadata

dataset = pq.ParquetDataset(output_path)
schema = dataset.schema.to_arrow_schema()
pq.write_metadata(schema, os.path.join(output_path, '_metadata'))
print(f"_metadata written at {output_path}/_metadata")

# ----------------- Save Human-Readable Metadata -----------------

num_rows = df.count()
meta = {
    "version":        args.version,
    "created_on":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "source_folders": subdirs,
    "output_path":    output_path,
    "schema":         safe_cols,
    "num_partitions": df.rdd.getNumPartitions(),
    "num_rows":       num_rows,
}
meta_dir = os.path.join(output_path, "metadata")
os.makedirs(meta_dir, exist_ok=True)
meta_path = os.path.join(meta_dir, f"{args.version}.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=4)

print(f"Petastorm dataset saved at: {output_path}")
print(f"Metadata written to: {meta_path}")
