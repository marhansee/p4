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

# Command-line args

parser = argparse.ArgumentParser(description="Convert prod-ready CSVs to Petastorm format.")
parser.add_argument('--mode', choices=['train', 'test'], required=True,
                    help="Mode: 'train' or 'test'")
args = parser.parse_args()

# Paths by mode

if args.mode == 'train':
    input_folder = "/ceph/project/gatehousep4/data/train_labeled"
    output_base = "/ceph/project/gatehousep4/data/petastorm/train"
elif args.mode == 'test':
    input_folder = "/ceph/project/gatehousep4/data/test_labeled"
    output_base = "/ceph/project/gatehousep4/data/petastorm/test"
else:
    raise ValueError("Invalid --mode, must be 'train' or 'test'")

os.makedirs(output_base, exist_ok=True)

# Auto-versioning

existing = [int(d[1:]) for d in os.listdir(output_base)
            if d.startswith('v') and d[1:].isdigit()]
next_v = max(existing) + 1 if existing else 1
output_path = f"{output_base}/v{next_v}"
print(f"Saving new Petastorm dataset to: {output_path}")

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
for i in range(1, 21):
    fields.append(UnischemaField(f'future_lat_{i}', np.float32, (), ScalarCodec(FloatType()), True))
    fields.append(UnischemaField(f'future_lon_{i}', np.float32, (), ScalarCodec(FloatType()), True))

AISSchema = Unischema('AISSchema', fields)

# Spark session

spark = SparkSession.builder \
    .appName(f"Convert to Petastorm ({args.mode})") \
    .config("spark.sql.shuffle.partitions", "15") \
    .config("spark.driver.memory", "100g") \
    .getOrCreate()
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
               "file:///ceph/projAISSchemaect/gatehousep4/data/petastorm_cache")
spark.sparkContext.setLogLevel("ERROR")

# Read prod-ready CSVs

csv_path = f"{input_folder}/*.csv"
df = spark.read.option("header", True).option("inferSchema", True).csv(csv_path)

# Select fields & coalesce

df = df.select(list(AISSchema.fields.keys()))
df = df.coalesce(15)

# Materialize Petastorm dataset

with materialize_dataset(spark, f"file://{output_path}", AISSchema):
    df.write.mode('overwrite').parquet(f"file://{output_path}")

# Regenerate Parquet _metadata

fs_path = output_path
dataset = pq.ParquetDataset(fs_path)
schema = dataset.schema.to_arrow_schema()
pq.write_metadata(schema, os.path.join(fs_path, '_metadata'))
print(f"⛅ _metadata written at {fs_path}/_metadata")

# Human-readable metadata

num_rows = df.count()
meta = {
    "version":        next_v,
    "created_on":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "source_path":    csv_path,
    "output_path":    output_path,
    "schema":         [f.name for f in AISSchema.fields.values()],
    "num_partitions": df.rdd.getNumPartitions(),
    "num_rows":       num_rows,
}
meta_dir = os.path.join(output_base, "metadata")
os.makedirs(meta_dir, exist_ok=True)
meta_path = os.path.join(meta_dir, f"v{next_v}.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=4)
print(f" Human-readable metadata saved at {meta_path}")
print(f" Finished saving Petastorm dataset at: {output_path}")
