from pyspark.sql import SparkSession
from petastorm.spark import SparkDatasetConverter, make_spark_converter

# Configure Spark
spark = SparkSession.builder \
    .appName("ConvertAISDataToPetastorm") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///tmp/petastorm_cache") \
    .getOrCreate()

# Path to your training CSVs
csv_path = "/ceph/project/gatehousep4/data/train*.csv"

# Load and preprocess CSVs
df = spark.read.option("header", True).option("inferSchema", True).csv(csv_path)

# Choosing relevant features
df = df.select("# Timestamp", "MMSI", "Latitude", "Longitude","ROT", "SOG", "COG", "Heading","Width","Length","Draught", "Gear Type", "trawling")  # adjust as needed

# Write to Petastorm-compatible Parquet
output_path = "file:///ceph/project/gatehousep4/data/train"

# Create and cache the converter
converter = make_spark_converter(df)
converter._dataset_path = output_path
