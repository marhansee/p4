from pyspark.sql import SparkSession
from petastorm.spark import SparkDatasetConverter, make_spark_converter

# Configure Spark
spark = SparkSession.builder \
    .appName("ConvertAISDataToPetastorm") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///tmp/petastorm_cache") \
    .getOrCreate()

# Path to your training CSVs
csv_path = "/project/data/train/*.csv"

# Load and preprocess CSVs
df = spark.read.option("header", True).option("inferSchema", True).csv(csv_path)

# Optional: filter or preprocess TILPAS
df = df.select("timestamp", "lat", "lon", "speed", "mmsi", "gear_type")  # adjust as needed

# Write to Petastorm-compatible Parquet
output_path = "file:///project/data/petastorm/train"

# Create and cache the converter
converter = make_spark_converter(df)
converter._dataset_path = output_path  # set output path manually
