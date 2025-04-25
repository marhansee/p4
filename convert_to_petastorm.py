from pyspark.sql import SparkSession
from petastorm.spark import SparkDatasetConverter, make_spark_converter

# Configure Spark
spark = SparkSession.builder \
    .appName("ConvertAISDataToPetastorm") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "100g") \
    .config(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///ceph/project/gatehousep4/data/petastorm_cache") \
    .getOrCreate()

# Path to your training CSVs
csv_path = "/ceph/project/gatehousep4/data/train/*.csv"

# Load and preprocess CSVs
df = spark.read.option("header", True).option("inferSchema", True).csv(csv_path)

# Choosing relevant features
df = df.select("# Timestamp", "MMSI", "Latitude", "Longitude","ROT", "SOG", "COG", "Heading","Width","Length","Draught", "Gear Type", "trawling")  # adjust as needed

# Merge partitions

df = df.coalesce(8)

# Write to Petastorm-compatible Parquet
output_path = "file:///ceph/project/gatehousep4/data/petastorm/train"

# Create and cache the converter
converter = make_spark_converter(df)
converter._dataset_path = output_path
