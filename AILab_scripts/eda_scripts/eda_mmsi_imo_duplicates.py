from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col
from pyspark.sql.functions import col, countDistinct

# Create a SparkSession
spark = (SparkSession.builder.appName("AIS_EDA")
         .config("spark.driver.memory", "100g")
         .getOrCreate())

# Load the AIS data from multiple CSV files in the "data" folder
df = spark.read.csv("data/*.csv", header=True, inferSchema=True)

# Rename the column '# Timestamp' to 'Timestamp'
df = df.withColumnRenamed("# Timestamp", "Timestamp")

# Convert the 'Timestamp' column to a proper timestamp type using the correct format
df = df.withColumn("Timestamp", to_timestamp("Timestamp", "dd/MM/yyyy HH:mm:ss"))

# Verify the conversion by showing a sample of the timestamp column
df.select("Timestamp").show(5, False)

# Define the date range for one week (e.g., December 1, 2024 to December 8, 2024, where the end date is exclusive)
start_date = "2024-12-01"
end_date = "2024-12-31"

# Filter the data for the specified week
df_filtered = df.filter((col("Timestamp") >= start_date) & (col("Timestamp") < end_date))


# Find unique IMO count per MMSI
mmsi_imo_counts = df_filtered.groupBy("MMSI").agg(countDistinct("IMO").alias("unique_imo_count"))

filtered_count = mmsi_imo_counts.where(col("unique_imo_count") > 1)
# Show the results
filtered_count.show(truncate=False)