from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col

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

# Keep multiple columns (adjust column names as needed)
# For example, we keep 'Timestamp', 'mmsi', and 'latitude'
# df_filtered = df_week.select("Timestamp","MMSI","Latitude","Longitude")

# Show a sample of the filtered DataFrame
df_filtered.show(5, False)

# Optional: Check for duplicate rows based on the selected columns
duplicates_df = df_filtered.groupBy(df_filtered.columns).count().filter("count > 1")
duplicates_df.show(truncate=False)

# Compare total vs distinct counts of the DataFrame
total_count = df_filtered.count()
distinct_count = df_filtered.distinct().count()
duplicates_count = total_count - distinct_count

print(f"Total rows in the data: {total_count}")
print(f"Distinct rows in the data: {distinct_count}")
print(f"Number of duplicate rows: {duplicates_count}")
print(f"Share of duplicate rows: {duplicates_count / total_count * 100:.2f}%")


