from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col
import matplotlib.pyplot as plt

# Create a SparkSession
spark = (SparkSession.builder.appName("AIS_EDA")
         .config("spark.driver.memory", "100g")
         .config("spark.driver.maxResultSize", "100g")
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

# df_filtered.summary().show()

# Define columns of interest
columns = ["Longitude", "Latitude", "ROT", "SOG", "COG", "Heading", "Width", "Length", "Draught"]

# Loop through each column to calculate the IQR and outliers
for column in columns:
    # Calculate quantiles
    quantiles = df_filtered.approxQuantile(column, [0.25, 0.5, 0.75], 0)
    q1 = quantiles[0]
    median = quantiles[1]
    q3 = quantiles[2]

    # Calculate IQR and outlier bounds
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    print(f"{column}:")
    print(f"  Q1: {q1}, Median: {median}, Q3: {q3}")
    print(f"  IQR: {iqr}")
    print(f"  Lower bound: {lower_bound}")
    print(f"  Upper bound: {upper_bound}")
