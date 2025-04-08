from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col
import missingno as msno
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
end_date = "2024-12-3"

# Filter the data for the specified week
df_filtered = df.filter((col("Timestamp") >= start_date) & (col("Timestamp") < end_date))

msno.bar(df_filtered.toPandas())
plt.savefig("msno_bar.png")

msno.matrix(df_filtered.toPandas())
plt.savefig("msno_matrix.png")

msno.heatmap(df_filtered.toPandas())
plt.savefig("msno_heatmap.png")