from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col
from pyspark.sql.functions import col, countDistinct
import matplotlib.pyplot as plt

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

# List of features to plot:

columns_to_plot = ["Longitude", "Latitude","ROT","SOG","COG","Heading","Width","Length","Draught"]
for col in columns_to_plot:
    data = df_filtered.select(f"{col}").dropna().rdd.flatMap(lambda x: x).collect()
    plt.hist(data, bins=20)
    plt.xlabel(f'{col}')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Data Distribution of {col}')
    plt.savefig(f"hist_{col}.png", dpi=300)
    plt.close()
