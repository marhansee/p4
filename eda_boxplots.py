import matplotlib.pyplot as plt
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


# Define the date range for one week (e.g., December 1, 2024 to December 8, 2024, where the end date is exclusive)
start_date = "2024-12-01"
end_date = "2024-12-31"

# Filter the data for the specified week
df_filtered = df.filter((col("Timestamp") >= start_date) & (col("Timestamp") < end_date))

# Choose the column you want to plot
col = "latitude"

# Compute the minimum and maximum for the column
min_val = df_filtered.agg({col: "min"}).collect()[0][0]
max_val = df_filtered.agg({col: "max"}).collect()[0][0]

# Compute approximate quantiles: 25th, 50th, and 75th percentiles
# The third parameter (relative error) can be adjusted; lower error means more computation
q1, med, q3 = df_filtered.approxQuantile(col, [0.25, 0.5, 0.75], 0.01)

# Create a dictionary for the boxplot statistics
box_stats = {
    'med': med,
    'q1': q1,
    'q3': q3,
    'whislo': min_val,  # Lower whisker (min)
    'whishi': max_val,  # Upper whisker (max)
    'fliers': []        # Optionally, you could compute outliers; empty list here means none are drawn
}

# Create the boxplot using bxp. Note that bxp() accepts a list of dicts (one for each box).
plt.figure(figsize=(8, 6))
plt.bxp([box_stats], showfliers=True)
plt.title(f"Boxplot for {col}")
plt.ylabel(col)
plt.savefig(f"boxplot{col}.png", dpi=300)