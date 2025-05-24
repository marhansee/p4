from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import to_timestamp, lag, unix_timestamp, col
import matplotlib.pyplot as plt
import pandas as pd

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

# Define window specification partitioned by MMSI and ordered by Timestamp
window_spec = Window.partitionBy("MMSI").orderBy("Timestamp")

# Compute previous timestamp and time difference on the filtered data
df_filtered = df_filtered.withColumn("Previous_timestamp", lag("Timestamp").over(window_spec))
df_filtered = df_filtered.withColumn("Time_difference",
                                     unix_timestamp(col("Timestamp")) - unix_timestamp(col("Previous_timestamp")))

threshold = 30

df_filtered_typical = df_filtered.filter(col("Time_difference") <= threshold)
time_diff_list = df_filtered_typical.select("Time_difference").dropna().rdd.flatMap(lambda x: x).collect()

# Convert the list to a Pandas DataFrame

time_diff_pd = pd.DataFrame(time_diff_list, columns=["Time_difference"])

print(time_diff_pd.describe())
# Now plot the histogram using matplotlib

plt.figure(figsize=(10,6))
plt.hist(time_diff_pd["Time_difference"], bins=30)
plt.xlabel('Time difference in seconds')
plt.ylabel('Frequency')
plt.title('Histogram of Inter-message Time Intervals')
plt.savefig("hist_time_intervals.png", dpi=300)
plt.close()