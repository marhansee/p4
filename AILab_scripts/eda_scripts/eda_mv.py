from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
from pyampute.exploration.mcar_statistical_tests import MCARTest


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

# Define the date range (note: your end_date is exclusive)
start_date = "2024-12-01"
end_date = "2024-12-03"

# Filter the data for the specified date range
df_filtered = df.filter((col("Timestamp") >= start_date) & (col("Timestamp") < end_date))

# Workaround:
# 1. Convert the Timestamp column to a string to avoid conversion issues during toPandas()
df_filtered = df_filtered.withColumn("Timestamp", col("Timestamp").cast("string"))
# 2. Convert the Spark DataFrame to a Pandas DataFrame
pd_df = df_filtered.toPandas()
# 3. Convert the Timestamp column back to a proper datetime type with explicit unit (datetime64[ns])
# Adjust the format string if the timestamp string includes microseconds (e.g., use "%Y-%m-%d %H:%M:%S.%f")
pd_df["Timestamp"] = pd.to_datetime(pd_df["Timestamp"], format="%Y-%m-%d %H:%M:%S")

# Perform MCARTest

mt = MCARTest(method="little")
print(mt.little_mcar_test(pd_df))

# Create Missingno plots
# msno.bar(pd_df)
# plt.savefig("msno_bar.png")
# plt.clf()
#
# msno.matrix(pd_df)
# plt.savefig("msno_matrix.png")
# plt.clf()
#
# msno.heatmap(pd_df)
# plt.savefig("msno_heatmap.png")
# plt.clf()
