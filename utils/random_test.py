from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as max_
from pyspark.sql.functions import sum as spark_sum

y = "/home/martin-birch/p4/aisdk-2024-03-01_prod_ready.csv"
x = "/home/martin-birch/p4/data/fishing_vessel_data/aisdk-2024-03-01_fishing_labeled.csv"
xy = "/home/martin/p4/data/aisdk-2024-03-01.csv"
# def main():
#     # 1. Create Spark session
#     spark = SparkSession.builder \
#         .appName("UniqueSOGCheckWithNullsAndLastTimestamp") \
#         .getOrCreate()
#
#
#
#     # 2. Load your data (adjust path & format as needed)
#     df = spark.read \
#         .format("csv") \
#         .option("header", "true") \
#         .load(y)
#     # /home/martin/p4/aisdk-2024-03-01_prod_ready.csv
#     # /home/martin/aisdk-2024-03-01_fishing_labeled.csv
#
#     # 3. Filter for the specific MMSI
#     # mmsi_val = "219004308"
#     # filtered = df.filter(col("MMSI") == mmsi_val)
#
#     # 4. Select distinct non-null SOG values
#     distinct_sog = df \
#         .filter(col("SOG").isNotNull()) \
#         .select("SOG") \
#         .distinct()
#
#     # 5. Show distinct SOGs and count
#     distinct_sog.show(truncate=False)
#     non_null_count = distinct_sog.count()
#     # print(f"Found {non_null_count} unique non-null SOG value(s) for MMSI {mmsi_val}.")
#
#     # 6. Find and print rows where SOG is NULL
#     null_rows = df.filter(col("SOG").isNull())
#     null_rows.show(truncate=False)
#     null_count = null_rows.count()
#     # print(f"Found {null_count} row(s) with NULL SOG for MMSI {mmsi_val}.")
#
#     # 7. Compute and print the last (maximum) timestamp_epoch
#     last_ts_row = df \
#         .select(max_("timestamp_epoch").alias("last_timestamp")) \
#         .collect()[0]
#     last_timestamp = last_ts_row["last_timestamp"]
#     # print(f"Last timestamp_epoch for MMSI {mmsi_val}: {last_timestamp}")
#
#     # (Optional) If you want to see the full row(s) having that timestamp:
#     last_rows = df \
#         .filter(col("timestamp_epoch") == last_timestamp)
#     print("Row(s) with that last timestamp:")
#     last_rows.show(truncate=False)
#
#     spark.stop()
#
# if __name__ == "__main__":
#     main()


from pyspark.sql import SparkSession
from pyspark.sql.functions import col


#
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
# from pyspark.sql import functions as F
#
# #
# # Start Spark session
# spark = SparkSession.builder \
#     .appName("UniqueSOGandROTValues") \
#     .getOrCreate()
#
# # Read CSV into DataFrame
# df = spark.read \
#     .option("header", "true") \
#     .csv(y)
# df = df.withColumn("ROT", F.col("ROT").cast("double"))
# # Cast SOG and ROT to integers
# df = df \
#     .withColumn("SOG", col("SOG").cast("int")) \
#     .withColumn("ROT", col("ROT"))
#
# # Collect unique SOG values
# unique_sog = [row.SOG for row in df.select("SOG").distinct().orderBy("SOG").collect()]
# print("Unique SOG values:", unique_sog)
#
# # Collect unique ROT values
# unique_rot = [row.ROT for row in df.select("ROT").distinct().orderBy("ROT").collect()]
# print("Unique ROT values:", unique_rot)
#
# df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).show()
#
# result = df.filter(F.col("ROT").isNotNull()).count() / df.count()
#
# df.select("ROT").printSchema()
# print(df.columns)
# print(result)
# # Stop Spark
# spark.stop()


#
# from pyspark.sql import SparkSession
#
# def save_first_100(input_path: str, output_path: str):
#     # 1. Start Spark
#     spark = SparkSession.builder \
#         .appName("SaveFirst100Rows") \
#         .getOrCreate()
#
#     # 2. Read the original CSV (adjust options as needed)
#     df = spark.read \
#         .format("csv") \
#         .option("header", "true") \
#         .load(input_path)
#
#     # 3. Take the first 100 rows
#     first_hundred = df.limit(100)
#
#     # 4. Write them to a new CSV (will create part files under output_path)
#     first_hundred.write \
#         .format("csv") \
#         .option("header", "true") \
#         .mode("overwrite") \
#         .save(output_path)
#
#     spark.stop()
#
# if __name__ == "__main__":
#     # example usage
#     save_first_100(
#         input_path="/home/martin/p4/aisdk-2024-03-01_prod_ready.csv",
#         output_path="/home/martin/p4/aisdk-2024-03-01_prod_ready100v1.csv"
#     )

# from pyspark.sql import SparkSession
#
# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("Unique MMSI Counter") \
#     .getOrCreate()
#
# # Load AIS data (replace with your actual file path)
# df = spark.read.csv("/home/martin/aisdk-2024-03-01_fishing_labeled.csv", header=True, inferSchema=True)
#
# # Count number of unique MMSIs
# unique_mmsi_count = df.select("MMSI").distinct().count()
#
# print(f"Number of unique MMSIs: {unique_mmsi_count}")
#
# # Stop Spark session
# spark.stop()

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, count, countDistinct, when
#
# # Start Spark
# spark = SparkSession.builder.appName("Check Null MMSI").getOrCreate()
#
# # Read your AIS data
# df = spark.read.csv(xy, header=True, inferSchema=True)
#
# # List of columns to check (excluding MMSI)
# feature_cols = [col for col in df.columns if col != "MMSI"]
#
# # For each feature, find MMSIs where that column is always null
# null_only_conditions = []
#
# for feature in feature_cols:
#     condition = (
#         df.groupBy("MMSI")
#         .agg(
#             count("*").alias("total"),
#             count(feature).alias(f"{feature}_non_null")
#         )
#         .filter(col(f"{feature}_non_null") == 0)
#         .select("MMSI")
#         .withColumn("feature", when(col("MMSI").isNotNull(), feature))
#     )
#     null_only_conditions.append(condition)
#
# # Union all MMSI-feature combinations where feature is always null
# from functools import reduce
# from pyspark.sql import DataFrame
#
# if null_only_conditions:
#     result_df = reduce(DataFrame.unionByName, null_only_conditions)
#     result_df = result_df.distinct()
#
#     # Count unique MMSIs
#     unique_mmsi_count = result_df.select("MMSI").distinct().count()
#     print(f"Number of unique MMSIs with at least one feature always null: {unique_mmsi_count}")
#
#     # Count MMSIs where ROT is always null
#     rot_null_mmsi_count = result_df.filter(col("feature") == "ROT").select("MMSI").distinct().count()
#     print(f"Number of MMSIs with only NULL values in ROT feature: {rot_null_mmsi_count}")
#
#     result_df.show(100, truncate=False)
# else:
#     print("No features with only nulls found for any MMSI.")
#
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
#
#
# def get_unique_rot(mmsi: str, input_path: str):
#     # 1. Start Spark session
#     spark = SparkSession.builder \
#         .appName("UniqueROTByMMSI") \
#         .getOrCreate()
#
#     # 2. Read the CSV file
#     df = spark.read \
#         .format("csv") \
#         .option("header", "true") \
#         .option("inferSchema", "true") \
#         .load(input_path)
#
#     # 3. Filter for the specified MMSI
#     filtered_df = df.filter(col("MMSI") == mmsi)
#
#     # 4. Select distinct non-null ROT values
#     unique_rot = filtered_df \
#         .filter(col("ROT").isNotNull()) \
#         .select("ROT") \
#         .distinct() \
#         .orderBy("ROT")
#
#     # 5. Collect and display results
#     rot_values = [row.ROT for row in unique_rot.collect()]
#
#     if rot_values:
#         print(f"Unique ROT values for MMSI {mmsi}: {rot_values}")
#     else:
#         print(f"No non-null ROT values found for MMSI {mmsi}.")
#
#     # 6. Stop Spark session
#     spark.stop()
#
#
# if __name__ == "__main__":
#     # Example usage
#     input_mmsi = input("Enter MMSI number: ")
#     input_path = "/home/martin/aisdk-2024-03-01_fishing_labeled.csv"
#     get_unique_rot(input_mmsi, input_path)

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
#
# # Initialize Spark session
# spark = SparkSession.builder.appName("UniqueMMSI").getOrCreate()
#
# # Load the CSV file into a DataFrame
# df = spark.read.option("header", "true").csv(xy)
#
# # Correctly filter the DataFrame where "Ship type" equals "Fishing"
# df = df.filter(col("Ship type") == "Fishing")
#
# # Count unique MMSI values
# unique_mmsi_count = df.select("MMSI").distinct().count()
#
# # Print the result
# print(f"Number of unique MMSI: {unique_mmsi_count}")
#
# unique_rot_values = df.select("ROT").distinct().orderBy("ROT").collect()
# print("Unique ROT values:")
# for row in unique_rot_values:
#     print(row.ROT)
#
#
# # Stop the Spark session
# spark.stop()
#
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, count
#
# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("Fishing MMSI and NULL ROT Count") \
#     .getOrCreate()
#
# # Path to your AIS CSV file
# csv_path = "/path/to/your/ais_data.csv"
#
# # Load the data
# df = spark.read.option("header", True).option("inferSchema", True).csv(y)
#
# # Filter rows where Ship Type is 'Fishing'
# # fishing_df = df.filter(col("Ship Type") == "Fishing")
#
# # Count unique MMSI numbers among fishing vessels
# unique_fishing_mmsi_count = df.select("MMSI").distinct().count()
# print(f"Number of unique fishing MMSI numbers: {unique_fishing_mmsi_count}")
#
# # MMSIs with only NULL ROT values
# mmsi_with_only_null_rot = df.groupBy("MMSI").agg(
#     count("*").alias("total"),
#     count("ROT").alias("non_null_rot")
# ).filter(col("non_null_rot") == 0)
#
# mmsi_null_rot_count = mmsi_with_only_null_rot.count()
# print(f"Number of fishing MMSI numbers with only NULL in ROT: {mmsi_null_rot_count}")
#
# # Stop Spark session
# spark.stop()
#
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, count, when, isnan
#
# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("Fishing MMSI with All NULL Columns") \
#     .getOrCreate()
#
# # Load CSV
#
# df = spark.read.option("header", True).option("inferSchema", True).csv(y)
#
# # Filter Ship Type == 'Fishing'
# fishing_df = df.filter(col("Ship Type") == "Fishing")
# fishing_df = fishing_df.drop("Type of mobile","Navigational status","IMO","Callsign","Name","Ship type","Cargo type","Type of position fixing device","Destination","ETA","Data source type","A","B","C","D")
#
# print(fishing_df.columns)
# # Get list of relevant data columns (excluding MMSI and Ship Type)
# data_columns = [c for c in fishing_df.columns if c not in ("MMSI", "Ship Type")]
#
# # Group by MMSI and count non-NULLs per column
# agg_exprs = [count(c).alias(f"non_null_{c}") for c in data_columns]
# grouped = fishing_df.groupBy("MMSI").agg(*agg_exprs)
#
# # Identify MMSIs where ANY column is completely NULL
# condition = None
# for c in data_columns:
#     null_check = col(f"non_null_{c}") == 0
#     condition = null_check if condition is None else condition | null_check
#
# mmsi_with_any_all_null_col = grouped.filter(condition)
#
# # Final counts
# unique_mmsi_count = fishing_df.select("MMSI").distinct().count()
# mmsi_null_col_count = mmsi_with_any_all_null_col.count()
#
# print(f"Number of unique fishing MMSI numbers: {unique_mmsi_count}")
# print(f"Number of fishing MMSIs with at least one column completely NULL: {mmsi_null_col_count}")
#
# # Identify columns that are completely NULL (non_null count == 0)
# null_flags = [(col(f"non_null_{c}") == 0).cast("int").alias(f"all_null_{c}") for c in data_columns]
# null_flagged_df = grouped.select("MMSI", *null_flags)
#
# # Filter to MMSIs where at least one column is completely NULL
# at_least_one_null = sum([col(f"all_null_{c}") for c in data_columns]) > 0
# mmsi_with_null_flags = null_flagged_df.filter(at_least_one_null)
#
# # Show the columns that are NULL for each MMSI
# mmsi_with_null_flags.show(truncate=False)
#
# # Calculate how many MMSIs have each column completely NULL
# null_counts = null_flagged_df.select([
#     spark_sum(col(f"all_null_{c}")).alias(f"total_all_null_{c}")
#     for c in data_columns
# ])
#
# # Show the result
# null_counts.show(truncate=False)
#
# # Step 1: Get MMSIs with at least one all-null column
# bad_mmsis = mmsi_with_null_flags.select("MMSI").distinct()
#
# # Step 2: Total number of rows in the fishing dataset
# total_rows = fishing_df.count()
#
# # Step 3: Join to find all rows belonging to bad MMSIs
# bad_rows_df = fishing_df.join(bad_mmsis, on="MMSI", how="inner")
#
# # Step 4: Count rows from bad MMSIs
# bad_rows_count = bad_rows_df.count()
#
# # Step 5: Compute and print proportion
# print(f"Total rows in fishing dataset: {total_rows}")
# print(f"Rows with MMSIs that have one or more all-null columns: {bad_rows_count}")
# print(f"Percentage: {100 * bad_rows_count / total_rows:.2f}%")



from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Unique Cargo Types") \
    .getOrCreate()

# Load CSV
df = spark.read.option("header", True).option("inferSchema", True).csv(x)

# Select distinct Cargo Type values (including null)
df.select("Cargo Type").distinct().show()