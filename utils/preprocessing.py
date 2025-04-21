import pyspark
from pyspark.sql import SparkSession, DataFrame
import logging
import findspark
from pyspark.sql.functions import to_timestamp, col
import random

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

logging.getLogger("py4j").setLevel(logging.ERROR)
findspark.init()
spark = SparkSession.builder \
    .appName("P4_project") \
    .config("spark.sql.shuffle.partitions","200") \
    .getOrCreate()

def load_data(data_path):
    # Load the AIS data from multiple CSV files in the "data" folder
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Rename the column '# Timestamp' to 'Timestamp'
    df = df.withColumnRenamed("# Timestamp", "Timestamp")

    # Convert 'Timestamp' column to a timestamp type using the correct format
    df = df.withColumn("Timestamp", to_timestamp("Timestamp", "dd/MM/yyyy HH:mm:ss"))

    return df

def drop_duplicates(df):
    df = df.dropDuplicates()
    print("Duplicates have been dropped!")
    return df

def drop_unknown_label(df):
    df = df.filter(df['Gear Type'] != "UNKNOWN")
    df = df.filter(df['Gear Type'] != "INCONCLUSIVE")
    print("Dropped unknown gear types!")
    return df

def drop_class_B(df):
    df = df.filter(df['Type of mobile'] != 'Class B')
    print("Removed Class B vessels!")
    return df

def split_data(df, train_size=0.7, test_size=0.15, val_size=0.15, random_state=42):
    # Ensure that split-sizes sum to 1
    assert abs(train_size + test_size + val_size) == 1, "Splits must sum to 1!"


    # Extract unique (MMSI, Gear Type) pairs
    vessel_df = df.select("MMSI", "Gear Type").distinct()

    # Collect to driver for stratified sampling
    vessel_list = vessel_df.collect()

    # Group MMSIs by gear type
    gear_type_map = {}
    for row in vessel_list:
        gear_type = row["Gear Type"]
        mmsi = row["MMSI"]
        gear_type_map.setdefault(gear_type, []).append(mmsi)

    # Make stratified split
    train_mmsis, val_mmsis, test_mmsis = set(), set(), set()
    random.seed(random_state)

    for gear_type, mmsis in gear_type_map.items():
        random.shuffle(mmsis) # Shuffle based on random state
        n = len(mmsis)
        n_train = int(train_size * n)
        n_val = int(val_size * n)
        train_mmsis.update(mmsis[:n_train])
        val_mmsis.update(mmsis[n_train:n_train + n_val])
        test_mmsis.update(mmsis[n_train + n_val:])


    # Filter the full AIS dataset
    train_df = df.filter(col("MMSI").isin(train_mmsis))
    val_df   = df.filter(col("MMSI").isin(val_mmsis))
    test_df  = df.filter(col("MMSI").isin(test_mmsis))

    # Order by MMSI and time
    train_df = train_df.orderBy('MMSI', 'Timestamp')
    val_df = val_df.orderBy('MMSI', 'Timestamp')
    test_df = test_df.orderBy('MMSI', 'Timestamp')

    print("Splitted the data!")

    return train_df, val_df, test_df

def define_forecasting_target(df, max_time_gap=3600, forecast_steps=20):
    """
    Function to create target columns based on the future positions of vessels,
    considering a new trajectory if the time difference is more than the given time_gap (in seconds).
    
    Args:
    df: PySpark DataFrame with columns ['MMSI', 'Timestamp', 'Latitude', 'Longitude'].
    max_time_gap: Time difference (in seconds) to consider a new trajectory.
    forecast_steps: Number of steps ahead to forecast (e.g., 20).
    
    Returns:
    df: DataFrame with target columns for the forecaster.
    """
        
    # Sort the DataFrame by MMSI and Timestamp
    df = df.orderBy("MMSI", "Timestamp")

    # Compute the time difference between consecutive timestamps for each MMSI
    window_spec = Window.partitionBy("MMSI").orderBy("Timestamp")
    df = df.withColumn("prev_timestamp", F.lag("Timestamp").over(window_spec))
    df = df.withColumn("time_diff", 
                        (F.unix_timestamp("Timestamp") - F.unix_timestamp("prev_timestamp")).cast(IntegerType()))

    # Mark the start of a new trajectory when the time difference is more than max_time_gap (1 hour = 3600 seconds)
    df = df.withColumn("new_trajectory", F.when(df["time_diff"] > max_time_gap, 1).otherwise(0))

    # Generate the cumulative trajectory ID for each MMSI
    df = df.withColumn("trajectory_id", 
                        F.sum("new_trajectory").over(Window.partitionBy("MMSI").orderBy("Timestamp").rowsBetween(Window.unboundedPreceding, 0)))

    # Define the target columns - future positions (up to forecast_steps)
    for i in range(1, forecast_steps + 1):
        df = df.withColumn(f"y_lat_{i}", F.lead("Latitude", i).over(window_spec))
        df = df.withColumn(f"y_lon_{i}", F.lead("Longitude", i).over(window_spec))

    # Step 6: Remove rows that don't have enough data for future prediction (those without target columns)
    df = df.filter(df[f"y_lat_{forecast_steps}"].isNotNull())

    return df

def main():
    # Load data
    data_path = "data/aisdk-2025-01-04_fishing_labeled.csv"
    df = load_data(data_path)

    # Drop class B vessels
    df = drop_class_B(df)

    # Drop static features
    features_to_drop = ('Type of mobile','Navigational status','IMO','Callsign',
                        'Name','Cargo type','Width','Length',
                        'Type of position fixing device','Destination',
                        'ETA','Data source type','A','B','C','D')

    df = df.drop(*features_to_drop)

    # Drop duplicates
    df = drop_duplicates(df)

    # Drop unknown labels
    df = drop_unknown_label(df)

    # Split the data
    train_df, val_df, test_df = split_data(df, train_size=0.7, test_size=0.15, 
                                           val_size=0.15, random_state=42)
    train_df.show(5)

    train_df = define_forecasting_target(train_df, forecast_steps=1)
    train_df.show(5, truncate=False)


if __name__ == '__main__':
    main()