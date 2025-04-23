import pyspark
from pyspark.sql import SparkSession, DataFrame
import logging
import findspark
from pyspark.sql.functions import to_timestamp, col, count, when, isnan, lag, struct
import random
import sys
from tempo.tsdf import TSDF

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, NumericType, TimestampType

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
    """
    Function that splits the data PER unique vessel. 
    The split is stratified, ensuring that there are somewhat equal numbers of
    gear types in each split.

    """


    # Ensure that split-sizes sum to 1
    assert abs(train_size + test_size + val_size) == 1, "Splits must sum to 1!"


    # Extract unique (MMSI, Gear Type) pairs
    vessel_df = df.select("MMSI", "Gear Type").distinct()
    total_vessels = vessel_df.count() 

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
    print(f'Total number of vessels: {total_vessels}')
    print(f'Train vessels: {len(train_mmsis)}')
    print(f'Val vessels: {len(val_mmsis)}')
    print(f'Test vessels: {len(test_mmsis)}')

    # Print percentages
    print(f'\nSplit percentages:')
    print(f'Train: {len(train_mmsis)/total_vessels:.1%}')
    print(f'Val: {len(val_mmsis)/total_vessels:.1%}')
    print(f'Test: {len(test_mmsis)/total_vessels:.1%}')

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

    # Remove rows that don't have enough data for future prediction (those without target columns)
    df = df.filter(df[f"y_lat_{forecast_steps}"].isNotNull())

    # Drop intermediate variables
    df = df.drop('prev_timestamp','time_diff','new_trajectory','trajectory_id')

    return df



def drop_vessels_with_all_nulls(df, id_col, timestamp_col, debug_examples=False):
    """
    Drops vessels (groups) from the DataFrame where any column has only NULL values for any given day.
    Now includes debug output showing removed vessels and their NULL columns.

    Args:
        df: Input Spark DataFrame
        id_col: Column name used to identify each vessel (e.g., 'MMSI')
        timestamp_col: Column name with timestamp data (e.g., 'Timestamp')

    Returns:
        Spark DataFrame with vessels dropped for days where at least one column had all NULL values.
    """
    # Extract date from timestamp
    df_with_date = df.withColumn("date", F.to_date(F.col(timestamp_col)))

    # Get the list of columns excluding ID, timestamp, and date
    cols = [c for c in df.columns if c not in [id_col, timestamp_col, "date"]]

    # Count non-null values per vessel per day
    non_null_counts = [
        F.count(F.when(F.col(c).isNotNull(), 1)).alias(c) for c in cols
    ]
    
    # Get vessel-date combinations to keep
    df_with_counts = df_with_date.groupBy(id_col, "date").agg(*non_null_counts)
    drop_condition = " OR ".join([f"`{c}` = 0" for c in cols])
    
    # Identify bad pairs first for debugging
    bad_pairs = df_with_counts.filter(drop_condition).select(id_col, "date")
    
    if debug_examples == True:
        # DEBUG: Show sample of removed vessels and their NULL columns
        if bad_pairs.count() > 0:
            print(f"\nDEBUG: Found {bad_pairs.count()} vessel-day combinations to remove")
            sample_bad = bad_pairs.limit(10).collect()
            
            for i, row in enumerate(sample_bad):
                vessel_id = row[id_col]
                date = row["date"]
                print(f"\nDEBUG EXAMPLE {i+1}: Vessel {vessel_id} on {date}")
                
                # Show NULL counts for this vessel-day
                vessel_data = df_with_date.filter(
                    (F.col(id_col) == vessel_id) & 
                    (F.col("date") == date)
                )
                
                null_counts = vessel_data.agg(
                    *[F.count(F.when(F.isnull(c), 1)).alias(c) for c in cols]
                ).collect()[0]
                
                total_rows = vessel_data.count()
                print(f"Total rows for this vessel-day: {total_rows}")
                print("NULL counts per column:")
                for col in cols:
                    if null_counts[col] == total_rows:
                        print(f"  {col}: ALL NULL ({null_counts[col]}/{total_rows})")
                    elif null_counts[col] > 0:
                        print(f"  {col}: {null_counts[col]}/{total_rows} NULL")
    
    # Now filter to get valid pairs
    valid_vessels_per_day = df_with_counts.filter(f"NOT ({drop_condition})").select(id_col, "date")

    # Log statistics before dropping
    total_vessel_days = df_with_date.select(id_col, "date").distinct().count()
    dropped_vessel_days = total_vessel_days - valid_vessels_per_day.distinct().count()
    print(f"\nDropping {dropped_vessel_days} vessel-day combinations where at least one column had all NULLs")

    # Join to retain only valid vessel-day data
    df_cleaned = df_with_date.join(
        F.broadcast(valid_vessels_per_day),
        (df_with_date[id_col] == valid_vessels_per_day[id_col]) & 
        (df_with_date["date"] == valid_vessels_per_day["date"]),
        "left_semi"
    ).drop("date")

    # Log final statistics
    original_rows = df.count()
    cleaned_rows = df_cleaned.count()
    print(f"Dropped {original_rows - cleaned_rows} rows ({((original_rows - cleaned_rows)/original_rows*100):.2f}% of total)")

    # Sort the dataframe by MMSI and timestamp
    df_cleaned = df_cleaned.orderBy(id_col, timestamp_col)

    return df_cleaned

def print_missing_value_count(df):
    # Check for missing values
    if 'Timestamp' in df.columns:
        df = df.drop('Timestamp')

    print("Missing values count:")
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
    ).show()

def fill_linear_interpolation(df,id_cols,order_col, value_cols, partitions=200):
    """ 
    Apply linear interpolation to dataframe to fill gaps. 

    This function has been copied from a Stackoverflow post.
    SOURCE: https://stackoverflow.com/questions/53077639/pyspark-interpolation-of-missing-values-in-pyspark-dataframe-observed

    Args:
        df: spark dataframe
        id_cols: string or list of column names to partition by the window function 
        order_col: column to use to order by the window function
        value_col: column to be filled

    Returns: 
        spark dataframe updated with interpolated values
    """
    new_df = df

    new_df = new_df.repartition(partitions, *id_cols)

    # Make sure id_cols is a list
    if not isinstance(id_cols, list):
        id_cols = [id_cols]

    for value_col in value_cols:
        w = Window.partitionBy(id_cols).orderBy(order_col)
        new_df = new_df.withColumn('rn',F.row_number().over(w))
        new_df = new_df.withColumn('rn_not_null',F.when(F.col(value_col).isNotNull(),F.col('rn')))

        # create relative references to the start value (last value not missing)
        w_start = Window.partitionBy(id_cols).orderBy(order_col).rowsBetween(Window.unboundedPreceding,-1)
        new_df = new_df.withColumn('start_val',F.last(value_col,True).over(w_start))
        new_df = new_df.withColumn('start_rn',F.last('rn_not_null',True).over(w_start))

        # create relative references to the end value (first value not missing)
        w_end = Window.partitionBy(id_cols).orderBy(order_col).rowsBetween(0,Window.unboundedFollowing)
        new_df = new_df.withColumn('end_val',F.first(value_col,True).over(w_end))
        new_df = new_df.withColumn('end_rn',F.first('rn_not_null',True).over(w_end))

        # create references to gap length and current gap position  
        new_df = new_df.withColumn('diff_rn',F.col('end_rn')-F.col('start_rn'))
        new_df = new_df.withColumn('curr_rn',F.col('diff_rn')-(F.col('end_rn')-F.col('rn')))

        # calculate linear interpolation value
        lin_interp_func = (F.col('start_val')+(F.col('end_val')-F.col('start_val'))/F.col('diff_rn')*F.col('curr_rn'))
        new_df = new_df.withColumn(value_col,F.when(F.col(value_col).isNull(),lin_interp_func).otherwise(F.col(value_col)))

        # Forward fill remaining NULLs (e.g., at start of each partition)
        w_ff = Window.partitionBy(id_cols).orderBy(order_col).\
            rowsBetween(Window.unboundedPreceding, 0)
        new_df = new_df.withColumn(value_col, F.last(value_col, 
                                    ignorenulls=True).over(w_ff))

        # Backward fill remaining NULLs (e.g., at the end of each partition)
        w_bf = Window.partitionBy(id_cols).orderBy(order_col).\
            rowsBetween(0, Window.unboundedFollowing)
        new_df = new_df.withColumn(value_col, F.first(value_col, 
                                    ignorenulls=True).over(w_bf))

        # Drop intermediate features
        new_df = new_df.drop('rn', 'rn_not_null', 'start_val', 'end_val', 
                            'start_rn', 'end_rn', 'diff_rn', 'curr_rn')

    return new_df



def add_lagged_features(df, id_col, timestamp_col, lat_col, lon_col):
    """
    Adds lagged position features and drops rows with NULLs introduced by lagging
    
    Args:
        df: Input DataFrame (should already have NULL position rows removed)
        id_col: Vessel identifier column (e.g., 'MMSI')
        timestamp_col: Timestamp column
        lat_col: Latitude column name
        lon_col: Longitude column name
        
    Returns:
        DataFrame with lagged features and no NULLs in lagged columns
    """

    windowSpec = Window.partitionBy(id_col).orderBy(timestamp_col)
    
    # Add lagged features (corrected syntax)
    df_with_lags = (df
        .withColumn(f'{lat_col}_lag1', F.lag(lat_col, 1).over(windowSpec))
        .withColumn(f'{lon_col}_lag1', F.lag(lon_col, 1).over(windowSpec))
        .withColumn(f'{lat_col}_lag5', F.lag(lat_col, 5).over(windowSpec))
        .withColumn(f'{lon_col}_lag5', F.lag(lon_col, 5).over(windowSpec))
    )

    # Drop initial rows with NULLs
    condition = (
        F.col(f'{lat_col}_lag1').isNull() |
        F.col(f'{lon_col}_lag1').isNull() |
        F.col(f'{lat_col}_lag5').isNull() |
        F.col(f'{lon_col}_lag5').isNull()
    )

    final_df = df_with_lags.filter(~condition)
    
    return final_df

def resampling(df, id_col, timestamp_col, method='resampling', sampling_interval='min'):
    """
    Resample or downsample the DataFrame.
    Note, requires pip install dbl-tempo

    Args:
        df: DataFrame
        id_col: Vessel identifier column (e.g., 'MMSI')
        timestamp_col: Timestamp column
        method: 'resampling' or 'downsampling': Specifies which method to use
    """
    # Make sure id_cols is a list
    if not isinstance(id_col, list):
        id_col = [id_col]

    df = df.orderBy(id_col + [timestamp_col])

    tsdf = TSDF(df, ts_col=timestamp_col, partition_cols=id_col)

    if method == 'resampling':
        func = 'mean'
    elif method == 'downsampling':
        func = 'floor'

    resampled_tsdf = tsdf.resample(freq=sampling_interval, func=func)
    
    df = resampled_tsdf.df
    df = df.orderBy(id_col + [timestamp_col])

    print("Resampling complete")
    print(f"DataFrame size after resampling: {df.count()} rows x {len(df.columns)} columns")

    return df

def preprocess_pipeline(df, forecasting=True):
    # Mandatory preprocessing
    df = drop_class_B(df)

    features_to_drop = ('Type of mobile','Navigational status','IMO','Callsign',
                        'Name','Cargo type','Width','Length',
                        'Type of position fixing device','Destination',
                        'ETA','Data source type','A','B','C','D','Ship type')
    df = df.drop(*features_to_drop)

    df = drop_duplicates(df)
    df.cache()
    df.count()

    df = drop_unknown_label(df)
    df.cache()
    df.count()

    df = drop_vessels_with_all_nulls(
        df=df,
        id_col='MMSI',
        timestamp_col='Timestamp'
    )
    df.cache()
    df.count()

    # Split the data
    train_df, val_df, test_df = split_data(df, train_size=0.7, test_size=0.15, 
                                           val_size=0.15, random_state=42)

    # Drop gear type feature
    train_df = train_df.drop('Gear Type')
    val_df = val_df.drop('Gear Type')
    test_df = test_df.drop('Gear Type')

    train_df.cache()
    train_df.count()
    val_df.cache()
    val_df.count()
    test_df.cache()
    test_df.count()


    # Apply resampling/downsampling
    train_df = resampling(train_df, 'MMSI', 'Timestamp',method='resampling') # Or downsampling
    val_df = resampling(val_df, 'MMSI', 'Timestamp',method='resampling')
    test_df = resampling(test_df, 'MMSI', 'Timestamp',method='resampling')
    train_df.cache()
    train_df.count()
    val_df.cache()
    val_df.count()
    test_df.cache()
    test_df.count()
    print("Resampling complete")

    # Impute missing values with linear interpolation
    value_cols = ['ROT','SOG','COG','Heading']
    train_df = fill_linear_interpolation(train_df, ['MMSI'],'Timestamp', value_cols=value_cols)
    val_df = fill_linear_interpolation(val_df, ['MMSI'],'Timestamp', value_cols=value_cols)
    test_df = fill_linear_interpolation(test_df, ['MMSI'],'Timestamp', value_cols=value_cols)

    # Make trawling label to int
    train_df = train_df.withColumn('trawling', col('trawling').cast('int'))
    val_df = val_df.withColumn('trawling', col('trawling').cast('int'))
    test_df = test_df.withColumn('trawling', col('trawling').cast('int'))
    
    print("Missing value imputation complete!")

    if forecasting:
        train_df = add_lagged_features(
            df=train_df,
            id_col='MMSI',
            timestamp_col='Timestamp',
            lat_col='Latitude',
            lon_col='Longitude'
        )
        train_df = define_forecasting_target(
            df=train_df,
            forecast_steps=20
        )

        val_df = add_lagged_features(
            df=val_df,
            id_col='MMSI',
            timestamp_col='Timestamp',
            lat_col='Latitude',
            lon_col='Longitude'
        )
        val_df = define_forecasting_target(
            df=val_df,
            forecast_steps=20
        )

        test_df = add_lagged_features(
            df=test_df,
            id_col='MMSI',
            timestamp_col='Timestamp',
            lat_col='Latitude',
            lon_col='Longitude'
        )
        test_df = define_forecasting_target(
            df=test_df,
            forecast_steps=20
        )
    
    train_df = train_df.orderBy("MMSI", "Timestamp")
    val_df = val_df.orderBy('MMSI','Timestamp')
    test_df = test_df.orderBy('MMSI','Timestamp')

    return train_df, val_df, test_df

def define_input_output(df, forecasting=True):
    # Drop non-numerical features
    features_to_drop = ['MMSI','Timestamp']
    df = df.drop(*features_to_drop)

    if forecasting:
        df = df.drop('trawling')
        y_combined = []

        for i in range(1, 21):
            lat_col = 'y_lat_' + str(i)
            lon_col = 'y_lon_' + str(i)

            # Pair latitude and longitude columns for each timestep (1 to 20)
            y_combined.append(struct(col(lat_col), col(lon_col)).alias(f'y_pair_{i}'))

        # Create a new DataFrame with these combined pairs
        y_df = df.select(*y_combined)
        # Define target columns
        X_df = df.drop(*y_combined)
        y_df.show(5)
    else:
        X_df = df.drop('trawling')
        y_df = df.select('trawling')  
    return X_df, y_df

def main():
    # Load data
    data_path = "data/aisdk-2025-01-01_fishing_labeled.csv"
    df = load_data(data_path)

    train_df, val_df, test_df = preprocess_pipeline(df, forecasting=True)
    X_train, y_train = define_input_output(train_df, forecasting=True)

    X_train.show(5)
    y_train.show(5)

    # # Drop class B vessels
    # df = drop_class_B(df)

    # # Drop static features
    # features_to_drop = ('Type of mobile','Navigational status','IMO','Callsign',
    #                     'Name','Cargo type','Width','Length',
    #                     'Type of position fixing device','Destination',
    #                     'ETA','Data source type','A','B','C','D','Ship type')

    # df = df.drop(*features_to_drop)

    # # Drop duplicates
    # df = drop_duplicates(df)

    # # Drop unknown labels
    # df = drop_unknown_label(df)

    # df = drop_vessels_with_all_nulls(
    #     df=df,
    #     id_col='MMSI',
    #     timestamp_col='Timestamp'
    # )


    # # Split the data
    # train_df, val_df, test_df = split_data(df, train_size=0.7, test_size=0.15, 
    #                                        val_size=0.15, random_state=42)
    
    # train_df = train_df.drop('Gear Type')
    # train_df.cache()
    # train_df.count()

    # print(f"DataFrame size: {train_df.count()} rows x {len(train_df.columns)} columns")

    # print("")

    # print("BEFORE RESAMPLING:")
    # print_missing_value_count(train_df)

    # # train_df.show(20)
    # train_df = resampling(
    #     df=train_df,
    #     id_col='MMSI',
    #     timestamp_col='Timestamp'
    # )
    # train_df.cache()
    # train_df.count()
    # train_df.show(5)
    
    # print("AFTER RESAMPLING")
    # print_missing_value_count(train_df)

    # train_df.show(10, truncate=False)

    # IMPUTE MISSING VALUES
    # train_df = fill_linear_interpolation(
    #     df=train_df,
    #     id_cols=['MMSI'],
    #     order_col='Timestamp',
    #     value_cols=['ROT','SOG','COG','Heading']
    # )
    # train_df.cache()
    # train_df.count()
    # print_missing_value_count(train_df)


    # Add lags
    # train_df = add_lagged_features(
    #     df=train_df,
    #     id_col='MMSI',
    #     timestamp_col='Timestamp',
    #     lat_col='Latitude',
    #     lon_col='Longitude'
    # )


    """
    Debugging:

    # Make sure vessel is removed after calling drop_vessels_with_all_nulls:

    bad_vessel_id = 219004242  # Example vessel that should have been removed
    bad_date = "2023-01-01"    # Example date when it was bad

    # Check if this vessel-date combination exists in cleaned data
    exists_in_cleaned = train_df.filter(
        (F.col('MMSI') == bad_vessel_id) & 
        (F.to_date(F.col('Timestamp')) == bad_date)
    ).count() > 0

    print(f"Vessel {bad_vessel_id} on {bad_date} {'STILL EXISTS' if exists_in_cleaned else 'was properly removed'}")

    """


if __name__ == '__main__':
    main()