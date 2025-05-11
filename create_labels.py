import os
import shutil
import glob
import json
import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from functools import reduce
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql.functions import col, unix_timestamp, to_timestamp, explode, sequence, min, max


# Argument parsing
parser = argparse.ArgumentParser(description="Process AIS data with resampling, interpolation, and lag generation.")
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--version', type=str, required=True, help="Version name, e.g. v1, v2")

args = parser.parse_args()

base_input = f"/ceph/project/gatehousep4/data/{args.mode}/{args.version}"
base_output = f"/ceph/project/gatehousep4/data/{args.mode}_labeled/{args.version}"

os.makedirs(base_output, exist_ok=True)

input_folder = base_input
output_folder = base_output

# Initialize Spark
spark = SparkSession.builder \
    .appName(f"AIS Labeling ({args.mode})") \
    .config("spark.sql.shuffle.partitions", "128") \
    .config("spark.local.dir", "/ceph/project/gatehousep4/data/petastorm_cache") \
    .config("spark.driver.memory", "200") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

schema = StructType([
    StructField("MMSI", IntegerType(), True),
    StructField("# Timestamp", StringType(), True),
    StructField("Type of mobile", StringType(), True),
    StructField("Latitude", DoubleType(), True),
    StructField("Longitude", DoubleType(), True),
    StructField("Navigational status", StringType(), True),
    StructField("ROT", DoubleType(), True),
    StructField("SOG", DoubleType(), True),
    StructField("COG", DoubleType(), True),
    StructField("Heading", IntegerType(), True),
    StructField("IMO", StringType(), True),
    StructField("Callsign", StringType(), True),
    StructField("Name", StringType(), True),
    StructField("Ship type", StringType(), True),
    StructField("Cargo type", StringType(), True),
    StructField("Width", IntegerType(), True),
    StructField("Length", IntegerType(), True),
    StructField("Type of position fixing device", StringType(), True),
    StructField("Draught", DoubleType(), True),
    StructField("Destination", StringType(), True),
    StructField("ETA", StringType(), True),
    StructField("Data source type", StringType(), True),
    StructField("A", IntegerType(), True),
    StructField("B", IntegerType(), True),
    StructField("C", IntegerType(), True),
    StructField("D", IntegerType(), True),
    StructField("Gear Type", StringType(), True),
    StructField("trawling", IntegerType(), True),
])

# Feature lists
continuous_cols = ["Latitude", "Longitude", "ROT", "SOG", "COG", "Heading", "Draught"]
static_cols = ["Width", "Length", "trawling"]

outlier_iqr_cols = ["Latitude", "Longitude"]
static_outlier_cols = ["Width", "Length"]

skewed_positive = ["SOG", "Draught"]
skewed_signed = ["ROT"]

def filter_relevant_columns(df):
    return df.select(["MMSI", "ts"] + static_cols + continuous_cols)

def drop_unknown_label(df):
    return df.filter(~df['Gear Type'].isin(["UNKNOWN", "INCONCLUSIVE"]))

def drop_class_b(df):
    return df.filter(~df['Type of mobile'].isin(["B"]))

def save_normalization_stats(df, output_path):
    stats = df.select([
        F.mean(c).alias(f"{c}_mean") for c in continuous_cols
    ] + [
        F.stddev(c).alias(f"{c}_std") for c in continuous_cols
    ]).collect()[0]

    stats_dict = {
        c: {
            "mean": stats[f"{c}_mean"],
            "std": stats[f"{c}_std"]
        }
        for c in continuous_cols
    }

    os.makedirs(output_path, exist_ok=True)
    stats_file = os.path.join(output_path, f"{args.mode}_normalization_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats_dict, f, indent=4)

    print(f"\nNormalization stats written to: {stats_file}")

def filter_low_quality_mmsis(df):
    """Remove MMSIs where any continuous column is completely null."""
    null_agg = df.groupBy("MMSI").agg(*[
        F.sum(F.col(c).isNull().cast("int")).alias(f"{c}_nulls") for c in continuous_cols
    ])
    row_counts = df.groupBy("MMSI").agg(F.count("*").alias("row_count"))
    joined = null_agg.join(row_counts, "MMSI")

    for c in continuous_cols:
        joined = joined.withColumn(f"{c}_all_null", F.col(f"{c}_nulls") == F.col("row_count"))

    conditions = [~joined[f"{c}_all_null"] for c in continuous_cols]
    valid_mmsis = joined.where(reduce(lambda x, y: x & y, conditions)).select("MMSI")
    return df.join(valid_mmsis, on="MMSI", how="inner")


def filter_outliers(df):
    """Apply IQR filtering for lat/lon and quantile clipping for static dimensions."""
    df = df.withColumn("month", F.date_format("ts", "yyyy-MM"))
    quant_exprs = [
        F.expr(f"percentile_approx({c}, array(0.25, 0.75), 10000)").alias(f"{c}_quants") for c in outlier_iqr_cols
    ]
    month_quants = df.groupBy("month").agg(*quant_exprs)
    for c in outlier_iqr_cols:
        month_quants = month_quants.withColumn(f"{c}_q1", F.col(f"{c}_quants")[0]) \
                                   .withColumn(f"{c}_q3", F.col(f"{c}_quants")[1]) \
                                   .drop(f"{c}_quants")
    df = df.join(month_quants, "month", "left")
    for c in outlier_iqr_cols:
        iqr = F.col(f"{c}_q3") - F.col(f"{c}_q1")
        df = df.filter(F.col(c).between(F.col(f"{c}_q1") - 1.5 * iqr, F.col(f"{c}_q3") + 1.5 * iqr))
    df = df.drop(*[f"{c}_q1" for c in outlier_iqr_cols], *[f"{c}_q3" for c in outlier_iqr_cols], "month")

    for c in static_outlier_cols:
        q = df.stat.approxQuantile(c, [0.01, 0.99], 0.001)
        df = df.filter(F.col(c).between(q[0], q[1]))

    return df

def forward_fill_features(df):
    """Forward-fill static and dynamic features within each MMSI."""
    w = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(Window.unboundedPreceding, 0)
    for col in static_cols + ["ROT", "Heading", "Draught"]:
        df = df.withColumn(col, F.last(col, ignorenulls=True).over(w))
    return df

def clamp_features(df):
    """Clamp ROT and SOG to realistic limits."""
    df = df.withColumn("SOG", F.when(F.col("SOG") < 0, 0).when(F.col("SOG") > 40, 40).otherwise(F.col("SOG")))
    df = df.withColumn("ROT", F.when(F.col("ROT") < -90, -90).when(F.col("ROT") > 90, 90).otherwise(F.col("ROT")))
    return df


def reduce_skewness(df):
    """Automatically log-transform features if skewness is high."""
    for col in skewed_positive:
        skew_val = df.select(F.skewness(col).alias("skew")).collect()[0]["skew"]
        if abs(skew_val) > 1:
            print(f"Applying log1p() to positively skewed column: {col} (skew={skew_val:.2f})")
            df = df.withColumn(col, F.log1p(F.col(col)))
    for col in skewed_signed:
        skew_val = df.select(F.skewness(col).alias("skew")).collect()[0]["skew"]
        if abs(skew_val) > 1:
            print(f"Applying symmetric log1p() to signed skewed column: {col} (skew={skew_val:.2f})")
            df = df.withColumn(col, F.when(F.col(col) >= 0, F.log1p(F.col(col)))
                                     .otherwise(-F.log1p(-F.col(col))))

def round_timestamps(df):
    df = df.withColumn("ts", F.from_unixtime((F.unix_timestamp("ts") / 10).cast("int") * 10))
    return df

def interpolate_continuous_features(df):
    """Perform linear interpolation on continuous features within a 5-minute window"""
    for col in continuous_cols:

        w_prev = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(Window.unboundedPreceding, 0)
        w_next = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(0, Window.unboundedFollowing)

        df = df.withColumn(f"{col}_prev", F.last(col, ignorenulls=True).over(w_prev))
        df = df.withColumn(f"{col}_next", F.first(F.when(F.col(col).isNotNull(), F.col(col)), ignorenulls=True).over(w_next))
        df = df.withColumn(f"ts_{col}_prev", F.last(F.when(F.col(col).isNotNull(), F.col("ts")), ignorenulls=True).over(w_prev))
        df = df.withColumn(f"ts_{col}_next", F.first(F.when(F.col(col).isNotNull(), F.col("ts")), ignorenulls=True).over(w_next))

        df = df.withColumn(f"frac_{col}", F.when(
            (F.unix_timestamp(f"ts_{col}_next") != F.unix_timestamp(f"ts_{col}_prev")) &
            F.col(f"ts_{col}_prev").isNotNull() &
            F.col(f"ts_{col}_next").isNotNull(),
            (F.unix_timestamp("ts") - F.unix_timestamp(f"ts_{col}_prev")) /
            (F.unix_timestamp(f"ts_{col}_next") - F.unix_timestamp(f"ts_{col}_prev"))
        ))

        df = df.withColumn(f"time_diff_{col}", F.unix_timestamp(f"ts_{col}_next") - F.unix_timestamp(f"ts_{col}_prev"))

        df = df.withColumn(col, F.when(
            F.col(col).isNull() &
            F.col(f"{col}_prev").isNotNull() &
            F.col(f"{col}_next").isNotNull() &
            F.col(f"frac_{col}").isNotNull() &
            (F.col(f"time_diff_{col}") <= 300),  # 5 minutes
            F.col(f"{col}_prev") + (F.col(f"{col}_next") - F.col(f"{col}_prev")) * F.col(f"frac_{col}")
        ).otherwise(F.col(col)))

        helper_cols = [f"{col}_prev", f"{col}_next", f"ts_{col}_prev", f"ts_{col}_next", f"frac_{col}", f"time_diff_{col}"]
        df = df.drop(*helper_cols)

    return df


def get_spark_session():
    return spark

def get_paths():
    return input_folder, output_folder

def get_columns():
    return continuous_cols, static_cols

def load_csv_file(path):
    """Load and clean a single CSV file."""
    df = spark.read.option("header", True).schema(schema).csv(path)
    df = df.withColumnRenamed("# Timestamp", "ts_raw")
    df = df.withColumn("ts", F.to_timestamp("ts_raw", "dd/MM/yyyy HH:mm:ss")).drop("ts_raw")
    df = df.dropDuplicates(["MMSI", "ts"])
    return df

def write_single_output(df, base, output_folder):
    out_path = os.path.join(output_folder, base.replace(".csv", ""))
    df.write.mode("overwrite").parquet(out_path)

# def write_single_output(df, base, output_folder):
#     """Write the DataFrame to a single CSV file."""
#     temp_dir = os.path.join(output_folder, base + "_tmp")
#     df.coalesce(1).write.option("header", True).mode("overwrite").csv(temp_dir)
#     part_files = glob.glob(os.path.join(temp_dir, "part-*.csv"))
#     if not part_files:
#         raise FileNotFoundError(f"No CSV part file found in {temp_dir}")
#     tmp_csv = part_files[0]
#     dest = os.path.join(output_folder, base)
#     shutil.move(tmp_csv, dest)
#     shutil.rmtree(temp_dir)
#     return dest

def drop_mmsis_with_nulls(df):
    """
    Drops entire MMSIs from the DataFrame if any of the specified columns contain nulls.
    Intended to be used after interpolation/resampling.
    """
    # Count nulls per MMSI
    null_counts = df.groupBy("MMSI").agg(*[
        F.sum(F.col(c).isNull().cast("int")).alias(f"{c}_nulls") for c in continuous_cols
    ])

    # Filter MMSIs where all specified columns have zero nulls
    conditions = [F.col(f"{c}_nulls") == 0 for c in continuous_cols]
    good_mmsis = null_counts.where(reduce(lambda x, y: x & y, conditions)).select("MMSI")

    # Join back to original DF
    df_clean = df.join(good_mmsis, on="MMSI", how="inner")
    return df_clean


def resample_to_fixed_interval(df):
    """Resample data to fixed 10-second intervals using UNIX epoch seconds."""

    # First align ts to the 10s grid (crucial!)
    df = df.withColumn("timestamp_epoch", (F.unix_timestamp("ts") / 10).cast("int") * 10)
    df = df.withColumn("ts", F.to_timestamp("timestamp_epoch"))

    # Calculate bounds for each MMSI
    bounds = df.groupBy("MMSI").agg(
        F.min("timestamp_epoch").alias("min_epoch"),
        F.max("timestamp_epoch").alias("max_epoch")
    )

    # Create 10s-aligned grid
    grid = bounds.select(
        "MMSI",
        explode(sequence(col("min_epoch"), col("max_epoch"), F.lit(10))).alias("timestamp_epoch")
    ).withColumn("ts", to_timestamp(col("timestamp_epoch")))

    # Join aligned data to grid
    return grid.join(df, on=["MMSI", "ts"], how="left").orderBy("MMSI", "ts")

def drop_rows_with_null_future_coords(df, horizon=120):
    """Drop rows where any future latitude or longitude column is null."""
    conditions = [F.col(f"future_lat_{i}").isNotNull() & F.col(f"future_lon_{i}").isNotNull() for i in range(1, horizon + 1)]
    combined_condition = reduce(lambda x, y: x & y, conditions)
    return df.filter(combined_condition)

def add_future_lags(df, horizon=120):
    """Add future latitude and longitude columns for prediction targets."""

    # Drop existing timestamp_epoch if it exists to avoid ambiguity
    if "timestamp_epoch" in df.columns:
        df = df.drop("timestamp_epoch")

    # Recreate timestamp_epoch for ordering
    df = df.withColumn("timestamp_epoch", F.unix_timestamp("ts").cast("long"))

    # Define window for future values
    w = Window.partitionBy("MMSI").orderBy("timestamp_epoch")

    for i in range(1, horizon + 1):
        df = df.withColumn(f"future_lat_{i}", F.lead("Latitude", i).over(w))
        df = df.withColumn(f"future_lon_{i}", F.lead("Longitude", i).over(w))

    return df

def preprocess_all_files():
    input_folder, output_folder = get_paths()
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return []

    all_data = []

    for path in csv_files:

        # Load data
        df = load_csv_file(path)

        # Data processing

        df = drop_class_b(df)
        df = drop_unknown_label(df)
        df = filter_relevant_columns(df)
        df = filter_low_quality_mmsis(df)
        df = filter_outliers(df)

        df = df.repartition("MMSI")

        df = resample_to_fixed_interval(df)
        df = interpolate_continuous_features(df)
        df = forward_fill_features(df)
        df = drop_mmsis_with_nulls(df)
        df = clamp_features(df)
        df = add_future_lags(df)
        df = drop_rows_with_null_future_coords(df)

        all_data.append(df)

        # Saving processed data
        base = os.path.basename(path).replace("_fishing_labeled.csv", "_prod_ready.csv")
        output_file = write_single_output(df, base, output_folder)

    return None


if __name__ == '__main__':
    preprocess_all_files()
    spark.stop()
