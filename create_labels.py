import os
import shutil
import glob
import json
import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from functools import reduce

# Argument parsing
parser = argparse.ArgumentParser(
    description="Process AIS data with resampling, interpolation, and lag generation."
)
parser.add_argument('--mode', choices=['train', 'test'], required=True)
args = parser.parse_args()

# Set folders
if args.mode == 'train':
    input_folder = "/ceph/project/gatehousep4/data/train"
    output_folder = "/ceph/project/gatehousep4/data/train_labeled"
elif args.mode == 'test':
    input_folder = "/ceph/project/gatehousep4/data/random"
    output_folder = "/ceph/project/gatehousep4/data/random_labeled"
else:
    raise ValueError("Invalid mode")

os.makedirs(output_folder, exist_ok=True)

# Initialize Spark
spark = SparkSession.builder \
    .appName(f"AIS Labeling ({args.mode})") \
    .config("spark.sql.shuffle.partitions", "128") \
    .config("spark.local.dir", "/ceph/project/gatehousep4/data/petastorm_cache") \
    .config("spark.driver.memory", "100g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

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

def interpolate_continuous_features(df):
    """Perform linear interpolation on continuous features within a 5-minute window."""
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
            (F.col(f"time_diff_{col}") <= 300),
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
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    df = df.withColumnRenamed("# Timestamp", "ts_raw")
    df = df.withColumn("ts", F.to_timestamp("ts_raw", "dd/MM/yyyy HH:mm:ss")).drop("ts_raw")
    df = df.dropDuplicates(["MMSI", "ts"])
    return df

def write_single_output(df, base, output_folder):
    """Write the DataFrame to a single CSV file."""
    temp_dir = os.path.join(output_folder, base + "_tmp")
    df.coalesce(1).write.option("header", True).mode("overwrite").csv(temp_dir)
    part_files = glob.glob(os.path.join(temp_dir, "part-*.csv"))
    if not part_files:
        raise FileNotFoundError(f"No CSV part file found in {temp_dir}")
    tmp_csv = part_files[0]
    dest = os.path.join(output_folder, base)
    shutil.move(tmp_csv, dest)
    shutil.rmtree(temp_dir)
    return dest

def resample_to_fixed_interval(df):
    """Resample data to fixed 10-second intervals per MMSI."""
    bounds = df.groupBy("MMSI").agg(F.min("ts").alias("min_ts"), F.max("ts").alias("max_ts"))
    grid = bounds.select("MMSI", F.explode(F.sequence("min_ts", "max_ts", F.expr("INTERVAL 10 seconds"))).alias("ts"))
    return grid.join(df, on=["MMSI", "ts"], how="left").orderBy("MMSI", "ts")

def add_future_lags(df, horizon=20):
    """Add future latitude and longitude columns for prediction targets."""
    df = df.withColumn("timestamp_epoch", F.unix_timestamp("ts").cast("long"))
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
    output_files = []

    for path in csv_files:

        # Load data
        df = load_csv_file(path)

        # Data processing

        df = drop_class_b(df)
        df = drop_unknown_label(df)
        df = filter_relevant_columns(df)
        df = filter_low_quality_mmsis(df)
        df = filter_outliers(df)
        df = resample_to_fixed_interval(df)
        df = forward_fill_features(df)
        df = clamp_features(df)
        df = interpolate_continuous_features(df)
        df = add_future_lags(df)

        all_data.append(df)

        # Saving processed data
        base = os.path.basename(path).replace("_fishing_labeled.csv", "_prod_ready.csv")
        output_file = write_single_output(df, base, output_folder)
        output_files.append(output_file)

    # Computes normalization stats based on training data
    if all_data:
        full_df = all_data[0]
        for df_part in all_data[1:]:
            full_df = full_df.unionByName(df_part)
        if args.mode == 'train':
            save_normalization_stats(full_df, "/ceph/project/gatehousep4/data/configs")

    return output_files

if __name__ == '__main__':
    preprocess_all_files()
    spark.stop()
