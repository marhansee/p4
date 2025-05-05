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
    description="Process AIS data with resampling, interpolation, and lag generation. Prints only null counts before and after interpolation."
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


def preprocess(input_folder, output_folder):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        return []

    output_files = []
    all_data = []

    for path in csv_files:
        df = spark.read.option("header", True).option("inferSchema", True).csv(path)

        df = df.withColumnRenamed("# Timestamp", "ts_raw")
        df = df.withColumn("ts", F.to_timestamp("ts_raw", "dd/MM/yyyy HH:mm:ss")).drop("ts_raw")
        df = df.dropDuplicates(["MMSI", "ts"])

        # Print nulls before interpolation
        print(f"\n{os.path.basename(path)} - Nulls BEFORE interpolation:")
        df.select([F.count(F.when(F.col(c).isNull(), c)).alias(f"{c}_nulls") for c in continuous_cols]).show()

        # Monthly IQR outlier filtering for Latitude & Longitude
        df = df.withColumn("month", F.date_format("ts", "yyyy-MM"))
        iqr_cols = ["Latitude", "Longitude"]
        quantile_exprs = [F.expr(f"percentile_approx({c}, array(0.25,0.75), 10000)").alias(f"{c}_quants") for c in iqr_cols]
        month_quants = df.groupBy("month").agg(*quantile_exprs)
        for c in iqr_cols:
            month_quants = month_quants.withColumn(f"{c}_q1", F.col(f"{c}_quants")[0]) \
                                       .withColumn(f"{c}_q3", F.col(f"{c}_quants")[1]) \
                                       .drop(f"{c}_quants")
        df = df.join(month_quants, "month", "left")
        for c in iqr_cols:
            iqr = F.col(f"{c}_q3") - F.col(f"{c}_q1")
            df = df.filter(F.col(c).between(F.col(f"{c}_q1") - 1.5 * iqr, F.col(f"{c}_q3") + 1.5 * iqr))
        df = df.drop(*[f"{c}_q1" for c in iqr_cols], *[f"{c}_q3" for c in iqr_cols], "month")

        # Forward-fill static features
        w_static = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(Window.unboundedPreceding, 0)
        for col in static_cols:
            df = df.withColumn(col, F.last(col, ignorenulls=True).over(w_static))

        # Static outlier filtering
        width_q = df.stat.approxQuantile("Width", [0.01, 0.99], 0.001)
        length_q = df.stat.approxQuantile("Length", [0.01, 0.99], 0.001)
        df = df.filter(F.col("Width").between(width_q[0], width_q[1]) &
                       F.col("Length").between(length_q[0], length_q[1]))

        # Forward-fill dynamic features
        w_dyn = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(Window.unboundedPreceding, 0)
        for col in ["ROT", "Heading", "Draught"]:
            df = df.withColumn(col, F.last(col, ignorenulls=True).over(w_dyn))

        # Clamp ROT and SOG
        df = df.withColumn("SOG", F.when(F.col("SOG") < 0, 0).when(F.col("SOG") > 40, 40).otherwise(F.col("SOG")))
        df = df.withColumn("ROT", F.when(F.col("ROT") < -90, -90).when(F.col("ROT") > 90, 90).otherwise(F.col("ROT")))

        # Generate resample grid
        bounds = df.groupBy("MMSI").agg(F.min("ts").alias("min_ts"), F.max("ts").alias("max_ts"))
        grid = bounds.select("MMSI", F.explode(F.sequence("min_ts", "max_ts", F.expr("INTERVAL 10 seconds"))).alias("ts"))
        df = grid.join(df, on=["MMSI", "ts"], how="left").orderBy("MMSI", "ts")

        for col in static_cols:
            df = df.withColumn(col, F.last(col, ignorenulls=True).over(w_static))

        # Interpolation
        w_prev = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(Window.unboundedPreceding, 0)
        w_next = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(0, Window.unboundedFollowing)
        for col in continuous_cols:
            df = df.withColumn(f"{col}_prev", F.last(col, ignorenulls=True).over(w_prev))
            df = df.withColumn(f"ts_{col}_prev", F.last(F.when(F.col(col).isNotNull(), F.col("ts")), ignorenulls=True).over(w_prev))
            df = df.withColumn(f"{col}_next", F.first(F.when(F.col(col).isNotNull(), F.col(col)), ignorenulls=True).over(w_next))
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

        # Print nulls after interpolation
        print(f"{os.path.basename(path)} - Nulls AFTER interpolation:")
        df.select([F.count(F.when(F.col(c).isNull(), c)).alias(f"{c}_nulls") for c in continuous_cols]).show()

        # Drop helper columns
        helper_cols = [f for c in continuous_cols for f in (
            f"{c}_prev", f"ts_{c}_prev", f"{c}_next", f"ts_{c}_next", f"frac_{c}", f"time_diff_{c}"
        )]
        df = df.drop(*helper_cols)



        # Add epoch and future lags
        df = df.withColumn("timestamp_epoch", F.unix_timestamp("ts").cast("long"))
        window_lag = Window.partitionBy("MMSI").orderBy("timestamp_epoch")
        for i in range(1, 21):
            df = df.withColumn(f"future_lat_{i}", F.lead("Latitude", i).over(window_lag))
            df = df.withColumn(f"future_lon_{i}", F.lead("Longitude", i).over(window_lag))

        # Select output schema
        base_cols = ["timestamp_epoch", "MMSI"] + continuous_cols + static_cols
        future_cols = [f for i in range(1, 21) for f in (f"future_lat_{i}", f"future_lon_{i}")]
        df = df.select(*(base_cols + future_cols))

        # Store for final stat computation
        all_data.append(df)

        # Write output
        base = os.path.basename(path).replace("_fishing_labeled.csv", "_prod_ready.csv")
        temp_dir = os.path.join(output_folder, base + "_tmp")
        df.coalesce(1).write.option("header", True).mode("overwrite").csv(temp_dir)

        part_files = glob.glob(os.path.join(temp_dir, "part-*.csv"))
        if not part_files:
            raise FileNotFoundError(f"No CSV part file found in {temp_dir}")
        tmp_csv = part_files[0]
        dest = os.path.join(output_folder, base)
        shutil.move(tmp_csv, dest)
        shutil.rmtree(temp_dir)
        output_files.append(dest)

    # Save global normalization stats
    if all_data:
        full_df = all_data[0]
        for df_part in all_data[1:]:
            full_df = full_df.unionByName(df_part)
        save_normalization_stats(full_df, "/ceph/project/gatehousep4/data/configs")

    return output_files

# Run
if __name__ == '__main__':
    labeled = preprocess(input_folder, output_folder)
    spark.stop()
