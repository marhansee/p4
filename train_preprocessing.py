import os
import glob
import json
import shutil
import argparse
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

# Parse config file
parser = argparse.ArgumentParser(description="Preprocess AIS training data and compute normalization stats")
parser.add_argument('--config', required=True, help="Path to config file with input/output paths")
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

input_folder = config['input_folder']
output_folder = config['output_folder']
output_stats_file = config['output_stats_file']

# Spark session
spark = SparkSession.builder \
    .appName("AIS Training Preprocessing") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Feature lists
continuous_cols = ["Latitude", "Longitude", "ROT", "SOG", "COG", "Heading", "Draught"]
static_cols = ["Width", "Length", "trawling"]

def preprocess_and_compute_stats(input_folder, output_folder):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f"No CSVs found in {input_folder}")
        return []

    all_data = None

    for path in csv_files:
        print(f"Processing {path}...")
        df = spark.read.option("header", True).option("inferSchema", True).csv(path)
        df = df.withColumnRenamed("# Timestamp", "ts_raw")
        df = df.withColumn("ts", F.to_timestamp("ts_raw", "dd/MM/yyyy HH:mm:ss")).drop("ts_raw")
        df = df.dropDuplicates().dropDuplicates(["MMSI", "ts"])

        # Monthly IQR outlier removal
        iqr_cols = ["Latitude", "Longitude"]
        df = df.withColumn("month", F.date_format("ts", "yyyy-MM"))
        quantile_exprs = [F.expr(f"percentile_approx({c}, array(0.25,0.75), 10000)").alias(f"{c}_quants") for c in iqr_cols]
        month_quants = df.groupBy("month").agg(*quantile_exprs)
        for c in iqr_cols:
            month_quants = month_quants.withColumn(f"{c}_q1", F.col(f"{c}_quants")[0])
            month_quants = month_quants.withColumn(f"{c}_q3", F.col(f"{c}_quants")[1]).drop(f"{c}_quants")
        df = df.join(month_quants, "month", "left")
        for c in iqr_cols:
            iqr = F.col(f"{c}_q3") - F.col(f"{c}_q1")
            df = df.filter(F.col(c).between(F.col(f"{c}_q1") - 1.5 * iqr, F.col(f"{c}_q3") + 1.5 * iqr))
        df = df.drop(*[f"{c}_q1" for c in iqr_cols], *[f"{c}_q3" for c in iqr_cols], "month")

        df = df.filter(F.col("Latitude").isNotNull() & F.col("Longitude").isNotNull() &
                       F.col("SOG").isNotNull() & F.col("COG").isNotNull())
        df = df.withColumn("SOG", F.when(F.col("SOG") < 0, 0).when(F.col("SOG") > 40, 40).otherwise(F.col("SOG")))
        df = df.withColumn("ROT", F.when(F.col("ROT") < -90, -90).when(F.col("ROT") > 90, 90).otherwise(F.col("ROT")))

        width_q = df.stat.approxQuantile("Width", [0.01, 0.99], 0.001)
        length_q = df.stat.approxQuantile("Length", [0.01, 0.99], 0.001)
        df = df.filter(F.col("Width").between(width_q[0], width_q[1]) & F.col("Length").between(length_q[0], length_q[1]))

        bounds = df.groupBy("MMSI").agg(F.min("ts").alias("min_ts"), F.max("ts").alias("max_ts"))
        grid = bounds.select("MMSI", F.explode(F.sequence("min_ts", "max_ts", F.expr("INTERVAL 10 seconds"))).alias("ts"))
        df = grid.join(df, on=["MMSI", "ts"], how="left").orderBy("MMSI", "ts")

        w_prev = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(Window.unboundedPreceding, 0)
        w_next = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(0, Window.unboundedFollowing)
        for col in continuous_cols:
            df = df.withColumn(f"{col}_prev", F.last(col, ignorenulls=True).over(w_prev))
            df = df.withColumn(f"ts_{col}_prev", F.last(F.when(F.col(col).isNotNull(), F.col("ts")), ignorenulls=True).over(w_prev))
            df = df.withColumn(f"{col}_next", F.first(F.when(F.col(col).isNotNull(), F.col(col)), ignorenulls=True).over(w_next))
            df = df.withColumn(f"ts_{col}_next", F.first(F.when(F.col(col).isNotNull(), F.col("ts")), ignorenulls=True).over(w_next))
            df = df.withColumn(f"frac_{col}", F.when((F.unix_timestamp(f"ts_{col}_next") != F.unix_timestamp(f"ts_{col}_prev")) &
                                                      F.col(f"ts_{col}_prev").isNotNull() &
                                                      F.col(f"ts_{col}_next").isNotNull(),
                                                      (F.unix_timestamp("ts") - F.unix_timestamp(f"ts_{col}_prev")) /
                                                      (F.unix_timestamp(f"ts_{col}_next") - F.unix_timestamp(f"ts_{col}_prev"))))
            df = df.withColumn(f"time_diff_{col}", F.unix_timestamp(f"ts_{col}_next") - F.unix_timestamp(f"ts_{col}_prev"))
            df = df.withColumn(col, F.when(F.col(col).isNull() &
                                           F.col(f"{col}_prev").isNotNull() &
                                           F.col(f"{col}_next").isNotNull() &
                                           F.col(f"frac_{col}").isNotNull() &
                                           (F.col(f"time_diff_{col}") <= 300),
                                           F.col(f"{col}_prev") + (F.col(f"{col}_next") - F.col(f"{col}_prev")) * F.col(f"frac_{col}"))
                                      .otherwise(F.col(col)))
        df = df.drop(*[f for c in continuous_cols for f in (f"{c}_prev", f"ts_{c}_prev", f"{c}_next", f"ts_{c}_next", f"frac_{c}", f"time_diff_{c}")])

        w_static = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(Window.unboundedPreceding, 0)
        for col in static_cols:
            df = df.withColumn(col, F.last(col, ignorenulls=True).over(w_static))

        w_ffill = Window.partitionBy("MMSI").orderBy("ts").rowsBetween(Window.unboundedPreceding, 0)
        w_bfill = Window.partitionBy("MMSI").orderBy(F.col("ts").desc()).rowsBetween(Window.unboundedPreceding, 0)
        for col in continuous_cols:
            df = df.withColumn(col, F.coalesce(F.col(col), F.last(col, ignorenulls=True).over(w_ffill), F.last(col, ignorenulls=True).over(w_bfill)))

        all_data = df if all_data is None else all_data.unionByName(df)

    stats_exprs = [F.mean(c).alias(f"{c}_mean") for c in continuous_cols] + [F.stddev(c).alias(f"{c}_std") for c in continuous_cols]
    stats_row = all_data.select(*stats_exprs).collect()[0]
    norm_stats = {col: {"mean": stats_row[f"{col}_mean"], "std": stats_row[f"{col}_std"]} for col in continuous_cols}

    with open(output_stats_file, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"Normalization stats saved to {output_stats_file}")
    return norm_stats

if __name__ == '__main__':
    preprocess_and_compute_stats(input_folder, output_folder)
    spark.stop()
