import sys

import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, lit, col
from pyspark.sql.types import IntegerType, LongType, DoubleType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# Spark session
spark = (SparkSession.builder
         .appName("AIS_EDA")
         .config("spark.driver.memory", "100g")
         .config("spark.driver.maxResultSize", "100g")
         .getOrCreate())

# Load from data/test/*.csv
df = spark.read.csv("data/test/*.csv", header=True, inferSchema=True)

# Rename and parse Timestamp
df = (df
      .withColumnRenamed("# Timestamp", "Timestamp")
      .withColumn("Timestamp",
                  to_timestamp("Timestamp", "dd/MM/yyyy HH:mm:ss")))

# Define proper filtering bounds and filter
start_ts = to_timestamp(lit("2024-12-01"), "yyyy-MM-dd")
end_ts   = to_timestamp(lit("2025-01-01"), "yyyy-MM-dd")

df_filtered = df.filter((col("Timestamp") >= start_ts) &
                        (col("Timestamp") <  end_ts))

# Identify numeric columns (exclude Timestamp)
numeric_cols = [
    f.name for f in df_filtered.schema.fields
    if isinstance(f.dataType, (IntegerType, LongType, FloatType, DoubleType))
    and f.name != "Timestamp"
]

# Drop rows with nulls in any numeric column
df_clean = df_filtered.na.drop(subset=numeric_cols)

clean_count = df_clean.count()
if clean_count == 0:
    print("All rows have nulls in the selected numeric columns. Exiting.")
    sys.exit(1)
else:
    print(f"{clean_count} rows remain after dropping nulls. Continuing…")

# Assemble into feature vector
assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol="features"
    # .setHandleInvalid("skip")   # alternative to dropping nulls
)
df_vec = assembler.transform(df_clean).select("features")

# Compute Pearson correlation
corr_row = Correlation.corr(df_vec, "features", method="pearson").head()
corr_matrix = corr_row[0].toArray()

# Plot and save (instead of show)
plt.figure(figsize=(10, 8))
im = plt.imshow(corr_matrix, interpolation="nearest", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, fraction=0.045, pad=0.04)

ticks = np.arange(len(numeric_cols))
plt.xticks(ticks, numeric_cols, rotation=45, ha="right")
plt.yticks(ticks, numeric_cols)

# Annotate each cell with its correlation coefficient
# fmt=".2f" will format numbers to two decimal places
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        plt.text(j, i, f"{corr_matrix[i, j]:.2f}",
                 ha="center", va="center", fontsize=8, color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")

plt.title("December 2024 AIS Data – Pearson Correlation")
plt.tight_layout()

# Save instead of show
plt.savefig("corr_matrix_dec2024_annotated.png", dpi=300)
plt.close()