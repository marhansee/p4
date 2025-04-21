import pyspark
from pyspark.sql import SparkSession, DataFrame
import logging
import findspark
from pyspark.sql.functions import to_timestamp, col

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
    return df

def drop_unknown_label(df):
    pass




def main():
    # Load data
    data_path = "data/aisdk-2025-01-01_fishing_labeled.csv"
    df = load_data(data_path)

    df = drop_duplicates(df)
    df.show(5)

    # df.show(5)


if __name__ == '__main__':
    main()