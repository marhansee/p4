from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os

spark = SparkSession.builder \
    .appName("CSV loader") \
    .config("spark.sql.shuffle.partitions","16") \
    .getOrCreate()

data_path = "data/aisdk-2025-02-03.csv"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(data_path)

def filter_mmsi(mmsi):
    df_filtered = df.filter(df['MMSI'] == mmsi)
    df_filtered.show()



def uniques_in_col(col):
    unique_values = [row[col] for row in df.select(col).distinct().collect()]
    print(f"Unique values in column '{col}': {unique_values}")


def filter_and_unique(col, mmsi):
    df_filtered = df.filter(df['MMSI'] == mmsi)
    unique_values = [row[col] for row in df_filtered .select(col).distinct().collect()]
    print(f"Unique values in column '{col}': {unique_values}")

filter_and_unique('Ship Type',219005954)
def fill_undefined_ship_type(df):
    '''
    Uses PySpark to find the Ship Type and fill in the 'Undefined' rows for each MMSI number.
    :param df: input dataframe
    :return: output dataframe where ship type is filled.
    '''

    # Creates new column 'Filled ship type' and fills in the ship type thats not equal to Undefined
    ship_type = ((df.filter(col('Ship Type') != 'Undefined'))
                 .groupby('MMSI')
                 .agg(first('Ship Type')
                 .alias('Filled ship type')))
    # Joins the 'Filled ship type' onto 'Undefined' ship types
    # Drops the 'Filled Ship Type' column to retain the original df structure
    df_filled = (
        df.join(ship_type, on ='MMSI', how='left')
        .withColumn(
            'Ship Type',
            when(col('Ship Type') == 'Undefined', col('Filled ship type'))
            .otherwise(col('Ship Type'))
        )
    .drop('Filled ship type')
    )
    df_filled.show()

def check_vessel_class(df):
    fishing_df = df.filter(col('Ship Type') == 'Fishing')
    fishing_df.select('Type of mobile').distinct().show()
    fishing_df.groupby('Type of mobile').count().show()

def extract_fishing_vessels(df):
    df_filtered = df.filter(col('Ship Type') == 'Fishing')
    os.makedirs("data/fishing_vessel_data", exist_ok=True)
    new_file_name = data_path.replace(".csv", "fish.csv")
    df_filtered.write.csv(f"data/fishing_vessel_data/{new_file_name}",)


# fill_undefined_ship_type(df)

# check_vessel_class(df)

# extract_fishing_vessels(df)