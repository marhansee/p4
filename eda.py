from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .appName("CSV loader") \
    .config("spark.sql.shuffle.partitions","16") \
    .getOrCreate()

data_path = "data/aisdk-2025-02-26.csv"

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(data_path)

def filter_mmsi(mmsi):
    df_filtered = df.filter(df['MMSI'] == mmsi)
    df_filtered.show()

# filter_mmsi(219013885)

def uniques_in_col(col):
    unique_values = [row[col] for row in df.select(col).distinct().collect()]
    print(f"Unique values in column '{col}': {unique_values}")

def filter_and_unique(col, mmsi):
    df_filtered = df.filter(df['MMSI'] == mmsi)
    unique_values = [row[col] for row in df_filtered .select(col).distinct().collect()]
    print(f"Unique values in column '{col}': {unique_values}")

# filter_and_unique('Ship Type',636022800)
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



fill_undefined_ship_type(df)



# df.show(50)