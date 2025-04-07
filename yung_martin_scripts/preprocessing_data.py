import pyspark
import warnings
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, broadcast, when, lit
import logging
import findspark
import os

# Load utils
from utils.utils import load_data_pyspark


logging.getLogger("py4j").setLevel(logging.ERROR)
findspark.init()
spark = SparkSession.builder \
    .appName("P4_project") \
    .config("spark.sql.shuffle.partitions","200") \
    .getOrCreate()

def join_meta_data(meta_data: DataFrame, ais_data: DataFrame):
    """Function does three things:
    (1) Broadcasts the 'Gear Type' column from metadata and joins it with AIS data 
    on the 'MMSI' column. 


    Args:
        - meta_data (DataFrame): DataFrame with the meta data
        - ais_data (DataFrame): DataFrame with the AIS data

    Returns:
        - df (DataFrame): AIS DataFrame with the added 'Gear Type'-column 
    """

    # Select columns for broadcasting
    meta_selected = meta_data.select("MMSI", "Gear Type")
    df = ais_data.join(broadcast(meta_selected), on="MMSI", how="left")

    return df

def label_data(df: DataFrame, target_name="trawling"):
    df = df.withColumn(target_name, 
                       when((col('Navigational status') == 'Engaged in fishing') & 
                            (col('Gear Type') == 'TRAWLERS'), lit(1))
                       .otherwise(lit(0)))
    return df
    

def main():
    # Define data paths
    ais_path = os.path.join(os.path.dirname(__file__), 'ais-2025-01-13_fishing.csv')
    meta_path = os.path.join(os.path.dirname(__file__), 'meta_data.csv')

    # Load datasets
    ais_df = load_data_pyspark(spark=spark, file_name=ais_path)
    meta_df = load_data_pyspark(spark=spark, file_name=meta_path)
    ais_df.cache()
    meta_df.cache()

    # Broadcast join metadata with AIS data
    df = join_meta_data(meta_data=meta_df, ais_data=ais_df)

    # Update 'Gear Type' where 'Destination'-feature is 'TRAWL FISHING'
    df = df.withColumn(
        "Gear Type", when(col("Destination") == "TRAWL FISHING", "TRAWLERS") \
            .otherwise(col("Gear Type"))
    )

    # Replace missing values with "UNKNOWN" in Gear Type
    df = df.fillna({"Gear Type": "UNKNOWN"})
 
    # Remove Class B's
    df = df.filter(col('Type of mobile') != 'Class B')

    # Label the data
    df = label_data(df)

    # Export as CSV
    df.toPandas().to_csv('preprocessed_data.csv')




if __name__ == '__main__':
    main()