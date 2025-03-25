import pyspark
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, \
    broadcast, when
import logging
import findspark
import os


logging.getLogger("py4j").setLevel(logging.ERROR)
findspark.init()
spark = SparkSession.builder.appName("P4_project").getOrCreate()


def load_data(file_name: str) -> DataFrame:
    """Load a csv-file into a PySpark DataFrame

    Args:
        file_name (string): Name of the csv-file

    Returns:
        A PySpark DataFrame 
    """

    try:
        data = os.path.join(os.path.dirname(__file__), file_name)
        df = spark.read.csv(data, header=True, inferSchema=True)
        return df
    
    except FileNotFoundError:
        print("Error: The specified CSV file could not be found")


def join_meta_data(meta_data: DataFrame, ais_data: DataFrame):
    """Function does three things:
    (1) Broadcasts the 'Gear Type' column from metadata and joins it with AIS data 
    on the 'MMSI' column. 
    
    (2) If 'Destination'-column is 'TRAWL FISHING', then 
    'TRAWLERS' gets added to the 'Gear Type' for the respective row.

    (3) Replace missing values in 'Gear Type'-column with "UNKNOWN" 

    Args:
        - meta_data (DataFrame): DataFrame with the meta data
        - ais_data (DataFrame): DataFrame with the AIS data

    Returns:
        - df (DataFrame): AIS DataFrame with the added 'Gear Type'-column 
    """

    # Select columns for broadcasting
    meta_selected = meta_data.select("MMSI", "Gear Type")
    df = ais_data.join(broadcast(meta_selected), on="MMSI", how="left")


    # Update 'Gear Type' where 'Destination' is 'TRAWL FISHING'
    df = df.withColumn(
        "Gear Type",
        when(col("Destination") == "TRAWL FISHING", "TRAWLERS").otherwise(col("Gear Type"))
    )

    # Replace missing values with "UNKNOWN" in Gear Type
    df = df.fillna({"Gear Type": "UNKNOWN"})
    return df


def main():
    # Load datasets
    ais_df = load_data('AIS FIL NAVN') # AIS data here
    meta_df = load_data('METADATA FIL NAVN') # Meta data here

    # Broadcast join metadata
    df = join_meta_data(meta_data=meta_df, ais_data=ais_df)
    df.show(5)
    




if __name__ == '__main__':
    main()