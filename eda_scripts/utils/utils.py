import pyspark
import warnings
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import regexp_replace, col, \
    substring, broadcast, col, isnan, when, count

import os
import pandas as pd


def load_data_pyspark(spark, file_name: str) -> DataFrame:
    """Load a csv-file into a PySpark DataFrame

    Args:
        - spark (SparkSession): Active Spark session.
        - file_name (string): Name of the csv-file

    Returns:
        A PySpark DataFrame 
    """

    try:
        data = os.path.join(os.path.dirname(__file__), file_name)
        df = spark.read.csv(data, header=True, inferSchema=True)

        # Remove empty columns
        if "_c0" in df.columns:
            df = df.drop("_c0")
        return df
    
    except FileNotFoundError:
        print("Error: The specified CSV file could not be found")


def load_data_pandas(file_name: str) -> DataFrame:
    """Load a csv-file into a Pandas DataFrame

    Args:
        file_name (string): Name of the csv-file

    Returns:
        A Pandas DataFrame 
    """

    try:
        data = os.path.join(os.path.dirname(__file__), file_name)
        df = pd.read_csv(data)
        return df
    
    except FileNotFoundError:
        print("Error: The specified CSV file could not be found")



def standardize_dateformat(df: DataFrame, time_feature: str):
    """Standardizes the date format in the given DataFrame column by replacing 
    hyphens ('-') with slashes ('/'). The new date format is yyyy/MM/dd.

    Args:
        - df (DataFrame): The input PySpark DataFrame.
        - time_feature (str): The name of the column containing date values.

    Returns:
        - DataFrame | None: The transformed DataFrame if successful, otherwise None.
    """
    if not isinstance(df, pyspark.sql.DataFrame):
        raise AssertionError("The DataFrame must be a PySpark DataFrame")
    
    try:
        # Replace '-' with '/' for the specified time feature
        df = df.withColumn(
            time_feature, regexp_replace(col(time_feature), "-", "/")
        )

        # Only extract the characters 'YYYY/MM/DD' from time feature
        start_idx = 1
        end_idx = 10
        df = df.withColumn(
            time_feature, substring(col(time_feature), start_idx, end_idx)
        )

        return df

    except pyspark.errors.exceptions.captured.AnalysisException:
        print(f"Error: The column {time_feature} cannot be found in the DataFrame")
        return None

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