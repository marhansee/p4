from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os


def filter_and_unique(col, mmsi):
    """
    Filters a DataFrame based on a specific 'MMSI' value and retrieves unique
    values from a specified column in the filtered DataFrame. The function
    also prints the unique values and displays the filtered DataFrame.

    Args:
        col: The name of the column to retrieve unique values from.
        mmsi: The value to filter the 'MMSI' column of the DataFrame.
    """
    df_filtered = df.filter(df['MMSI'] == mmsi)
    unique_values = [row[col] for row in df_filtered .select(col).distinct().collect()]
    print(f"Unique values in column '{col}': {unique_values}")
    df_filtered.show()

def fill_undefined_ship_type(df):
    """
    Fills undefined ship types in the given DataFrame by determining the appropriate ship type
    based on existing non-'Undefined' ship type values for the same MMSI.
    This function creates a temporary filled ship type column and then substitutes undefined values
    in 'Ship Type' with the determined values while preserving the original DataFrame structure.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame that contains a column 'Ship Type' for ship type
            and a column 'MMSI' for the unique ship identifier.

    Returns:
        pyspark.sql.DataFrame: A DataFrame where 'Ship Type' column values equal to 'Undefined'
            have been updated based on other non-'Undefined' values grouped by the corresponding MMSI.
    """
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
    """
    Identifies and processes vessel classes specific to fishing vessels.
    Filters for rows where the 'Ship Type' is categorized as 'Fishing', extracts
    distinct values from the 'Type of mobile' field, and displays the aggregated count of these
    distinct types.

    Args:
        df: pyspark.sql.DataFrame
            A DataFrame containing vessel information, which must include the columns
            'Ship Type' and 'Type of mobile'.

    """
    fishing_df = df.filter(col('Ship Type') == 'Fishing')

    # Identifies unique values in 'Type of mobile'
    fishing_df.select('Type of mobile').distinct().show()
    fishing_df.groupby('Type of mobile').count().show()


def extract_fishing_vessels(df):
    """
    Filters a DataFrame for rows where the "Ship Type" column equals "Fishing", and
    saves the filtered data to a new CSV file in a specified directory. The function
    ensures the target directory exists before writing the CSV file.

    Args:
        df: The input DataFrame containing ship data, with a column named "Ship Type".

    """
    df_filtered = df.filter(col('Ship Type') == 'Fishing')

    # Create data directory if it does not exist
    os.makedirs("data/fishing_vessel_data", exist_ok=True)
    new_file_name = data_path.replace(".csv", "fish.csv")

    # Writes filtered df as csv with new file name
    df_filtered.write.csv(f"data/fishing_vessel_data/{new_file_name}",)


if __name__ == '__main__':
    # Initialize the Spark session
    spark = SparkSession.builder \
        .appName("CSV loader") \
        .config("spark.sql.shuffle.partitions","200") \
        .getOrCreate()

    # Set the data path (ensure the path is correct relative to the project root)
    data_path = "data/aisdk-2025-02-03.csv"

    # Load the CSV file into a DataFrame
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(data_path)
