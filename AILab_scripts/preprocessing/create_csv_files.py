import os
import glob
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast, when, lit


def join_meta_data(meta_data, ais_data):
    """
    Broadcasts the 'Gear Type' column from the meta data and joins it with AIS data on the 'MMSI' column.

    Args:
        meta_data (DataFrame): DataFrame with the meta data.
        ais_data (DataFrame): DataFrame with the AIS data.

    Returns:
        DataFrame: AIS DataFrame with the added 'Gear Type' column.
    """
    meta_selected = meta_data.select("MMSI", "Gear Type")
    df = ais_data.join(broadcast(meta_selected), on="MMSI", how="left")
    return df


def label_data(df, target_name="trawling"):
    """
    Adds a new column to the DataFrame, setting the value to 1 when:
      - 'Navigational status' equals 'Engaged in fishing' and
      - 'Gear Type' equals 'TRAWLERS'.
    Otherwise, the value is set to 0.

    Args:
        df (DataFrame): Input AIS DataFrame.
        target_name (str): Name of the new label column.

    Returns:
        DataFrame: DataFrame with the new label column added.
    """
    df = df.withColumn(
        target_name,
        when(
            (col("Navigational status") == "Engaged in fishing") & (col("Gear Type") == "TRAWLERS"),
            lit(1)
        ).otherwise(lit(0))
    )
    return df


def process_fishing_vessels_with_meta(input_dir="data/raw_data",
                                      output_dir="data",
                                      meta_data_path="data/meta_data.csv"):
    """
    Processes AIS data by:
      1. Filtering for fishing vessels ('Ship Type' == 'Fishing')
      2. Joining with metadata (on the 'MMSI' column)
      3. Labeling data (adding a column to flag trawling vessels)
      4. Saving the processed DataFrame as a CSV file.

    Args:
        input_dir (str): Directory containing the AIS CSV files.
        output_dir (str): Directory to save the processed CSV files.
        meta_data_path (str): Path to the CSV file containing meta data.

    Returns:
        List[str]: List of output file paths created.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load meta data once
    meta_data = spark.read.csv(meta_data_path, header=True, inferSchema=True)

    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return []

    print(f"Found {len(csv_files)} CSV files to process")
    output_files = []

    for data_path in csv_files:
        print(f"Processing {data_path}...")

        # Read the CSV file
        df = spark.read.csv(data_path, header=True, inferSchema=True)

        # Step 1: Filter for fishing vessels
        df_filtered = df.filter(col("Ship Type") == "Fishing")

        # Step 2: Join with meta data
        df_joined = join_meta_data(meta_data, df_filtered)

        # Step 3: Label the data
        df_labeled = label_data(df_joined)

        # Create output filename based on input file
        base_filename = os.path.basename(data_path)
        new_file_name = base_filename.replace(".csv", "_fishing_labeled.csv")
        output_path = os.path.join(output_dir, new_file_name)

        # Create a temporary directory for writing output
        temp_dir = output_path + "_temp"

        # Write the processed DataFrame as a single CSV file with header
        (df_labeled.coalesce(1)
         .write
         .option("header", "true")
         .mode("overwrite")
         .csv(temp_dir))

        # Find the CSV file in the temporary directory (should be only one)
        csv_file = glob.glob(os.path.join(temp_dir, "*.csv"))[0]

        # Move the file to the desired output path
        shutil.move(csv_file, output_path)

        # Remove the temporary directory
        shutil.rmtree(temp_dir)

        count = df_labeled.count()
        print(f"Saved {count} records to {output_path}")
        output_files.append(output_path)

    return output_files


if __name__ == '__main__':
    # Initialize the Spark session
    spark = SparkSession.builder \
        .appName("Fishing Vessel Data Processor") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    # Set directories and meta data path (update paths as necessary)
    input_dir = "data"
    output_dir = "data/fishing_vessel_data"
    meta_data_path = "data/metadata/meta_data.csv"

    # Process the AIS files: filter, join meta data, and label data
    output_files = process_fishing_vessels_with_meta(input_dir, output_dir, meta_data_path)

    print(f"Completed processing {len(output_files)} files")

    # Stop the Spark session when done
    spark.stop()

