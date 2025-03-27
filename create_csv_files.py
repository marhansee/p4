import os
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import glob
import shutil


def extract_fishing_vessels(input_dir="data", output_dir="data/fishing_vessel_data"):
    """
    Extracts fishing vessels from all CSV files in the input directory
    and saves each as a new CSV in the output directory.

    Args:
        input_dir: Directory containing CSV files with vessel data
        output_dir: Directory to save the filtered CSV files

    Returns:
        List of output file paths created
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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

        # Filter for fishing vessels
        df_filtered = df.filter(col('Ship Type') == 'Fishing')

        # Create output filename based on input file
        base_filename = os.path.basename(data_path)
        new_file_name = base_filename.replace(".csv", "_fishing.csv")
        output_path = os.path.join(output_dir, new_file_name)

        # Create a temporary directory for the output
        temp_dir = output_path + "_temp"

        # Use coalesce(1) to ensure a single file output and write with header
        (df_filtered.coalesce(1).write.option("header", "true")
         .mode("overwrite").csv(temp_dir))

        # Find the CSV file in the temp directory (should be only one)
        csv_file = glob.glob(os.path.join(temp_dir, "*.csv"))[0]

        # Move the file to the desired output path
        shutil.move(csv_file, output_path)

        # Remove the temporary directory
        shutil.rmtree(temp_dir)

        # Count the number of rows in the filtered DataFrame
        count = df_filtered.count()

        print(f"Saved {count} fishing vessels to {output_path}")
        output_files.append(output_path)

    return output_files


if __name__ == '__main__':
    # Initialize the Spark session
    spark = SparkSession.builder \
        .appName("Fishing vessel data loader") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    # Set the directories
    input_dir = "data"
    output_dir = "data/fishing_vessel_data"

    # Call the function with input and output directories
    output_files = extract_fishing_vessels(input_dir, output_dir)

    print(f"Completed processing {len(output_files)} files")
