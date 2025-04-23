import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col

# Create a SparkSession
spark = (SparkSession.builder.appName("AIS_EDA")
         .config("spark.driver.memory", "100g")
         .config("spark.driver.maxResultSize", "100g")
         .getOrCreate())

# Load the AIS data from multiple CSV files in the "data" folder
df = spark.read.csv("data/*.csv", header=True, inferSchema=True)

# Rename the column '# Timestamp' to 'Timestamp'
df = df.withColumnRenamed("# Timestamp", "Timestamp")

# Convert 'Timestamp' column to a timestamp type using the correct format
df = df.withColumn("Timestamp", to_timestamp("Timestamp", "dd/MM/yyyy HH:mm:ss"))

# Define the date range (e.g., December 2024)
start_date = "2024-12-01"
end_date = "2024-12-31"  # end date is exclusive

# Filter the data for the specified period
df_filtered = df.filter((col("Timestamp") >= start_date) & (col("Timestamp") < end_date))

# Drop the 'Timestamp' column as it's no longer needed
df_filtered = df_filtered.drop("Timestamp")

# Convert the filtered Spark DataFrame to a Pandas DataFrame
pdf = df_filtered.toPandas()

# List of features to plot
columns_to_plot = ["Longitude", "Latitude", "ROT", "SOG", "COG", "Heading", "Width", "Length", "Draught"]

# Create and save a boxplot using Matplotlib's built-in function for each feature
for col_name in columns_to_plot:
    fig, ax = plt.subplots(figsize=(8, 2))  # Set the figure size

    # Select the data for the current column and drop missing values if any
    data = pdf[col_name].dropna()

    # Create the boxplot using Matplotlib's boxplot function
    ax.boxplot(data, patch_artist=True)

    # Set the title and the x-axis label to the current column name
    ax.set_title(f"Boxplot for {col_name}")
    ax.set_xticklabels([col_name])

    # Save the boxplot as an image file
    plt.savefig(f"boxplot_{col_name}.png", dpi=300)
    plt.close()

    print(f"Boxplot saved as boxplot_{col_name}.png")
