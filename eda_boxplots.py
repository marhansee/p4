import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col

# Create a SparkSession
spark = (SparkSession.builder.appName("AIS_EDA")
         .config("spark.driver.memory", "100g")
         .getOrCreate())

# Load the AIS data from multiple CSV files in the "data" folder
df = spark.read.csv("data/*.csv", header=True, inferSchema=True)

# Rename the column '# Timestamp' to 'Timestamp'
df = df.withColumnRenamed("# Timestamp", "Timestamp")

# Convert 'Timestamp' column to a timestamp type using the correct format
df = df.withColumn("Timestamp", to_timestamp("Timestamp", "dd/MM/yyyy HH:mm:ss"))

# Define the date range (e.g., December 2024)
start_date = "2024-12-01"
end_date   = "2024-12-31"  # end date is exclusive

# Filter the data for the specified period
df_filtered = df.filter((col("Timestamp") >= start_date) & (col("Timestamp") < end_date))

# Choose the column you want to plot
column_to_plot = "Latitude"

# Compute the min and max for the column
min_val = df_filtered.agg({column_to_plot: "min"}).collect()[0][0]
max_val = df_filtered.agg({column_to_plot: "max"}).collect()[0][0]

# Compute approximate quantiles: 25th, 50th, and 75th percentiles
q1, med, q3 = df_filtered.approxQuantile(column_to_plot, [0.25, 0.5, 0.75], 0.01)

print(f"Min: {min_val}, Q1: {q1}, Median: {med}, Q3: {q3}, Max: {max_val}")

# Now manually create a boxplot using the computed statistics

fig, ax = plt.subplots(figsize=(8, 2))  # A narrow height for a single box

# Draw the box from Q1 to Q3 at y=0.5
box_width = 0.4
box = patches.Rectangle((q1, 0.5 - box_width/2), q3 - q1, box_width,
                        edgecolor='black', facecolor='lightblue')
ax.add_patch(box)

# Draw the median line
ax.plot([med, med], [0.5 - box_width/2, 0.5 + box_width/2],
        color='red', linewidth=2)

# Draw whiskers: lines from min to Q1 and from Q3 to max
ax.plot([min_val, q1], [0.5, 0.5], color='black', linewidth=1.5)
ax.plot([q3, max_val], [0.5, 0.5], color='black', linewidth=1.5)

# Draw caps on the whiskers
cap_width = box_width / 2
ax.plot([min_val, min_val], [0.5 - cap_width/2, 0.5 + cap_width/2],
        color='black', linewidth=1.5)
ax.plot([max_val, max_val], [0.5 - cap_width/2, 0.5 + cap_width/2],
        color='black', linewidth=1.5)

# Set axis limits and labels
range_val = max_val - min_val
ax.set_xlim(min_val - 0.1 * range_val, max_val + 0.1 * range_val)
ax.set_ylim(0, 1)
ax.set_yticks([])  # Remove y-axis ticks for a clean look
ax.set_xlabel(column_to_plot)
ax.set_title(f"Boxplot for {column_to_plot}")

# Save the plot to a file so you can transfer it later
plt.savefig(f"boxplot_{column_to_plot}.png", dpi=300)
plt.close()

print(f"Boxplot saved as boxplot_{column_to_plot}.png")
