import pandas as pd
import os
from utils.utils import load_data_pandas, load_data_pyspark
import missingno as msno
import matplotlib.pyplot as plt
import pyspark
import warnings
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, isnan, when, count
import logging
import findspark


logging.getLogger("py4j").setLevel(logging.ERROR)
findspark.init()
spark = SparkSession.builder \
    .appName("P4_project") \
    .config("spark.driver.memory", "15g") \
    .getOrCreate()

def missing_values_description(df: DataFrame):

    # Check for missing values
    print("Missing values:")
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
    ).show()

    # Print rows with NULL's in Gear Type feature
    print("Rows with MV's in Gear Type-feature:")
    df.where(col("Gear Type").isNull()).show(5)

    print("Missing values without Class B's:")
    df = df.filter(col('Type of mobile') != 'Class B')
    # df.where(col("Gear Type").isNull()).show(5)

       # Check for missing values
    print("Missing values:")
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
    ).show()



def visualize_MV_matrix(df: pd.DataFrame):
    """Visualize missing values in a Matrix-Form"""
    msno.matrix(df, figsize=(10,8))
    plt.tight_layout()
    plt.show()

def visualize_MV_heatmap(df: pd.DataFrame):
    """Visualize missing values in as a heatmap."""
    msno.heatmap(df, figsize=(10,8))
    plt.tight_layout()
    plt.show()


def visualize_MV_barplot(df: pd.DataFrame):
    """Visualize missing values in as a barplot."""
    msno.bar(df, figsize=(10,8))
    plt.tight_layout()
    plt.show()

def main():
    data_path = os.path.join(os.path.dirname(__file__), 'mv_analysis_data.csv')
    df = load_data_pandas(data_path)
    df_spark = load_data_pyspark(spark=spark, file_name=data_path)
    df_spark.cache()
    # print(df.head(5))
    # print(df.isnull().sum())

    ############# PLOTTING ##################
    # visualize_MV_matrix(df)
    # visualize_MV_heatmap(df)
    # visualize_MV_barplot(df)
    missing_values_description(df_spark)


if __name__ == '__main__':
    main()