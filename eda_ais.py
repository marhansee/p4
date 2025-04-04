from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import os

spark = SparkSession.builder.appName("eda_ais").getOrCreate()

df = spark.read.option("header",True).csv("data")

df.describe().show()