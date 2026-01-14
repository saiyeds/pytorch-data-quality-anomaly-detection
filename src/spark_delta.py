
from pyspark.sql import SparkSession

def get_spark():
    return SparkSession.builder.appName("AnomalyDetection").getOrCreate()

def read_delta(spark, path):
    return spark.read.format("delta").load(path)

def write_delta(df, path):
    df.write.format("delta").mode("append").save(path)
