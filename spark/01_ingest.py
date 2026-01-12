from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("GTFS Ingestion") \
    .getOrCreate()

df = spark.read.csv(
    "file:///app/data/raw/CPS6005-Assessment 2_GTFS_Data.csv",
    header=True,
    inferSchema=True
)

df.printSchema()
df.show(5)
print("Total records:", df.count())

spark.stop()
