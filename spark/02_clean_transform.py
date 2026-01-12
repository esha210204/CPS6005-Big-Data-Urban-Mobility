from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, when, trim

spark = SparkSession.builder.appName("GTFS Cleaning + Features").getOrCreate()

df = spark.read.csv(
    "file:///app/data/raw/CPS6005-Assessment 2_GTFS_Data.csv",
    header=True,
    inferSchema=True
)

# Clean common string issues
df = df.withColumn("Degree_of_congestion", trim(col("Degree_of_congestion"))) \
       .withColumn("SRI", trim(col("SRI"))) \
       .withColumn("speed", trim(col("speed")))

# Cast numeric fields
df = df.withColumn("speed_num", col("speed").cast("double")) \
       .withColumn("SRI_num", col("SRI").cast("double"))

# Drop rows missing core fields
df = df.dropna(subset=["arrival_time", "time", "speed_num", "SRI_num", "Degree_of_congestion"])

# Features
df = df.withColumn("hour_of_day", hour(col("arrival_time")))

df = df.withColumn(
    "is_peak",
    when((col("hour_of_day").between(7, 9)) | (col("hour_of_day").between(16, 18)), 1).otherwise(0)
)

df = df.withColumn(
    "congestion_label",
    when(col("Degree_of_congestion") == "Very smooth", 0)
    .when(col("Degree_of_congestion") == "Smooth", 1)
    .when(col("Degree_of_congestion") == "Moderate congestion", 2)
    .when(col("Degree_of_congestion") == "Heavy congestion", 3)
    .otherwise(2)
)

out_path = "file:///app/data/processed/gtfs_parquet"
df.write.mode("overwrite").partitionBy("hour_of_day").parquet(out_path)

print("✅ Cleaned rows:", df.count())
print("✅ Saved processed parquet to:", out_path)

spark.stop()
