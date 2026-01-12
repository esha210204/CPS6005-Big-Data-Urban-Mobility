from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, round as sround

spark = SparkSession.builder.appName("GTFS EDA + Export").getOrCreate()

# Load processed parquet written by 02_clean_transform.py
df = spark.read.parquet("file:///app/data/processed/gtfs_parquet")

print("✅ Rows in processed dataset:", df.count())
df.select("hour_of_day", "is_peak", "speed_num", "SRI_num", "Degree_of_congestion", "congestion_label").show(5)

# 1) Congestion distribution overall
cong_dist = (
    df.groupBy("Degree_of_congestion")
      .agg(count("*").alias("count"))
      .orderBy(col("count").desc())
)

# 2) Average speed by congestion category
speed_by_cong = (
    df.groupBy("Degree_of_congestion")
      .agg(sround(avg("speed_num"), 2).alias("avg_speed"))
      .orderBy(col("avg_speed").desc())
)

# 3) Congestion by hour of day (for time-series chart)
cong_by_hour = (
    df.groupBy("hour_of_day", "Degree_of_congestion")
      .agg(count("*").alias("count"))
      .orderBy("hour_of_day")
)

# 4) Peak vs off-peak summary
peak_summary = (
    df.groupBy("is_peak")
      .agg(
          sround(avg("speed_num"), 2).alias("avg_speed"),
          sround(avg("SRI_num"), 2).alias("avg_sri"),
          count("*").alias("rows")
      )
      .orderBy("is_peak")
)

print("\n=== Congestion distribution ===")
cong_dist.show(truncate=False)

print("\n=== Average speed by congestion ===")
speed_by_cong.show(truncate=False)

print("\n=== Peak vs Off-peak summary ===")
peak_summary.show(truncate=False)

# ---- Export small summaries for dashboards (CSV) ----
out_base = "file:///app/dashboards/data_out"
cong_dist.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_base}/congestion_distribution")
speed_by_cong.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_base}/speed_by_congestion")
cong_by_hour.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_base}/congestion_by_hour")
peak_summary.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{out_base}/peak_summary")

print(f"\n✅ Exported dashboard CSVs to: {out_base}/...")

spark.stop()
