from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import math

spark = SparkSession.builder.appName("GTFS Stats Tests").getOrCreate()
df = spark.read.parquet("file:///app/data/processed/gtfs_parquet")

# -----------------------------
# Helper: Welch's t-test (approx)
# -----------------------------
def welch_t_test(n1, mean1, var1, n2, mean2, var2):
    # t statistic
    se = math.sqrt(var1 / n1 + var2 / n2)
    t = (mean1 - mean2) / se if se != 0 else float("inf")

    # degrees of freedom (Welch–Satterthwaite)
    num = (var1 / n1 + var2 / n2) ** 2
    den = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
    dfree = num / den if den != 0 else float("inf")
    return t, dfree

# -----------------------------
# 1) Peak vs Off-peak speed test
# -----------------------------
# Use speed_num created in 02_clean_transform.py
stats = (
    df.select("is_peak", "speed_num")
      .where(col("speed_num").isNotNull())
      .groupBy("is_peak")
      .agg(
          {"speed_num": "count", "speed_num": "avg", "speed_num": "variance"}
      )
)

# Spark agg dict overwrites keys, so do it explicitly:
stats = (
    df.select("is_peak", "speed_num")
      .where(col("speed_num").isNotNull())
      .groupBy("is_peak")
      .agg(
          # aliases
          (col("is_peak")).alias("is_peak"),
      )
)
# Recompute with proper aliases
stats = (
    df.select("is_peak", "speed_num")
      .where(col("speed_num").isNotNull())
      .groupBy("is_peak")
      .agg(
          col("is_peak").alias("is_peak"),
      )
)

# Easier: use SQL
df.createOrReplaceTempView("gtfs")
peak_stats = spark.sql("""
SELECT
  is_peak,
  COUNT(speed_num) AS n,
  AVG(speed_num) AS mean,
  VAR_SAMP(speed_num) AS var
FROM gtfs
WHERE speed_num IS NOT NULL
GROUP BY is_peak
ORDER BY is_peak
""").collect()

# Expect two rows: is_peak=0 and is_peak=1
s0 = next(r for r in peak_stats if r["is_peak"] == 0)
s1 = next(r for r in peak_stats if r["is_peak"] == 1)

t, dfree = welch_t_test(s0["n"], s0["mean"], s0["var"], s1["n"], s1["mean"], s1["var"])

print("\n=== Welch t-test: speed (peak vs off-peak) ===")
print(f"Off-peak: n={s0['n']}, mean={s0['mean']:.4f}, var={s0['var']:.4f}")
print(f"Peak:     n={s1['n']}, mean={s1['mean']:.4f}, var={s1['var']:.4f}")
print(f"t-statistic = {t:.4f}")
print(f"degrees of freedom (approx) = {dfree:.2f}")
print("Note: We’re printing t + df; p-value can be computed in Python/scipy if needed.")

# -----------------------------
# 2) Correlations
# -----------------------------
# Pearson correlations (Spark)
corr_speed_sri = df.stat.corr("speed_num", "SRI_num")
corr_speed_time = df.stat.corr("speed_num", "time")  # if time kept numeric
corr_speed_trips = df.stat.corr("speed_num", "Number_of_trips")

print("\n=== Pearson correlations ===")
print(f"corr(speed_num, SRI_num) = {corr_speed_sri:.4f}")
print(f"corr(speed_num, time)    = {corr_speed_time:.4f}")
print(f"corr(speed_num, Number_of_trips) = {corr_speed_trips:.4f}")

spark.stop()
