
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1. Start Spark
spark = (
    SparkSession.builder
    .appName("AirWiseIndia-SparkSQL")
    .master("local[*]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

BASE = Path(r"C:\Users\Asus\Desktop\AirWiseIndia")
DATA_PATH = (
    "file:///C:/Users/Asus/Desktop/"
    "AirWiseIndia/data/processed/aqi_clean.parquet"
)

OUTPUT_DIR = BASE / "outputs" / "analytics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load cleaned data
df = spark.read.parquet(DATA_PATH)

print("=" * 60)
print("AIRWISE INDIA - SPARK SQL ANALYSIS")
print("=" * 60)

print("Total records:", df.count())

# 3. Register a temporary SQL view
df.createOrReplaceTempView("air_quality")

# 4. Average AQI by city
print("\n1. TOP 10 CITIES BY AVERAGE AQI")

city_aqi = spark.sql("""
    SELECT
        City,
        COUNT(AQI) AS record_count,
        ROUND(AVG(AQI), 2) AS average_aqi
    FROM air_quality
    WHERE AQI IS NOT NULL
    GROUP BY City
    HAVING COUNT(AQI) >= 30
    ORDER BY average_aqi DESC
    LIMIT 10
""")

city_aqi.show(truncate=False)

city_aqi.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(str(OUTPUT_DIR / "top_cities_by_average_aqi"))

# 5. AQI category distribution using SQL
print("\n2. AQI CATEGORY DISTRIBUTION")

category_summary = spark.sql("""
    SELECT
        AQI_Bucket,
        COUNT(*) AS record_count,
        ROUND(100.0 * COUNT(*) /
            (SELECT COUNT(*) FROM air_quality), 2) AS percentage
    FROM air_quality
    GROUP BY AQI_Bucket
    ORDER BY record_count DESC
""")

category_summary.show(truncate=False)

category_summary.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(str(OUTPUT_DIR / "aqi_category_sql_summary"))

# 6. Monthly average AQI
print("\n3. MONTHLY AVERAGE AQI")

monthly_aqi = spark.sql("""
    SELECT
        DATE_FORMAT(TO_DATE(Date), 'yyyy-MM') AS year_month,
        COUNT(AQI) AS record_count,
        ROUND(AVG(AQI), 2) AS average_aqi
    FROM air_quality
    WHERE AQI IS NOT NULL
    GROUP BY DATE_FORMAT(TO_DATE(Date), 'yyyy-MM')
    ORDER BY year_month
""")

monthly_aqi.show(24, truncate=False)

monthly_aqi.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv(str(OUTPUT_DIR / "monthly_average_aqi"))

print("\nSpark SQL analysis completed.")
print("Results saved under:", OUTPUT_DIR)

spark.stop()
