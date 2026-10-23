
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --------------------------------------------------
# 1. Create Spark session
# --------------------------------------------------
spark = (
    SparkSession.builder
    .appName("AirWiseIndia-EDA")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# --------------------------------------------------
# 2. Load cleaned Parquet dataset
# --------------------------------------------------
data_path = (
    "file:///C:/Users/Asus/Desktop/"
    "AirWiseIndia/data/processed/aqi_clean.parquet"
)

df = spark.read.parquet(data_path)

print("=" * 60)
print("AIRWISE INDIA - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 3. Basic dataset information
# --------------------------------------------------
print("\n1. DATASET INFORMATION")

print("Total records:", df.count())
print("Total columns:", len(df.columns))
print("Column names:", df.columns)

print("\nDataset schema:")
df.printSchema()

# --------------------------------------------------
# 4. Rename PM2.5 to avoid dot-related issues
# --------------------------------------------------
eda_df = df.withColumnRenamed("PM2.5", "PM25")

# --------------------------------------------------
# 5. Pollutant summary statistics
# --------------------------------------------------
pollutants = ["PM25", "PM10", "NO2", "SO2", "CO", "O3"]

print("\n2. POLLUTANT SUMMARY STATISTICS")
print("Count, mean, standard deviation, minimum and maximum")

eda_df.select(*pollutants).describe().show(
    truncate=False
)

# --------------------------------------------------
# 6. Missing-value analysis
# --------------------------------------------------
print("\n3. MISSING VALUES BY COLUMN")

missing_exprs = [
    F.count(
        F.when(
            F.col(f"`{c}`").isNull(), 1
        )
    ).alias(c)
    for c in df.columns
]

df.select(*missing_exprs).show(truncate=False)

# --------------------------------------------------
# 7. AQI category distribution
# --------------------------------------------------
print("\n4. AQI CATEGORY DISTRIBUTION")

df.groupBy("AQI_Bucket") \
    .count() \
    .orderBy("AQI_Bucket") \
    .show(truncate=False)

# --------------------------------------------------
# 8. AQI statistics
# --------------------------------------------------
print("\n5. AQI SUMMARY STATISTICS")

df.select("AQI").describe().show(truncate=False)

print("\nEDA analysis completed successfully.")

# --------------------------------------------------
# 9. Stop Spark
# --------------------------------------------------
spark.stop()
