
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --------------------------------------------------
# 1. Start Spark
# --------------------------------------------------
spark = (
    SparkSession.builder
    .appName("AirWiseIndia-ML-Preparation")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# --------------------------------------------------
# 2. Load cleaned data
# --------------------------------------------------
data_path = (
    "file:///C:/Users/Asus/Desktop/"
    "AirWiseIndia/data/processed/aqi_clean.parquet"
)

df = spark.read.parquet(data_path)

print("=" * 55)
print("AIRWISE INDIA - ML DATA PREPARATION")
print("=" * 55)

print("Total records:", df.count())

# --------------------------------------------------
# 3. Select features and target
# Exclude AQI to prevent target leakage
# --------------------------------------------------
feature_columns = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

# Rename PM2.5 for easier Spark column handling
df = df.withColumnRenamed("PM2.5", "PM25")

feature_columns = [
    "PM25", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

target_column = "AQI_Bucket"

# --------------------------------------------------
# 4. Check missing values in features
# --------------------------------------------------
print("\nMissing values in selected features:")

missing_counts = [
    F.count(
        F.when(
            F.col(c).isNull()
            | F.isnan(F.col(c).cast("double")),
            1
        )
    ).alias(c)
    for c in feature_columns
]

df.select(*missing_counts).show(truncate=False)

# --------------------------------------------------
# 5. Display target class distribution
# --------------------------------------------------
print("\nTarget class distribution:")

df.groupBy(target_column) \
    .count() \
    .orderBy(target_column) \
    .show(truncate=False)

# --------------------------------------------------
# 6. Display selected schema
# --------------------------------------------------
print("\nSelected feature columns:")
print(feature_columns)

print("\nTarget column:", target_column)

print("\nML data inspection completed.")

spark.stop()
