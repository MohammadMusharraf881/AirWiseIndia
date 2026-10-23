
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Start Spark locally
spark = (
    SparkSession.builder
    .appName("AirWiseIndia")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Load dataset
path = "file:///C:/Users/Asus/Desktop/AirWiseIndia/data/raw/city_day.csv"

df = spark.read.csv(
    path,
    header=True,
    inferSchema=True
)

# Inspect dataset
print("Total records:", df.count())
print("Total columns:", len(df.columns))

print("\nDataset schema:")
df.printSchema()

print("\nFirst five records:")
df.show(5, truncate=False)


# 1. Missing values in each column
print("\n--- Missing Values ---")

missing = df.select([
    F.count(
        F.when(F.col(f"`{c}`").isNull(), 1)
    ).alias(c)
    for c in df.columns
])

missing.show(truncate=False)

# 2. Exact duplicate records
print("\n--- Duplicate Records ---")

duplicates = df.count() - df.dropDuplicates().count()
print("Exact duplicate records:", duplicates)

# 3. AQI category distribution
print("\n--- AQI Category Distribution ---")

df.groupBy("AQI_Bucket") \
  .count() \
  .orderBy("AQI_Bucket") \
  .show()

spark.stop()
