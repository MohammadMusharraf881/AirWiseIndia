from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder
    .appName("AirWiseIndia-DataCleaning")
    .master("local[*]")
    .getOrCreate()
)

input_path = (
    "file:///C:/Users/Asus/Desktop/"
    "AirWiseIndia/data/raw/city_day.csv"
)

output_path = (
    "file:///C:/Users/Asus/Desktop/"
    "AirWiseIndia/data/processed/aqi_clean.parquet"
)

# 1. Load the original dataset
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(input_path)
)

print("Original row count:", df.count())

# 2. Check exact duplicate rows
duplicates = df.count() - df.dropDuplicates().count()
print("Exact duplicate rows:", duplicates)

# 3. Remove duplicates
clean = df.dropDuplicates()

# 4. Remove rows missing AQI or AQI category
clean = clean.filter(
    F.col("AQI").isNotNull()
    & F.col("AQI_Bucket").isNotNull()
)

# 5. Keep the six valid categories
valid_categories = [
    "Good", "Satisfactory", "Moderate",
    "Poor", "Very Poor", "Severe"
]

clean = clean.filter(
    F.col("AQI_Bucket").isin(valid_categories)
)

# 6. Display cleaned results
print("Cleaned row count:", clean.count())

clean.groupBy("AQI_Bucket").count() \
    .orderBy("AQI_Bucket").show()

# 7. Save as Parquet
clean.write.mode("overwrite").parquet(output_path)

# 8. Verify the saved dataset
saved = spark.read.parquet(output_path)
print("Verified saved row count:", saved.count())

spark.stop()