
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    VectorAssembler,
    StandardScaler
)

# --------------------------------------------------
# 1. Start Spark
# --------------------------------------------------
spark = (
    SparkSession.builder
    .appName("AirWiseIndia-ML-Pipeline-Preparation")
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
df = df.withColumnRenamed("PM2.5", "PM25")

feature_columns = [
    "PM25", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

target_column = "AQI_Bucket"

# --------------------------------------------------
# 3. Convert features to numeric and NaN to null
# --------------------------------------------------
for column in feature_columns:
    df = df.withColumn(
        column,
        F.col(column).cast("double")
    )

    df = df.withColumn(
        column,
        F.when(
            F.isnan(F.col(column)), None
        ).otherwise(F.col(column))
    )

# --------------------------------------------------
# 4. Split data before fitting preprocessing
# --------------------------------------------------
train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("Training records:", train_df.count())
print("Testing records:", test_df.count())

# --------------------------------------------------
# 5. Encode target labels
# --------------------------------------------------
label_indexer = StringIndexer(
    inputCol=target_column,
    outputCol="label",
    stringOrderType="alphabetAsc"
)

# --------------------------------------------------
# 6. Impute missing numeric features
# Median values are learned from training data
# --------------------------------------------------
imputed_columns = [
    column + "_imputed"
    for column in feature_columns
]

imputer = Imputer(
    inputCols=feature_columns,
    outputCols=imputed_columns,
    strategy="median"
)

# --------------------------------------------------
# 7. Assemble feature vector
# --------------------------------------------------
assembler = VectorAssembler(
    inputCols=imputed_columns,
    outputCol="raw_features"
)

# --------------------------------------------------
# 8. Scale features
# --------------------------------------------------
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=False,
    withStd=True
)

pipeline = Pipeline(
    stages=[
        label_indexer,
        imputer,
        assembler,
        scaler
    ]
)

# Fit preprocessing using training data only
preprocessor_model = pipeline.fit(train_df)

train_prepared = preprocessor_model.transform(train_df)
test_prepared = preprocessor_model.transform(test_df)

print("\nPrepared training records:", train_prepared.count())
print("Prepared testing records:", test_prepared.count())

print("\nPrepared feature and label example:")
train_prepared.select(
    "features", "label", target_column
).show(5, truncate=False)

print("\nPreprocessing pipeline completed successfully.")

spark.stop()
