
from pathlib import Path
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = (
    SparkSession.builder
    .appName("AirWiseIndia-LinearRegression")
    .master("local[*]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

BASE = Path(r"C:\Users\Asus\Desktop\AirWiseIndia")
DATA = (
    "file:///C:/Users/Asus/Desktop/"
    "AirWiseIndia/data/processed/aqi_clean.parquet"
)
MODEL_URI = (
    "file:///C:/Users/Asus/Desktop/"
    "AirWiseIndia/outputs/models/linear_regression"
)
METRICS_DIR = BASE / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = spark.read.parquet(DATA)
df = df.withColumnRenamed("PM2.5", "PM25")

features = [
    "PM25", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

# Convert features and AQI to numeric; convert NaN to null
for c in features + ["AQI"]:
    df = df.withColumn(c, F.col(c).cast("double"))
    df = df.withColumn(
        c,
        F.when(F.isnan(F.col(c)), None).otherwise(F.col(c))
    )

# AQI is the regression target, not a feature
df = df.filter(F.col("AQI").isNotNull())

train, test = df.randomSplit([0.8, 0.2], seed=42)

print("Training records:", train.count())
print("Testing records:", test.count())

imputed_features = [c + "_imputed" for c in features]

imputer = Imputer(
    inputCols=features,
    outputCols=imputed_features,
    strategy="median"
)

assembler = VectorAssembler(
    inputCols=imputed_features,
    outputCol="features"
)

lr = LinearRegression(
    featuresCol="features",
    labelCol="AQI",
    predictionCol="prediction",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

pipeline = Pipeline(stages=[imputer, assembler, lr])

print("\nTraining Linear Regression...")
start = time.time()
model = pipeline.fit(train)
elapsed = time.time() - start

predictions = model.transform(test).cache()

def evaluate(metric_name):
    evaluator = RegressionEvaluator(
        labelCol="AQI",
        predictionCol="prediction",
        metricName=metric_name
    )
    return float(evaluator.evaluate(predictions))

rmse = evaluate("rmse")
mae = evaluate("mae")
r2 = evaluate("r2")

print("\nLINEAR REGRESSION EVALUATION")
print(f"RMSE:          {rmse:.4f}")
print(f"MAE:           {mae:.4f}")
print(f"R-squared:     {r2:.4f}")
print(f"Training time: {elapsed:.2f} seconds")

print("\nSample predictions:")
predictions.select("AQI", "prediction").show(10, truncate=False)

# Save model
model.write().overwrite().save(MODEL_URI)

metrics = {
    "model": "Linear Regression",
    "training_records": train.count(),
    "testing_records": test.count(),
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "training_seconds": elapsed
}

metrics_path = METRICS_DIR / "linear_regression.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved:", MODEL_URI)
print("Metrics saved:", metrics_path)

predictions.unpersist()
spark.stop()
