
from pathlib import Path
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator
)

# --------------------------------------------------
# 1. Start Spark
# --------------------------------------------------
spark = (
    SparkSession.builder
    .appName("AirWiseIndia-LogisticRegression")
    .master("local[*]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

BASE_DIR = Path(r"C:\Users\Asus\Desktop\AirWiseIndia")
DATA_PATH = (
    "file:///C:/Users/Asus/Desktop/"
    "AirWiseIndia/data/processed/aqi_clean.parquet"
)

MODEL_DIR = BASE_DIR / "outputs" / "models" / "logistic_regression"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load and prepare data
# --------------------------------------------------
df = spark.read.parquet(DATA_PATH)
df = df.withColumnRenamed("PM2.5", "PM25")

features = [
    "PM25", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

for c in features:
    df = df.withColumn(c, F.col(c).cast("double"))
    df = df.withColumn(
        c,
        F.when(F.isnan(F.col(c)), None).otherwise(F.col(c))
    )

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("Training records:", train_df.count())
print("Testing records:", test_df.count())

# --------------------------------------------------
# 3. Build preprocessing and classification pipeline
# --------------------------------------------------
label_indexer = StringIndexer(
    inputCol="AQI_Bucket",
    outputCol="label",
    stringOrderType="alphabetAsc"
)

imputed_features = [c + "_imputed" for c in features]

imputer = Imputer(
    inputCols=features,
    outputCols=imputed_features,
    strategy="median"
)

assembler = VectorAssembler(
    inputCols=imputed_features,
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=False,
    withStd=True
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

pipeline = Pipeline(stages=[
    label_indexer,
    imputer,
    assembler,
    scaler,
    lr
])

# --------------------------------------------------
# 4. Train model
# --------------------------------------------------
print("\nTraining Logistic Regression...")
start = time.time()

model = pipeline.fit(train_df)

training_seconds = time.time() - start
print(f"Training time: {training_seconds:.2f} seconds")

# --------------------------------------------------
# 5. Predict test data
# --------------------------------------------------
predictions = model.transform(test_df).cache()
print("\nSample predictions:")
predictions.select(
    "AQI_Bucket", "label", "prediction", "probability"
).show(10, truncate=False)

# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------
def evaluate(metric_name):
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric_name
    )
    return float(evaluator.evaluate(predictions))

accuracy = evaluate("accuracy")
f1_weighted = evaluate("f1")
f1_macro = evaluate("f1")  # Replaced below with per-class macro F1

# Compute macro F1 explicitly by averaging class F1 scores.
class_f1_scores = []

for row in predictions.select("label").distinct().collect():
    label = float(row["label"])
    tp = predictions.filter(
        (F.col("label") == label) &
        (F.col("prediction") == label)
    ).count()
    fp = predictions.filter(
        (F.col("label") != label) &
        (F.col("prediction") == label)
    ).count()
    fn = predictions.filter(
        (F.col("label") == label) &
        (F.col("prediction") != label)
    ).count()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    class_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    class_f1_scores.append(class_f1)

f1_macro = sum(class_f1_scores) / len(class_f1_scores)

print("\nMODEL EVALUATION")
print(f"Accuracy:       {accuracy:.4f}")
print(f"Weighted F1:    {f1_weighted:.4f}")
print(f"Macro F1:       {f1_macro:.4f}")
print(f"Training time:  {training_seconds:.2f} seconds")

# --------------------------------------------------
# 7. Confusion matrix
# --------------------------------------------------
print("\nConfusion matrix (rows=actual, columns=predicted):")

confusion = (
    predictions
    .groupBy("label")
    .pivot("prediction", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    .count()
    .na.fill(0)
    .orderBy("label")
)

confusion.show(truncate=False)

# --------------------------------------------------
# 8. Save model and metrics
# --------------------------------------------------
model.write().overwrite().save(
    "file:///C:/Users/Asus/Desktop/AirWiseIndia/outputs/models/logistic_regression"
)

metrics = {
    "model": "Logistic Regression",
    "training_records": train_df.count(),
    "testing_records": test_df.count(),
    "accuracy": accuracy,
    "weighted_f1": f1_weighted,
    "macro_f1": f1_macro,
    "training_seconds": training_seconds
}

metrics_path = METRICS_DIR / "logistic_regression.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved to:", MODEL_DIR)
print("Metrics saved to:", metrics_path)

predictions.unpersist()
spark.stop()
