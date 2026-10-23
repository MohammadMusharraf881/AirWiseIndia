
from pathlib import Path
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Start Spark using the local filesystem
spark = (
    SparkSession.builder
    .appName("AirWiseIndia-DecisionTree")
    .master("local[*]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

BASE_DIR = Path(r"C:\Users\Asus\Desktop\AirWiseIndia")
DATA_PATH = (
    "file:///C:/Users/Asus/Desktop/"
    "AirWiseIndia/data/processed/aqi_clean.parquet"
)
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
MODEL_URI = (
    "file:///C:/Users/Asus/Desktop/"
    "AirWiseIndia/outputs/models/decision_tree"
)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load data and rename PM2.5
df = spark.read.parquet(DATA_PATH)
df = df.withColumnRenamed("PM2.5", "PM25")

features = [
    "PM25", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

# Normalize numeric features and missing values
for c in features:
    df = df.withColumn(c, F.col(c).cast("double"))
    df = df.withColumn(
        c,
        F.when(F.isnan(F.col(c)), None).otherwise(F.col(c))
    )

# Use the same seed for a comparable split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("Training records:", train_df.count())
print("Testing records:", test_df.count())

# 3. Build the pipeline
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
    outputCol="features"
)

tree = DecisionTreeClassifier(
    labelCol="label",
    featuresCol="features",
    maxDepth=10,
    maxBins=64,
    seed=42
)

pipeline = Pipeline(stages=[
    label_indexer,
    imputer,
    assembler,
    tree
])

# 4. Train
print("\nTraining Decision Tree...")
start = time.time()
model = pipeline.fit(train_df)
training_seconds = time.time() - start

print(f"Training time: {training_seconds:.2f} seconds")

# 5. Predict
predictions = model.transform(test_df).cache()

print("\nSample predictions:")
predictions.select(
    "AQI_Bucket", "label", "prediction"
).show(10, truncate=False)

# 6. Evaluate
def evaluate(metric_name):
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric_name
    )
    return float(evaluator.evaluate(predictions))

accuracy = evaluate("accuracy")
weighted_f1 = evaluate("f1")

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

    score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    class_f1_scores.append(score)

macro_f1 = sum(class_f1_scores) / len(class_f1_scores)

print("\nDECISION TREE EVALUATION")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Weighted F1:   {weighted_f1:.4f}")
print(f"Macro F1:      {macro_f1:.4f}")
print(f"Training time: {training_seconds:.2f} seconds")

# 7. Confusion matrix
print("\nConfusion matrix (rows=actual, columns=predicted):")
predictions.groupBy("label") \
    .pivot("prediction", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) \
    .count().na.fill(0).orderBy("label").show(truncate=False)

# 8. Save model and metrics
model.write().overwrite().save(MODEL_URI)

metrics = {
    "model": "Decision Tree",
    "training_records": train_df.count(),
    "testing_records": test_df.count(),
    "accuracy": accuracy,
    "weighted_f1": weighted_f1,
    "macro_f1": macro_f1,
    "training_seconds": training_seconds
}

metrics_path = METRICS_DIR / "decision_tree.json"
with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved to:", MODEL_URI)
print("Metrics saved to:", metrics_path)

predictions.unpersist()
spark.stop()
