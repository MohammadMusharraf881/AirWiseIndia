
from pathlib import Path
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("AirWiseIndia-RandomForest")
    .master("local[*]")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

BASE = Path(r"C:\Users\Asus\Desktop\AirWiseIndia")
DATA = "file:///C:/Users/Asus/Desktop/AirWiseIndia/data/processed/aqi_clean.parquet"
MODEL_URI = "file:///C:/Users/Asus/Desktop/AirWiseIndia/outputs/models/random_forest"

metrics_dir = BASE / "outputs" / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)

df = spark.read.parquet(DATA)
df = df.withColumnRenamed("PM2.5", "PM25")

features = [
    "PM25", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

for c in features:
    df = df.withColumn(c, F.col(c).cast("double"))
    df = df.withColumn(
        c, F.when(F.isnan(F.col(c)), None).otherwise(F.col(c))
    )

train, test = df.randomSplit([0.8, 0.2], seed=42)

print("Training records:", train.count())
print("Testing records:", test.count())

indexer = StringIndexer(
    inputCol="AQI_Bucket",
    outputCol="label",
    stringOrderType="alphabetAsc"
)

imputed = [c + "_imputed" for c in features]

imputer = Imputer(
    inputCols=features,
    outputCols=imputed,
    strategy="median"
)

assembler = VectorAssembler(
    inputCols=imputed,
    outputCol="features"
)

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=100,
    maxDepth=10,
    maxBins=64,
    seed=42
)

pipeline = Pipeline(stages=[indexer, imputer, assembler, rf])

print("\nTraining Random Forest...")
start = time.time()
model = pipeline.fit(train)
elapsed = time.time() - start

pred = model.transform(test).cache()

def metric(name):
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=name
    )
    return float(evaluator.evaluate(pred))

accuracy = metric("accuracy")
weighted_f1 = metric("f1")

# Calculate macro F1 as the mean of per-class F1 scores.
class_f1 = []
for row in pred.select("label").distinct().collect():
    label = float(row["label"])
    tp = pred.filter(
        (F.col("label") == label) &
        (F.col("prediction") == label)
    ).count()
    fp = pred.filter(
        (F.col("label") != label) &
        (F.col("prediction") == label)
    ).count()
    fn = pred.filter(
        (F.col("label") == label) &
        (F.col("prediction") != label)
    ).count()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    class_f1.append(score)

macro_f1 = sum(class_f1) / len(class_f1)

print("\nRANDOM FOREST EVALUATION")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Weighted F1:   {weighted_f1:.4f}")
print(f"Macro F1:      {macro_f1:.4f}")
print(f"Training time: {elapsed:.2f} seconds")

print("\nConfusion matrix (rows=actual, columns=predicted):")
pred.groupBy("label") \
    .pivot("prediction", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) \
    .count().na.fill(0).orderBy("label").show(truncate=False)

model.write().overwrite().save(MODEL_URI)

results = {
    "model": "Random Forest",
    "training_records": train.count(),
    "testing_records": test.count(),
    "accuracy": accuracy,
    "weighted_f1": weighted_f1,
    "macro_f1": macro_f1,
    "training_seconds": elapsed
}

with open(metrics_dir / "random_forest.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nModel saved:", MODEL_URI)
print("Metrics saved:", metrics_dir / "random_forest.json")

pred.unpersist()
spark.stop()
