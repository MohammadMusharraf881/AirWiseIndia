
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(r"C:\Users\Asus\Desktop\AirWiseIndia")
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
CHART_DIR = BASE_DIR / "outputs" / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)

# Read actual evaluation results
model_files = [
    "logistic_regression.json",
    "decision_tree.json",
    "random_forest.json",
]

records = []

for filename in model_files:
    file_path = METRICS_DIR / filename

    with open(file_path, "r", encoding="utf-8") as file:
        records.append(json.load(file))

results = pd.DataFrame(records)

# Display important comparison columns
columns = [
    "model",
    "accuracy",
    "weighted_f1",
    "macro_f1",
    "training_seconds",
]

results = results[columns]

print("\nMODEL COMPARISON")
print(results.round(4).to_string(index=False))

# Save comparison table
csv_path = METRICS_DIR / "model_comparison.csv"
results.to_csv(csv_path, index=False)

# Create accuracy chart
plt.figure(figsize=(10, 6))
plt.bar(results["model"], results["accuracy"])
plt.title("AQI Classification Model Accuracy")
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()

chart_path = CHART_DIR / "model_accuracy_comparison.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Create F1-score chart
plt.figure(figsize=(10, 6))
x = range(len(results))
width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    results["weighted_f1"],
    width=width,
    label="Weighted F1",
)

plt.bar(
    [i + width / 2 for i in x],
    results["macro_f1"],
    width=width,
    label="Macro F1",
)

plt.xticks(list(x), results["model"], rotation=15, ha="right")
plt.ylabel("F1 Score")
plt.title("Weighted F1 vs Macro F1")
plt.legend()
plt.tight_layout()

f1_chart_path = CHART_DIR / "model_f1_comparison.png"
plt.savefig(f1_chart_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\nSaved comparison CSV:", csv_path)
print("Saved accuracy chart:", chart_path)
print("Saved F1 chart:", f1_chart_path)
