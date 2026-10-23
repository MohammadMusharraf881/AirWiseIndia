
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
BASE_DIR = Path(r"C:\Users\Asus\Desktop\AirWiseIndia")
DATA_PATH = BASE_DIR / "data" / "processed" / "aqi_clean.parquet"
CHART_DIR = BASE_DIR / "outputs" / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# 2. Load the cleaned dataset
# --------------------------------------------------
df = pd.read_parquet(DATA_PATH)

print("AirWise India - Data Visualization")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. Chart 1: AQI category distribution
# --------------------------------------------------
category_counts = df["AQI_Bucket"].value_counts()

category_order = [
    "Good", "Satisfactory", "Moderate",
    "Poor", "Very Poor", "Severe"
]

category_counts = category_counts.reindex(
    category_order, fill_value=0
)

plt.figure(figsize=(10, 6))
category_counts.plot(kind="bar")
plt.title("Distribution of AQI Categories in India")
plt.xlabel("AQI Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "aqi_category_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# --------------------------------------------------
# 4. Chart 2: Average pollutant levels
# --------------------------------------------------
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

available_pollutants = [
    col for col in pollutants if col in df.columns
]

pollutant_means = df[available_pollutants].mean(
    numeric_only=True
)

plt.figure(figsize=(10, 6))
pollutant_means.plot(kind="bar")
plt.title("Average Pollutant Levels")
plt.xlabel("Pollutant")
plt.ylabel("Average Recorded Value")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "average_pollutant_levels.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# --------------------------------------------------
# 5. Chart 3: Correlation heatmap
# --------------------------------------------------
numeric_columns = [
    "PM2.5", "PM10", "NO2", "SO2",
    "CO", "O3", "AQI"
]

available_numeric = [
    col for col in numeric_columns if col in df.columns
]

correlation = df[available_numeric].corr(
    numeric_only=True
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Heatmap: Pollutants and AQI")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "pollutant_aqi_correlation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

print("\nAll three charts saved to:")
print(CHART_DIR)
print("\nVisualization stage completed.")
