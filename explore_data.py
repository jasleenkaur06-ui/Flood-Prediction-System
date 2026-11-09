
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("train.csv")

# 2. First 5 Rows
print("\n--- First 5 Rows ---")
print(df.head())

# 3. Dataset Shape
print("\n--- Dataset Shape ---")
print(df.shape)

# 4. Column Names
print("\n--- Column Names ---")
print(df.columns.tolist())

# 5. Dataset Information
print("\n--- Dataset Information ---")
df.info()

# 6. Missing Values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 7. Statistical Summary
print("\n--- Statistical Summary ---")
print(df.describe())

# 8. Flood Probability Analysis
print("\n--- Flood Probability Analysis ---")
print("Minimum:", df["FloodProbability"].min())
print("Maximum:", df["FloodProbability"].max())
print("Mean:", df["FloodProbability"].mean())
print("Median:", df["FloodProbability"].median())

# 9. Duplicate Records
print("\n--- Duplicate Records ---")
print(df.duplicated().sum())

# 10. Duplicate IDs
print("\n--- Duplicate IDs ---")
print(df["id"].duplicated().sum())


# 11. Correlation Analysis
# Remove ID because it is only an identifier
correlation = df.drop(columns=["id"]).corr()

# 12. Correlation with FloodProbability
flood_corr = correlation["FloodProbability"].drop("FloodProbability")

flood_corr = flood_corr.sort_values(ascending=False)

print("\n--- Correlation with FloodProbability ---")
print(flood_corr)


# 13. Flood Probability Distribution
plt.figure(figsize=(10, 5))

plt.hist(df["FloodProbability"], bins=30, edgecolor="black")

plt.title("Distribution of Flood Probability")
plt.xlabel("Flood Probability")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# 14. Correlation Heatmap
plt.figure(figsize=(16, 12))

plt.imshow(
    correlation,
    cmap="coolwarm",
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap of Flood Prediction Dataset")

plt.tight_layout()
plt.show()
print("\n--- Feature Minimum and Maximum Values ---")

print(df.drop(columns=["id", "FloodProbability"]).agg(["min", "max"]).T)
print("\n--- Feature Minimum and Maximum Values ---")

feature_ranges = df.drop(
    columns=["id", "FloodProbability"]
).agg(["min", "max"]).T

print(feature_ranges)