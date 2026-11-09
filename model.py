
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("train.csv")

# Separate features and target
X = df.drop(columns=["id", "FloodProbability"])
y = df["FloodProbability"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use a training sample to reduce laptop processing time
X_sample = X_train.sample(n=200000, random_state=42)
y_sample = y_train.loc[X_sample.index]

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train model
print("Training Random Forest model...")
model.fit(X_sample, y_sample)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n--- Random Forest Evaluation ---")

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R2 Score:", r2_score(y_test, y_pred))

# Save separately for comparison
joblib.dump(model, "random_forest_model.pkl")

print("\nRandom Forest model saved successfully!")
