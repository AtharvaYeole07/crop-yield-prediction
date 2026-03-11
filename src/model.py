import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

# Load data
df = pd.read_csv('data/crop_yields.csv')

# Encode categorical columns
df['Country_code'] = df['Country'].astype('category').cat.codes
df['Crop_code'] = df['Crop'].astype('category').cat.codes

# Features and target
X = df[['Year', 'Rainfall_mm', 'Temperature_C', 'Country_code', 'Crop_code']]
y = df['Yield_tonnes_per_hectare']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train 3 models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {'R2': round(r2, 3), 'RMSE': round(rmse, 3)}
    print(f"{name}: R2={r2:.3f}, RMSE={rmse:.3f}")

# Plot model comparison
plt.figure(figsize=(8, 5))
names = list(results.keys())
r2_scores = [results[m]['R2'] for m in names]
plt.bar(names, r2_scores, color=['#2E4A7A', '#1A6B3C', '#8B0000'])
plt.title('Model Comparison - R2 Score')
plt.ylabel('R2 Score (higher is better)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('outputs/model_comparison.png')
print("\nModel comparison plot saved!")
print("\nBest model:", max(results, key=lambda x: results[x]['R2']))