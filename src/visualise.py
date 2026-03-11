import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('outputs', exist_ok=True)

# Load data
df = pd.read_csv('data/crop_yields.csv')
print("Data loaded! Shape:", df.shape)

# Plot 1 - Yield over time by crop
plt.figure(figsize=(12, 5))
for crop in df['Crop'].unique():
    subset = df[df['Crop'] == crop]
    plt.plot(subset['Year'], subset['Yield_tonnes_per_hectare'], label=crop, marker='o', markersize=3)
plt.title('Crop Yield Over Time by Crop Type')
plt.xlabel('Year')
plt.ylabel('Yield (tonnes/hectare)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/yield_over_time.png')
print("Plot 1 saved!")

# Plot 2 - Yield vs Rainfall
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Rainfall_mm', y='Yield_tonnes_per_hectare', hue='Crop', s=80)
plt.title('Yield vs Rainfall')
plt.tight_layout()
plt.savefig('outputs/yield_vs_rainfall.png')
print("Plot 2 saved!")

# Plot 3 - Average yield by country
plt.figure(figsize=(8, 5))
avg = df.groupby('Country')['Yield_tonnes_per_hectare'].mean().reset_index()
sns.barplot(data=avg, x='Country', y='Yield_tonnes_per_hectare', palette='viridis')
plt.title('Average Yield by Country')
plt.tight_layout()
plt.savefig('outputs/yield_by_country.png')
print("Plot 3 saved!")

print("\nAll plots saved to outputs/ folder!")