import pandas as pd
import os

os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

data = {
    'Year': list(range(1990, 2023)) * 3,
    'Country': ['USA'] * 33 + ['India'] * 33 + ['China'] * 33,
    'Crop': ['Corn'] * 33 + ['Rice'] * 33 + ['Wheat'] * 33,
    'Yield_tonnes_per_hectare':
        [6.0 + i*0.1 + (i%3)*0.2 for i in range(33)] +
        [2.5 + i*0.05 + (i%4)*0.1 for i in range(33)] +
        [3.5 + i*0.08 + (i%5)*0.15 for i in range(33)],
    'Rainfall_mm':
        [800 + (i*7)%200 for i in range(33)] +
        [1200 + (i*11)%300 for i in range(33)] +
        [600 + (i*9)%150 for i in range(33)],
    'Temperature_C':
        [15 + (i*0.3)%5 for i in range(33)] +
        [28 + (i*0.2)%4 for i in range(33)] +
        [12 + (i*0.25)%6 for i in range(33)],
    'Fertilizer_kg_per_ha':
        [150 + (i*5)%100 for i in range(33)] +
        [100 + (i*4)%80 for i in range(33)] +
        [120 + (i*6)%90 for i in range(33)]
}

df = pd.DataFrame(data)
print("Dataset created! Shape:", df.shape)
print(df.head())

df.to_csv('data/crop_yields.csv', index=False)
print("\nSaved to data/crop_yields.csv")
print("Columns:", df.columns.tolist())