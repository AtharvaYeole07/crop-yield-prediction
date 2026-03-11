import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")
st.title("🌾 Crop Yield Prediction Dashboard")
st.markdown("**MSc Bioinformatics Portfolio Project | University of Birmingham**")
st.markdown("---")

@st.cache_data
def load_data():
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/crop_yields.csv'
    
    # Create data if file doesn't exist
    if not os.path.exists(csv_path):
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
        df.to_csv(csv_path, index=False)
    
    return pd.read_csv(csv_path)

@st.cache_resource
def train_model(df):
    df = df.copy()
    df['Country_code'] = df['Country'].astype('category').cat.codes
    df['Crop_code'] = df['Crop'].astype('category').cat.codes
    X = df[['Year', 'Rainfall_mm', 'Temperature_C', 'Fertilizer_kg_per_ha', 'Country_code', 'Crop_code']]
    y = df['Yield_tonnes_per_hectare']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

df = load_data()
model, r2 = train_model(df)

st.sidebar.header("🔧 Filters & Prediction")
selected_country = st.sidebar.selectbox("Select Country", df['Country'].unique())
selected_crop = st.sidebar.selectbox("Select Crop", df['Crop'].unique())
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Predict Yield")
pred_year        = st.sidebar.slider("Year", 1990, 2035, 2025)
pred_rainfall    = st.sidebar.slider("Rainfall (mm)", 400, 1600, 900)
pred_temperature = st.sidebar.slider("Temperature (°C)", 5, 40, 20)
pred_fertilizer  = st.sidebar.slider("Fertilizer (kg/ha)", 50, 300, 150)

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Total Records", df.shape[0])
with col2: st.metric("Countries", df['Country'].nunique())
with col3: st.metric("Crops", df['Crop'].nunique())
with col4: st.metric("Model R² Score", f"{r2:.3f}")

st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Yield Over Time")
    filtered = df[(df['Country'] == selected_country) & (df['Crop'] == selected_crop)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(filtered['Year'], filtered['Yield_tonnes_per_hectare'], color='#2E4A7A', linewidth=2.5, marker='o', markersize=5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Yield (tonnes/hectare)')
    ax.set_title(f'{selected_crop} Yield in {selected_country}')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("🌧️ Yield vs Rainfall")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    palette = {'Corn': '#2E4A7A', 'Rice': '#1A6B3C', 'Wheat': '#8B6914'}
    for crop in df['Crop'].unique():
        sub = df[df['Crop'] == crop]
        ax2.scatter(sub['Rainfall_mm'], sub['Yield_tonnes_per_hectare'], label=crop, color=palette[crop], alpha=0.6, s=50)
    ax2.set_xlabel('Rainfall (mm)')
    ax2.set_ylabel('Yield (tonnes/hectare)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()

st.markdown("---")
st.subheader("🤖 Live Yield Prediction")

country_map = {c: i for i, c in enumerate(sorted(df['Country'].unique()))}
crop_map    = {c: i for i, c in enumerate(sorted(df['Crop'].unique()))}

pred_input = np.array([[pred_year, pred_rainfall, pred_temperature, pred_fertilizer,
                         country_map[selected_country], crop_map[selected_crop]]])
prediction = model.predict(pred_input)[0]

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.info(f"**Country:** {selected_country}")
    st.info(f"**Crop:** {selected_crop}")
with col_p2:
    st.info(f"**Year:** {pred_year}")
    st.info(f"**Rainfall:** {pred_rainfall} mm")
with col_p3:
    st.success(f"### 🌾 Predicted Yield")
    st.success(f"## {prediction:.2f} tonnes/hectare")

st.markdown("---")
st.caption("Built with Python, scikit-learn & Streamlit | MSc Bioinformatics — University of Birmingham")