# 🌾 Crop Yield Prediction

**MSc Bioinformatics Portfolio Project | University of Birmingham**

A machine learning project that predicts crop yields using environmental
and climate data. Built with Python, scikit-learn, XGBoost, and Streamlit.

---

## 📊 Results

| Model | R2 Score | Performance |
|---|---|---|
| Linear Regression | 0.96 | Very Good |
| Random Forest | 0.99 | Excellent |
| XGBoost | 0.98 | Excellent |

**Best Model: Random Forest (R2 = 0.99)**

---

## 🚀 How to Run

### Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit

### Run in order
python src/data_download.py
python src/visualise.py
python src/model.py
streamlit run src/dashboard.py

Then open http://localhost:8501 in your browser.

---

## 📁 Project Structure

crop-yield-prediction/
├── data/               # Dataset (auto-created)
├── outputs/            # Plots and results (auto-created)
├── src/
│   ├── data_download.py   # Step 1: Create dataset
│   ├── visualise.py       # Step 2: Generate 5 charts
│   ├── model.py           # Step 3: Train ML models
│   └── dashboard.py       # Step 4: Interactive dashboard
└── README.md

---

## 🔬 Features

- Crop yield data for USA, India and China (1990-2022)
- Environmental features: rainfall, temperature, fertilizer usage
- Three ML models compared: Linear Regression, Random Forest, XGBoost
- Feature importance analysis showing key yield drivers
- Interactive Streamlit dashboard with live yield prediction

---

## 🛠️ Technologies

- Python, pandas, numpy
- scikit-learn, XGBoost
- matplotlib, seaborn
- Streamlit

---

## 👤 Author

Atharva Yeole | MSc Bioinformatics, University of Birmingham
GitHub: https://github.com/AtharvaYeole07