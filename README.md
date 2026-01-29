# Land Price Prediction

A machine learning application for predicting land prices using XGBoost and ARIMA forecasting.

## Project Structure

```text
Land-Price-Prediction/
│
├── app/                           # Streamlit application
│   ├── streamlit_app.py           # Main prediction UI (Home)
│   │
│   ├── pages/                     # Multi-page Streamlit app
│   │   ├── 2_Future_Price_Forecast.py
│   │   ├── 3_Model_Evaluation.py
│   │   └── 4_Prediction_History.py
│
├── data/                          # Datasets
│   ├── raw/
│   │   └── land_prices_raw.csv
│   │
│   └── processed/
│       └── land_prices_clean.csv
│
├── models/                        # Saved models & metadata
│   ├── xgb.pkl                    # Scaled ML pipeline
│   ├── arima_model.pkl            # Forecasting model
│   ├── features.json              # Feature order metadata
│   └── metrics.json               # RMSE & R² scores
│
├── src/                           # Backend / ML scripts
│   ├── data_generator.py          # Dummy data generation
│   ├── preprocessing.py           # Data cleaning & encoding
│   ├── train_model.py             # Training + scaling + metrics
│   ├── forecast.py                # ARIMA forecasting
│   └── db.py                      # PostgreSQL connection helper
│
├── notebooks/                     # Optional (local only)
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_forecasting.ipynb
│
├── requirements.txt               # Streamlit Cloud dependencies
├── README.md                      # Project documentation
├── .gitignore
└── main.py                        # Optional pipeline runner
```

## Installation

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Features

- **Land Price Prediction**: ML-based price prediction using XGBoost
- **Future Price Forecast**: Time series forecasting with ARIMA
- **Model Evaluation**: View model performance metrics
- **Prediction History**: Track and review past predictions

## Technologies

- Python 3.x
- Streamlit
- XGBoost
- Scikit-learn
- Pandas & NumPy
- Matplotlib
- Statsmodels (ARIMA)
- PostgreSQL
