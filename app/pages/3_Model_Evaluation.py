import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys


sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")

st.title("📊 Model Evaluation")
st.markdown("Performance metrics and analysis")
st.markdown("---")


def load_evaluation_data():
    
    models_dir = Path(__file__).parent.parent.parent / 'models'
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    
    try:
        model = joblib.load(models_dir / 'xgb.pkl')
        with open(models_dir / 'metrics.json', 'r') as f:
            metrics = json.load(f)
        with open(models_dir / 'features.json', 'r') as f:
            features = json.load(f)
        
        df = pd.read_csv(data_dir / 'land_prices_clean.csv')
        
        return model, metrics, features, df
    except FileNotFoundError:
        return None, None, None, None

model, metrics, features_meta, df = load_evaluation_data()

if model is not None and df is not None:
    st.subheader("🎯 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RMSE", f"₹{metrics['test']['rmse']:.2f}")
    
    with col2:
        st.metric("R² Score", f"{metrics['test']['r2_score']:.4f}")
    
    with col3:
        st.metric("MAE", f"₹{metrics['test']['mae']:.2f}")
    
    with col4:
        accuracy = metrics['test']['r2_score'] * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    st.markdown("---")
    
    st.subheader("📈 Prediction vs Actual Values")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        feature_cols = features_meta['features']
        X = df[feature_cols]
        y = df['price_per_sqft']
        y_pred = model.predict(X)
        
        
        sample_size = min(200, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        
        y_actual = y.iloc[sample_indices]
        y_predicted = y_pred[sample_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        
        ax.scatter(y_actual, y_predicted, alpha=0.5, s=50)
        
        
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price per sq ft (₹)', fontsize=12)
        ax.set_ylabel('Predicted Price per sq ft (₹)', fontsize=12)
        ax.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("📋 Metrics Explanation")
        
        st.markdown("""
        **RMSE** (Root Mean Squared Error)
        - Lower is better
        - Measures average prediction error
        
        **R² Score**
        - Range: 0 to 1
        - Higher is better
        - Explains variance in data
        
        **MAE** (Mean Absolute Error)
        - Average absolute difference
        - Easy to interpret
        
        **Accuracy**
        - R² score as percentage
        - Overall model performance
        """)
    
    st.markdown("---")
    
    
    st.subheader("🔍 Feature Importance")
    
    try:
        
        feature_importance = model.named_steps['model'].feature_importances_
        feature_cols = features_meta['features']
        
        
        feature_name_map = {
            'area_sqft': 'Area (sq ft)',
            'proximity_to_city_km': 'Distance from City',
            'year': 'Year',
            'month': 'Month',
            'city_encoded': 'City',
            'location_type_encoded': 'Location Type',
            'school_distance_km': 'School Distance',
            'hospital_distance_km': 'Hospital Distance',
            'metro_distance_km': 'Metro Distance',
            'airport_distance_km': 'Airport Distance',
            'mall_distance_km': 'Mall Distance',
            'railway_distance_km': 'Railway Distance',
            'road_access_encoded': 'Road Access',
            'water_supply_encoded': 'Water Supply',
            'electricity_encoded': 'Electricity',
            'sewage_system_encoded': 'Sewage System',
            'road_width_ft': 'Road Width',
            'corner_plot_encoded': 'Corner Plot',
            'avg_facility_distance': 'Avg Facility Distance',
            'infrastructure_score': 'Infrastructure Score',
            'premium_location': 'Premium Location'
        }
        
        readable_names = [feature_name_map.get(f, f) for f in feature_cols]
        
        importance_df = pd.DataFrame({
            'Feature': readable_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(15)  
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='#1f77b4')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis() 
        
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Could not display feature importance: {e}")

else:
    st.info("👈 Please train the model first by running `python main.py`")

st.markdown("---")
st.caption("📊 Model Evaluation | XGBoost Performance Analysis")
