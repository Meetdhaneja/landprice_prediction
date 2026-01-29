import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


st.set_page_config(
    page_title="Land Price Prediction - Indian Cities",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 Land Price Prediction System")
st.markdown("### Predict land prices across major Indian cities")
st.markdown("---")

@st.cache_resource
def load_model():
    models_dir = Path(__file__).parent.parent / 'models'
    
    try:
        model = joblib.load(models_dir / 'xgb.pkl')
        with open(models_dir / 'features.json', 'r') as f:
            features = json.load(f)
        with open(models_dir / 'metrics.json', 'r') as f:
            metrics = json.load(f)
        return model, features, metrics
    except FileNotFoundError:
        st.error("⚠️ Model not found! Please run the training pipeline first.")
        st.code("python main.py", language="bash")
        return None, None, None


@st.cache_data
def load_city_data():

    try:
        data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'land_prices_clean.csv'
        df = pd.read_csv(data_path)
        
        city_mapping = df[['city', 'city_encoded']].drop_duplicates().set_index('city')['city_encoded'].to_dict()
        location_mapping = df[['location_type', 'location_type_encoded']].drop_duplicates().set_index('location_type')['location_type_encoded'].to_dict()
        
        return city_mapping, location_mapping, df
    except:
        return None, None, None

model, features_meta, metrics = load_model()
city_mapping, location_mapping, data_df = load_city_data()

if model is not None and metrics is not None:
    
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model", "XGBoost")
    with col2:
        st.metric("Test RMSE", f"₹{metrics['test']['rmse']:.2f}")
    with col3:
        st.metric("Test R² Score", f"{metrics['test']['r2_score']:.4f}")
    with col4:
        accuracy = metrics['test']['r2_score'] * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    st.markdown("---")
    
    
    st.subheader("📝 Enter Land Details")
    
    
    cities = ['Mumbai', 'Pune', 'Nagpur', 'Ahmedabad', 'Rajkot', 'Thane', 'Nashik', 'Surat']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city = st.selectbox("🏙️ City", options=cities)
        
        location_type = st.selectbox(
            "📍 Location Type",
            options=["Urban", "Suburban", "Rural"]
        )
        
        area_sqft = st.number_input(
            "📏 Area (sq ft)",
            min_value=100,
            max_value=50000,
            value=2000,
            step=100
        )
        
        proximity_to_city_km = st.slider(
            "🚗 Distance from City Center (km)",
            min_value=1,
            max_value=100,
            value=10
        )
    
    with col2:
        st.markdown("**🏫 Nearby Facilities (Distance in km)**")
        
        school_distance_km = st.number_input(
            "School Distance (km)",
            min_value=0.5,
            max_value=20.0,
            value=2.0,
            step=0.5
        )
        
        hospital_distance_km = st.number_input(
            "Hospital Distance (km)",
            min_value=0.5,
            max_value=25.0,
            value=3.0,
            step=0.5
        )
        
        metro_distance_km = st.number_input(
            "Metro Station Distance (km)",
            min_value=0.5,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help="Set to 999 if no metro in city"
        )
        
        airport_distance_km = st.number_input(
            "Airport Distance (km)",
            min_value=5.0,
            max_value=60.0,
            value=15.0,
            step=1.0
        )
        
        mall_distance_km = st.number_input(
            "Shopping Mall Distance (km)",
            min_value=0.5,
            max_value=25.0,
            value=3.0,
            step=0.5
        )
        
        railway_distance_km = st.number_input(
            "Railway Station Distance (km)",
            min_value=0.5,
            max_value=20.0,
            value=2.0,
            step=0.5
        )
    
    with col3:
        st.markdown("**🏗️ Infrastructure**")
        
        road_access = st.radio(
            "Road Access",
            options=["Yes", "No"],
            horizontal=True
        )
        
        water_supply = st.radio(
            "Water Supply",
            options=["Yes", "No"],
            horizontal=True
        )
        
        electricity = st.radio(
            "Electricity",
            options=["Yes", "No"],
            horizontal=True
        )
        
        sewage_system = st.radio(
            "Sewage System",
            options=["Yes", "No"],
            horizontal=True
        )
        
        st.markdown("**🛣️ Property Details**")
        
        road_width_ft = st.selectbox(
            "Road Width (ft)",
            options=[20, 30, 40, 60, 80]
        )
        
        corner_plot = st.radio(
            "Corner Plot",
            options=["Yes", "No"],
            horizontal=True
        )
        
        year = st.number_input(
            "Year",
            min_value=2015,
            max_value=2030,
            value=2025
        )
        
        month = st.slider(
            "Month",
            min_value=1,
            max_value=12,
            value=1
        )
    
    st.markdown("---")
    
    
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        # Encode categorical variables
        city_encoded = city_mapping.get(city, 0)
        location_type_encoded = location_mapping.get(location_type, 0)
        road_access_encoded = 1 if road_access == "Yes" else 0
        water_supply_encoded = 1 if water_supply == "Yes" else 0
        electricity_encoded = 1 if electricity == "Yes" else 0
        sewage_system_encoded = 1 if sewage_system == "Yes" else 0
        corner_plot_encoded = 1 if corner_plot == "Yes" else 0
        
        # Calculate engineered features
        avg_facility_distance = (school_distance_km + hospital_distance_km + 
                                mall_distance_km + railway_distance_km) / 4
        
        infrastructure_score = (road_access_encoded + water_supply_encoded + 
                                electricity_encoded + sewage_system_encoded)
        
        premium_location = 1 if (metro_distance_km < 5 and school_distance_km < 2 and 
                                hospital_distance_km < 3) else 0
        
        input_features = np.array([[
            area_sqft,
            proximity_to_city_km,
            year,
            month,
            city_encoded,
            location_type_encoded,
            school_distance_km,
            hospital_distance_km,
            metro_distance_km,
            airport_distance_km,
            mall_distance_km,
            railway_distance_km,
            road_access_encoded,
            water_supply_encoded,
            electricity_encoded,
            sewage_system_encoded,
            road_width_ft,
            corner_plot_encoded,
            avg_facility_distance,
            infrastructure_score,
            premium_location
        ]])
        
        
        prediction = model.predict(input_features)[0]
        total_price = prediction * area_sqft
        
        
        st.success("✅ Prediction Complete!")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                "Predicted Price per sq ft",
                f"₹{prediction:,.2f}",
                delta=None
            )
        
        with result_col2:
            st.metric(
                "Total Estimated Price",
                f"₹{total_price:,.2f}",
                delta=None
            )
        
        with result_col3:
            
            price_in_lakhs = total_price / 100000
            st.metric(
                "Total Price (Lakhs)",
                f"₹{price_in_lakhs:,.2f}L",
                delta=None
            )
        
        
        with st.expander("📊 Input Summary"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                summary_df1 = pd.DataFrame({
                    'Feature': [
                        'City', 'Location Type', 'Area', 'Distance from City',
                        'Year', 'Month'
                    ],
                    'Value': [
                        city,
                        location_type,
                        f"{area_sqft:,} sq ft",
                        f"{proximity_to_city_km} km",
                        year,
                        month
                    ]
                })
                st.dataframe(summary_df1, use_container_width=True, hide_index=True)
            
            with col_b:
                summary_df2 = pd.DataFrame({
                    'Infrastructure': [
                        'Road Access', 'Water Supply', 'Electricity', 'Sewage System',
                        'Road Width', 'Corner Plot'
                    ],
                    'Status': [
                        road_access,
                        water_supply,
                        electricity,
                        sewage_system,
                        f"{road_width_ft} ft",
                        corner_plot
                    ]
                })
                st.dataframe(summary_df2, use_container_width=True, hide_index=True)
        
        
        with st.expander("🏫 Nearby Facilities Summary"):
            facility_df = pd.DataFrame({
                'Facility': [
                    'School', 'Hospital', 'Metro Station', 'Airport',
                    'Shopping Mall', 'Railway Station'
                ],
                'Distance (km)': [
                    school_distance_km,
                    hospital_distance_km,
                    metro_distance_km,
                    airport_distance_km,
                    mall_distance_km,
                    railway_distance_km
                ]
            })
            st.dataframe(facility_df, use_container_width=True, hide_index=True)
            
            st.info(f"📍 Premium Location: {'Yes' if premium_location else 'No'} | "
                    f"Infrastructure Score: {infrastructure_score}/4")

else:
    st.info("👈 Please train the model first by running `python main.py`")
    st.code("python main.py", language="bash")

st.markdown("---")
st.caption("🏡 Land Price Prediction System | Powered by XGBoost & Streamlit | Indian Cities Edition")
