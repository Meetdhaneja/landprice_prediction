import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Future Price Forecast", page_icon="📈", layout="wide")

st.title("📈 Future Price Forecast (1-5 Years)")
st.markdown("Time series forecasting using ARIMA model for Indian cities")
st.markdown("---")

@st.cache_resource
def load_arima_model():
    models_dir = Path(__file__).parent.parent.parent / 'models'
    try:
        model_data = joblib.load(models_dir / 'arima_model.pkl')
        return model_data
    except FileNotFoundError:
        st.error("⚠️ ARIMA model not found! Please run the training pipeline first.")
        return None

@st.cache_data
def load_historical_data():
    try:
        data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'land_prices_clean.csv'
        df = pd.read_csv(data_path)
        return df
    except:
        return None

arima_model_data = load_arima_model()
historical_df = load_historical_data()

if arima_model_data is not None and historical_df is not None:
    st.subheader("⚙️ Forecast Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        years_ahead = st.slider(
            "Number of years to forecast",
            min_value=1,
            max_value=5,
            value=5
        )
    
    with col2:
        cities = historical_df['city'].unique().tolist()
        selected_city = st.selectbox(
            "Select City for Forecast",
            options=['Overall'] + cities
        )
    
    if st.button("🔮 Generate Forecast", type="primary"):
        try:
            if selected_city == 'Overall':
                
                model = arima_model_data['model']
                last_year = arima_model_data['last_year']
                time_series = arima_model_data['time_series']
                
                
                steps = years_ahead * 12
                forecast = model.forecast(steps=steps)
                
                
                forecast_years = []
                forecast_prices = []
                
                for year in range(1, years_ahead + 1):
                    year_value = int(last_year) + year
                    start_idx = (year - 1) * 12
                    end_idx = year * 12
                    avg_price = forecast.iloc[start_idx:end_idx].mean()
                    
                    forecast_years.append(year_value)
                    forecast_prices.append(avg_price)
                
                
                historical_df['year_month'] = historical_df['year'] + (historical_df['month'] - 1) / 12
                historical_series = historical_df.groupby('year_month')['price_per_sqft'].mean().sort_index()
                historical_years = [int(ym) for ym in historical_series.index]
                historical_prices = historical_series.values
                
            else:
                
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
                from forecast import forecast_city_prices
                
                forecast_df = forecast_city_prices(selected_city, years_ahead)
                
                if forecast_df is None:
                    st.error(f"Could not generate forecast for {selected_city}")
                    st.stop()
                
                forecast_years = forecast_df['year'].tolist()
                forecast_prices = forecast_df['forecasted_price_per_sqft'].tolist()
                
                
                city_df = historical_df[historical_df['city'] == selected_city].copy()
                city_df['year_month'] = city_df['year'] + (city_df['month'] - 1) / 12
                city_series = city_df.groupby('year_month')['price_per_sqft'].mean().sort_index()
                historical_years = [int(ym) for ym in city_series.index]
                historical_prices = city_series.values
            
            
            st.success("✅ Forecast Generated!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                
                ax.plot(historical_years, historical_prices, 
                        marker='o', label='Historical', linewidth=2, 
                        color='#1f77b4', markersize=4)
                
                
                ax.plot(forecast_years, forecast_prices, 
                        marker='s', label='Forecast', linewidth=2, 
                        linestyle='--', color='#ff7f0e', markersize=6)
                
                ax.set_xlabel('Year', fontsize=12, fontweight='bold')
                ax.set_ylabel('Average Price per sq ft (₹)', fontsize=12, fontweight='bold')
                
                title = f'Land Price Forecast - {selected_city}' if selected_city != 'Overall' else 'Overall Land Price Forecast'
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
                
                st.pyplot(fig)
            
            with col2:
                st.subheader("📊 Forecast Data")
                
                forecast_display_df = pd.DataFrame({
                    'Year': forecast_years,
                    'Forecasted Price (₹/sqft)': [f'₹{p:,.2f}' for p in forecast_prices]
                })
                
                st.dataframe(
                    forecast_display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                
                st.metric("Average Forecast", f"₹{sum(forecast_prices)/len(forecast_prices):,.2f}")
                
                trend = "📈 Increasing" if forecast_prices[-1] > forecast_prices[0] else "📉 Decreasing"
                growth_rate = ((forecast_prices[-1] / forecast_prices[0]) - 1) * 100
                st.metric("Trend", trend)
                st.metric(f"{years_ahead}-Year Growth", f"{growth_rate:+.2f}%")
            
            
            st.markdown("---")
            st.subheader("📋 Detailed Forecast Breakdown")
            
            detailed_df = pd.DataFrame({
                'Year': forecast_years,
                'Forecasted Price/sqft': forecast_prices,
                'For 1000 sqft': [p * 1000 for p in forecast_prices],
                'For 2000 sqft': [p * 2000 for p in forecast_prices],
                'For 5000 sqft': [p * 5000 for p in forecast_prices]
            })
            
            st.dataframe(
                detailed_df.style.format({
                    'Forecasted Price/sqft': '₹{:,.2f}',
                    'For 1000 sqft': '₹{:,.0f}',
                    'For 2000 sqft': '₹{:,.0f}',
                    'For 5000 sqft': '₹{:,.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("👈 Please train the ARIMA model first by running `python main.py`")
    st.code("python main.py", language="bash")

st.markdown("---")
st.caption("📈 Future Price Forecast | ARIMA Time Series Model | 1-5 Year Predictions")
