import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def train_arima_model():
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'land_prices_clean.csv'
    df = pd.read_csv(data_path)
    
    print("=" * 60)
    print("Training ARIMA Forecasting Model")
    print("=" * 60)
    
    df['year_month'] = df['year'] + (df['month'] - 1) / 12
    
    time_series = df.groupby('year_month')['price_per_sqft'].mean().sort_index()
    
    print(f"✓ Time series data points: {len(time_series)}")
    print(f"✓ Date range: {time_series.index.min():.2f} - {time_series.index.max():.2f}")
    
    best_aic = float('inf')
    best_model = None
    best_order = None
    
    orders_to_try = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
        (3, 1, 1),
        (1, 1, 3)
    ]
    
    print("\n🔄 Finding best ARIMA parameters...")
    for order in orders_to_try:
        try:
            model = ARIMA(time_series, order=order)
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_model = fitted
                best_order = order
        except:
            continue
    
    if best_model is None:
        
        print("⚠️  Using fallback ARIMA(1,1,1)")
        model = ARIMA(time_series, order=(1, 1, 1))
        best_model = model.fit()
        best_order = (1, 1, 1)
    
    print(f"✓ Best ARIMA order: {best_order}")
    print(f"✓ AIC: {best_aic:.2f}")
    
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': best_model,
        'time_series': time_series,
        'order': best_order,
        'last_year': time_series.index.max()
    }
    
    joblib.dump(model_data, models_dir / 'arima_model.pkl')
    
    print(f"✓ ARIMA model saved to: {models_dir / 'arima_model.pkl'}")
    print("=" * 60)
    
    return best_model


def forecast_future_prices(years_ahead=5):
    
    models_dir = Path(__file__).parent.parent / 'models'
    model_data = joblib.load(models_dir / 'arima_model.pkl')
    
    model = model_data['model']
    last_year = model_data['last_year']
    
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
    
    forecast_df = pd.DataFrame({
        'year': forecast_years,
        'forecasted_price_per_sqft': forecast_prices
    })
    
    return forecast_df


def forecast_city_prices(city_name, years_ahead=5):
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'land_prices_clean.csv'
    df = pd.read_csv(data_path)
    
    city_df = df[df['city'] == city_name].copy()
    
    if len(city_df) == 0:
        print(f"⚠️  No data found for city: {city_name}")
        return None
    
    city_df['year_month'] = city_df['year'] + (city_df['month'] - 1) / 12
    city_series = city_df.groupby('year_month')['price_per_sqft'].mean().sort_index()
    
    try:
        model = ARIMA(city_series, order=(1, 1, 1))
        fitted_model = model.fit()
        
        steps = years_ahead * 12
        forecast = fitted_model.forecast(steps=steps)
        
        last_year = city_series.index.max()
        forecast_years = []
        forecast_prices = []
        
        for year in range(1, years_ahead + 1):
            year_value = int(last_year) + year
            start_idx = (year - 1) * 12
            end_idx = year * 12
            avg_price = forecast.iloc[start_idx:end_idx].mean()
            
            forecast_years.append(year_value)
            forecast_prices.append(avg_price)
        
        return pd.DataFrame({
            'year': forecast_years,
            'city': city_name,
            'forecasted_price_per_sqft': forecast_prices
        })
    
    except Exception as e:
        print(f"⚠️  Error forecasting for {city_name}: {e}")
        return None


if __name__ == "__main__":
    train_arima_model()
    
    print("\nTesting 5-year forecast:")
    forecast_df = forecast_future_prices(5)
    print(forecast_df)
