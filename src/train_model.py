"""
Model training and evaluation with comprehensive features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor


def train_and_save_model():
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'land_prices_clean.csv'
    df = pd.read_csv(data_path)
    
    print("=" * 60)
    print("Training Land Price Prediction Model")
    print("=" * 60)
    
    feature_cols = [
        'area_sqft', 'proximity_to_city_km', 'year', 'month',
        
        'city_encoded', 'location_type_encoded',
        
        'school_distance_km', 'hospital_distance_km', 'metro_distance_km',
        'airport_distance_km', 'mall_distance_km', 'railway_distance_km',
        
        'road_access_encoded', 'water_supply_encoded',
        'electricity_encoded', 'sewage_system_encoded',
        
        'road_width_ft', 'corner_plot_encoded',
        
        'avg_facility_distance', 'infrastructure_score', 'premium_location'
    ]
    
    X = df[feature_cols]
    y = df['price_per_sqft']
    
    print(f"✓ Dataset: {len(df)} samples")
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Target: price_per_sqft")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("\n🔄 Training XGBoost model...")
    pipeline.fit(X_train, y_train)
    print("✓ Training completed!")
    
    y_train_pred = pipeline.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    y_test_pred = pipeline.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    print("\n" + "=" * 60)
    print("Model Performance Metrics")
    print("=" * 60)
    print("\nTraining Set:")
    print(f"  RMSE: ₹{train_rmse:.2f}")
    print(f"  MAE:  ₹{train_mae:.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print("\nTest Set:")
    print(f"  RMSE: ₹{test_rmse:.2f}")
    print(f"  MAE:  ₹{test_mae:.2f}")
    print(f"  R²:   {test_r2:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")
    print("=" * 60)
    
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, models_dir / 'xgb.pkl')
    print(f"\n✓ Model saved to: {models_dir / 'xgb.pkl'}")
    
    feature_metadata = {
        'features': feature_cols,
        'n_features': len(feature_cols),
        'feature_names': {
            'basic': ['area_sqft', 'proximity_to_city_km', 'year', 'month'],
            'categorical': ['city_encoded', 'location_type_encoded'],
            'facilities': ['school_distance_km', 'hospital_distance_km', 'metro_distance_km',
                           'airport_distance_km', 'mall_distance_km', 'railway_distance_km'],
            'infrastructure': ['road_access_encoded', 'water_supply_encoded',
                             'electricity_encoded', 'sewage_system_encoded'],
            'property': ['road_width_ft', 'corner_plot_encoded'],
            'engineered': ['avg_facility_distance', 'infrastructure_score', 'premium_location']
        }
    }
    
    with open(models_dir / 'features.json', 'w') as f:
        json.dump(feature_metadata, f, indent=2)
    print(f"✓ Features metadata saved to: {models_dir / 'features.json'}")
    
    metrics = {
        'train': {
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'r2_score': float(train_r2)
        },
        'test': {
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2_score': float(test_r2),
            'mape': float(test_mape)
        }
    }
    
    with open(models_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {models_dir / 'metrics.json'}")
    
    feature_importance = pipeline.named_steps['model'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 60)
    print("Top 10 Most Important Features")
    print("=" * 60)
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:.<40} {row['importance']:.4f}")
    print("=" * 60)
    
    return pipeline, metrics


if __name__ == "__main__":
    train_and_save_model()
