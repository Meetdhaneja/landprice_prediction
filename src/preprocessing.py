import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np


def preprocess_data():

    raw_path = Path(__file__).parent.parent / 'data' / 'raw' / 'land_prices_raw.csv'
    df = pd.read_csv(raw_path)
    
    print(f"✓ Loaded {len(df)} records")
    

    df['metro_distance_km'] = df['metro_distance_km'].fillna(999)
    
    
    df = df.dropna()
    
    
    label_encoders = {}
    categorical_cols = ['city', 'location_type', 'road_access', 'water_supply', 
                        'electricity', 'sewage_system', 'corner_plot']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    
    df['avg_facility_distance'] = (
        df['school_distance_km'] + 
        df['hospital_distance_km'] + 
        df['mall_distance_km'] + 
        df['railway_distance_km']
    ) / 4
     
    df['infrastructure_score'] = (
        (df['road_access'] == 'Yes').astype(int) +
        (df['water_supply'] == 'Yes').astype(int) +
        (df['electricity'] == 'Yes').astype(int) +
        (df['sewage_system'] == 'Yes').astype(int)
    )
    
    df['premium_location'] = (
        (df['metro_distance_km'] < 5) & 
        (df['school_distance_km'] < 2) & 
        (df['hospital_distance_km'] < 3)
    ).astype(int)
    
    output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'land_prices_clean.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Preprocessed {len(df)} records")
    print(f"✓ Features created: {len(df.columns)} total columns")
    print(f"✓ Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    preprocess_data()

