
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_data():
    np.random.seed(42)
    
    samples_per_year = 200
    years = list(range(2015, 2025))
    n_samples = samples_per_year * len(years)
    
    cities = {
        'Mumbai': {'base_price': 25000, 'growth_rate': 1.08},
        'Pune': {'base_price': 8000, 'growth_rate': 1.10},
        'Nagpur': {'base_price': 4500, 'growth_rate': 1.07},
        'Ahmedabad': {'base_price': 6000, 'growth_rate': 1.09},
        'Rajkot': {'base_price': 3500, 'growth_rate': 1.06},
        'Thane': {'base_price': 12000, 'growth_rate': 1.08},
        'Nashik': {'base_price': 5000, 'growth_rate': 1.07},
        'Surat': {'base_price': 5500, 'growth_rate': 1.08},
    }
    
    location_types = ['Urban', 'Suburban', 'Rural']
    
    data = []
    
    for _ in range(n_samples):
        city = np.random.choice(list(cities.keys()))
        city_info = cities[city]
        
        year = np.random.choice(years)
        
        if city in ['Mumbai', 'Pune', 'Ahmedabad']:
            location = np.random.choice(location_types, p=[0.6, 0.3, 0.1])
        else:
            location = np.random.choice(location_types, p=[0.4, 0.4, 0.2])
        
        if location == 'Urban':
            area_sqft = np.random.randint(500, 3000)
        elif location == 'Suburban':
            area_sqft = np.random.randint(1000, 5000)
        else: 
            area_sqft = np.random.randint(2000, 10000)
        
        if location == 'Urban':
            proximity_to_city_km = np.random.randint(1, 10)
        elif location == 'Suburban':
            proximity_to_city_km = np.random.randint(10, 30)
        else: 
            proximity_to_city_km = np.random.randint(30, 100)
        
        school_distance_km = np.random.uniform(0.5, 15)
        hospital_distance_km = np.random.uniform(0.5, 20)
        metro_distance_km = np.random.uniform(0.5, 25) if city in ['Mumbai', 'Pune', 'Nagpur', 'Ahmedabad'] else np.nan
        airport_distance_km = np.random.uniform(5, 50)
        mall_distance_km = np.random.uniform(1, 20)
        railway_distance_km = np.random.uniform(0.5, 15)
        
        
        road_access = np.random.choice(['Yes', 'No'], p=[0.85, 0.15])
        water_supply = np.random.choice(['Yes', 'No'], p=[0.90, 0.10])
        electricity = np.random.choice(['Yes', 'No'], p=[0.95, 0.05])
        sewage_system = np.random.choice(['Yes', 'No'], p=[0.75, 0.25])
        
        
        road_width_ft = np.random.choice([20, 30, 40, 60, 80], p=[0.2, 0.3, 0.25, 0.15, 0.1])
        corner_plot = np.random.choice(['Yes', 'No'], p=[0.15, 0.85])
        
        
        years_from_base = year - 2015
        base_price = city_info['base_price'] * (city_info['growth_rate'] ** years_from_base)
        
        
        price_per_sqft = base_price
        
        
        if location == 'Urban':
            price_per_sqft *= 1.5
        elif location == 'Suburban':
            price_per_sqft *= 1.0
        else: 
            price_per_sqft *= 0.5
        
        
        price_per_sqft *= (1 - (proximity_to_city_km / 200))
        
        
        if school_distance_km < 2:
            price_per_sqft *= 1.15
        elif school_distance_km < 5:
            price_per_sqft *= 1.08
        
        if hospital_distance_km < 3:
            price_per_sqft *= 1.12
        elif hospital_distance_km < 7:
            price_per_sqft *= 1.05
        
        if not pd.isna(metro_distance_km) and metro_distance_km < 2:
            price_per_sqft *= 1.25
        elif not pd.isna(metro_distance_km) and metro_distance_km < 5:
            price_per_sqft *= 1.15
        
        if airport_distance_km < 10:
            price_per_sqft *= 1.10
        
        if mall_distance_km < 3:
            price_per_sqft *= 1.08
        
        if railway_distance_km < 2:
            price_per_sqft *= 1.12
        
        
        if road_access == 'Yes':
            price_per_sqft *= 1.20
        if water_supply == 'Yes':
            price_per_sqft *= 1.15
        if electricity == 'Yes':
            price_per_sqft *= 1.10
        if sewage_system == 'Yes':
            price_per_sqft *= 1.08
        
        
        if road_width_ft >= 60:
            price_per_sqft *= 1.12
        elif road_width_ft >= 40:
            price_per_sqft *= 1.05
        
        if corner_plot == 'Yes':
            price_per_sqft *= 1.10
        
        
        price_per_sqft *= np.random.uniform(0.90, 1.10)
        
        
        record = {
            'city': city,
            'location_type': location,
            'area_sqft': area_sqft,
            'proximity_to_city_km': proximity_to_city_km,
            'school_distance_km': round(school_distance_km, 2),
            'hospital_distance_km': round(hospital_distance_km, 2),
            'metro_distance_km': round(metro_distance_km, 2) if not pd.isna(metro_distance_km) else None,
            'airport_distance_km': round(airport_distance_km, 2),
            'mall_distance_km': round(mall_distance_km, 2),
            'railway_distance_km': round(railway_distance_km, 2),
            'road_access': road_access,
            'water_supply': water_supply,
            'electricity': electricity,
            'sewage_system': sewage_system,
            'road_width_ft': road_width_ft,
            'corner_plot': corner_plot,
            'year': year,
            'month': np.random.randint(1, 13),
            'price_per_sqft': round(price_per_sqft, 2)
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    df = df.sort_values(['year', 'month']).reset_index(drop=True)
    
    output_path = Path(__file__).parent.parent / 'data' / 'raw' / 'land_prices_raw.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print("=" * 60)
    print("✓ Land Price Dataset Generated Successfully!")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Years covered: {df['year'].min()} - {df['year'].max()}")
    print(f"Cities included: {', '.join(df['city'].unique())}")
    print(f"Price range: ₹{df['price_per_sqft'].min():.2f} - ₹{df['price_per_sqft'].max():.2f} per sqft")
    print(f"\nSaved to: {output_path}")
    print("=" * 60)
    
    print("\nCity-wise Average Prices (₹/sqft):")
    print(df.groupby('city')['price_per_sqft'].mean().sort_values(ascending=False).round(2))
    
    return df


if __name__ == "__main__":
    generate_data()
