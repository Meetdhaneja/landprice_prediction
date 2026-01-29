import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Database:
    
    def __init__(self):
        self.conn = None
        self.cursor = None
    
    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'landprice_db'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'password'),
                port=os.getenv('DB_PORT', '5432')
            )
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            print("✓ Database connected successfully")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
    
    def create_tables(self):
        create_predictions_table = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            area_sqft INTEGER,
            location VARCHAR(50),
            proximity_to_city_km INTEGER,
            road_access VARCHAR(10),
            water_supply VARCHAR(10),
            electricity VARCHAR(10),
            year INTEGER,
            predicted_price FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            self.cursor.execute(create_predictions_table)
            self.conn.commit()
            print("✓ Tables created successfully")
        except Exception as e:
            print(f"✗ Table creation failed: {e}")
    
    def save_prediction(self, features, prediction):
        insert_query = """
        INSERT INTO predictions 
        (area_sqft, location, proximity_to_city_km, road_access, 
    water_supply, electricity, year, predicted_price)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        try:
            self.cursor.execute(insert_query, (
                features['area_sqft'],
                features['location'],
                features['proximity_to_city_km'],
                features['road_access'],
                features['water_supply'],
                features['electricity'],
                features['year'],
                prediction
            ))
            self.conn.commit()
            return self.cursor.fetchone()['id']
        except Exception as e:
            print(f"✗ Failed to save prediction: {e}")
            return None
    
    def get_prediction_history(self, limit=100):
        query = """
        SELECT * FROM predictions 
        ORDER BY created_at DESC 
        LIMIT %s;
        """
        
        try:
            self.cursor.execute(query, (limit,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"✗ Failed to retrieve history: {e}")
            return []
    
    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✓ Database connection closed")


if __name__ == "__main__":
    db = Database()
    db.connect()
    db.create_tables()
    db.close()
