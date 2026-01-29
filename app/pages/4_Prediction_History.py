"""
Prediction History Page
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Prediction History", page_icon="📜", layout="wide")

st.title("📜 Prediction History")
st.markdown("View and manage past predictions")
st.markdown("---")

# Try to connect to database
try:
    from src.db import Database
    
    db = Database()
    db.connect()
    
    if db.conn is not None:
        # Get prediction history
        history = db.get_prediction_history(limit=100)
        
        if history:
            # Convert to DataFrame
            history_df = pd.DataFrame(history)
            
            st.subheader(f"📊 Recent Predictions ({len(history_df)} records)")
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", len(history_df))
            
            with col2:
                avg_price = history_df['predicted_price'].mean()
                st.metric("Average Price/sq ft", f"₹{avg_price:.2f}")
            
            with col3:
                max_price = history_df['predicted_price'].max()
                st.metric("Highest Price/sq ft", f"₹{max_price:.2f}")
            
            with col4:
                min_price = history_df['predicted_price'].min()
                st.metric("Lowest Price/sq ft", f"₹{min_price:.2f}")
            
            st.markdown("---")
            
            # Display table
            display_df = history_df[[
                'id', 'area_sqft', 'location', 'proximity_to_city_km',
                'road_access', 'water_supply', 'electricity', 'year',
                'predicted_price', 'created_at'
            ]].copy()
            
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                display_df.style.format({
                    'predicted_price': '₹{:.2f}',
                    'area_sqft': '{:,}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Download option
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name="prediction_history.csv",
                mime="text/csv"
            )
        else:
            st.info("📭 No prediction history found. Make some predictions first!")
        
        db.close()
    else:
        st.warning("⚠️ Database connection failed. History feature requires PostgreSQL.")
        st.markdown("""
        **To enable prediction history:**
        1. Install PostgreSQL
        2. Create a database named `landprice_db`
        3. Set environment variables:
           - `DB_HOST`
           - `DB_NAME`
           - `DB_USER`
           - `DB_PASSWORD`
           - `DB_PORT`
        4. Run `python src/db.py` to create tables
        """)

except ImportError:
    st.error("⚠️ Database module not available. Please check your installation.")
except Exception as e:
    st.error(f"⚠️ Error loading prediction history: {e}")
    
    # Show sample data structure
    st.subheader("📋 Sample History Structure")
    sample_df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Area (sq ft)': [2000, 3500, 1500],
        'Location': ['Urban', 'Suburban', 'Rural'],
        'Distance (km)': [5, 15, 40],
        'Road Access': ['Yes', 'Yes', 'No'],
        'Water Supply': ['Yes', 'Yes', 'No'],
        'Electricity': ['Yes', 'Yes', 'Yes'],
        'Year': [2024, 2024, 2023],
        'Predicted Price': [85.50, 72.30, 45.20],
        'Created At': ['2024-01-15 10:30', '2024-01-15 11:45', '2024-01-14 09:15']
    })
    
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("📜 Prediction History | Database-backed storage")
