import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data_generator import generate_data
from src.preprocessing import preprocess_data
from src.train_model import train_and_save_model
from src.forecast import train_arima_model


def main():
    print("=" * 50)
    print("Land Price Prediction Pipeline")
    print("=" * 50)
    
    print("\n[1/4] Generating dummy data...")
    generate_data()
    
    print("\n[2/4] Preprocessing data...")
    preprocess_data()
    
    print("\n[3/4] Training XGBoost model...")
    train_and_save_model()
    
    print("\n[4/4] Training ARIMA model...")
    train_arima_model()
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)
    print("\nRun the Streamlit app with:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
