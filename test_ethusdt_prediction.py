"""Test script to verify ETHUSDT prediction functionality."""

from profitron.prediction_manager import PredictionManager

def main():
    # Create prediction manager (will automatically initialize with ETHUSDT data source)
    print("Initializing PredictionManager...")
    manager = PredictionManager()
    
    # Verify ETHUSDT data source is available
    print(f"Available data sources: {list(manager.data_sources.keys())}")
    
    # Get a price prediction
    print("Getting price prediction for ETHUSDT...")
    try:
        predicted_price, confidence = manager.get_price_prediction()
        print(f"Predicted price: ${predicted_price:.2f}")
        print(f"Confidence: {confidence:.2f}")
        print("Prediction successful!")
    except Exception as e:
        print(f"Error getting prediction: {e}")

if __name__ == "__main__":
    main()