"""Test script for the prediction module."""

import sys
from profitron.prediction_manager import PredictionManager
from profitron.data_sources import BinanceDataSource
from profitron.strategies.simple_moving_average import SimpleMovingAverageStrategy

def test_prediction():
    """Test the prediction module."""
    print("Initializing prediction manager...")
    pm = PredictionManager()
    
    # Add strategy
    print("Adding strategy...")
    strategy = SimpleMovingAverageStrategy()
    pm.add_strategy(strategy, weight=1.0)
    
    # Add data source
    print("Adding data source...")
    pm.add_data_source('ETHUSDT', BinanceDataSource('ETHUSDT'))
    
    # Get prediction
    print("Getting prediction...")
    try:
        price, confidence = pm.get_price_prediction()
        print(f"Prediction successful!")
        print(f"Predicted price: ${price:.2f}")
        print(f"Confidence: {confidence:.2f}")
        return True
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False

if __name__ == "__main__":
    test_prediction()