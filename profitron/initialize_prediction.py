"""Module for initializing prediction components."""

from .prediction_manager import PredictionManager
from .live_data_fetcher import LiveDataFetcher
from .strategies.simple_moving_average import SimpleMovingAverageStrategy

def initialize_prediction_components():
    """
    Initialize prediction manager and live data fetcher.
    
    Returns:
        tuple: (prediction_manager, live_data_fetcher)
    """
    # Initialize prediction manager
    prediction_manager = PredictionManager()
    
    # Add default strategy
    strategy = SimpleMovingAverageStrategy()
    prediction_manager.add_strategy(strategy, weight=1.0)
    
    # Initialize live data fetcher
    live_data_fetcher = LiveDataFetcher()
    
    return prediction_manager, live_data_fetcher