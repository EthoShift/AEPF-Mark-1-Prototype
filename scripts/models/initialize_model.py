from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_model():
    """Initialize the Gradient Boosting model with default parameters"""
    try:
        # Initialize model with random_state for reproducibility
        model = GradientBoostingClassifier(random_state=42)
        
        logger.info("Model initialized with parameters:")
        logger.info(f"- n_estimators: {model.n_estimators}")
        logger.info(f"- learning_rate: {model.learning_rate}")
        logger.info(f"- max_depth: {model.max_depth}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

def verify_model(model):
    """Verify model initialization"""
    try:
        # Check model attributes
        attributes = {
            'n_estimators': 100,  # default value
            'learning_rate': 0.1,  # default value
            'max_depth': 3        # default value
        }
        
        logger.info("\nVerifying model attributes:")
        for attr, expected in attributes.items():
            actual = getattr(model, attr)
            logger.info(f"- {attr}: {actual} (Expected: {expected})")
            assert actual == expected, f"Unexpected {attr} value"
        
        logger.info("Model verification successful")
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        return False

def main():
    """Main function to initialize and verify model"""
    try:
        logger.info("Starting model initialization...")
        
        # Initialize model
        model = initialize_model()
        
        # Verify initialization
        if verify_model(model):
            logger.info("\nModel ready for training")
            
            # Print model details
            logger.info("\nModel Details:")
            logger.info(f"Type: {type(model).__name__}")
            logger.info(f"Parameters: {model.get_params()}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 