from .generate_report import ReportGenerator
from .test_gradient_boost import test_gradient_boost_model
import logging
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Set up logging configuration"""
    log_dir = Path('logs/full_test')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f'full_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_full_test():
    """Run complete system test"""
    logger = setup_logging()
    
    try:
        logger.info("=== Starting Full System Test ===")
        
        # Run gradient boost test
        gb_success, gb_result = test_gradient_boost_model()
        if not gb_success:
            logger.error(f"Gradient Boost test failed: {gb_result}")
            return
        
        logger.info(f"Gradient Boost test completed successfully. Report at: {gb_result}")
        
    except Exception as e:
        logger.error(f"Error in full system test: {str(e)}")
        raise

if __name__ == "__main__":
    run_full_test() 