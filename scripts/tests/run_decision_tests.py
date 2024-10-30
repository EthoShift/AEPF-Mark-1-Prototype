import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.tests.test_decision_logic import DecisionLogicTestSuite

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run decision logic tests"""
    # Initialize test suite
    test_suite = DecisionLogicTestSuite()
    
    # Run tests
    results = test_suite.run_tests()
    
    # Generate report
    test_suite.generate_report(results)
    
    # Return success if all tests passed
    return all(result.passed for result in results)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}", exc_info=True)
        sys.exit(1) 