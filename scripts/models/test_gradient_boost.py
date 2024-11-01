import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from .gradient_boost import HRGradientBoostModel
from .generate_report import ReportGenerator

def setup_test_logging():
    """Configure logging for gradient boost tests"""
    log_dir = Path('logs/model_tests')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f'gradient_boost_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def generate_test_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'Attrition': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        'Age': np.random.normal(35, 8, n_samples),
        'YearsAtCompany': np.random.poisson(4, n_samples),
        'Salary': np.random.normal(70000, 20000, n_samples),
        'Department': np.random.choice(['Sales', 'IT', 'HR', 'Engineering'], size=n_samples),
        'Performance': np.random.choice(['Low', 'Medium', 'High'], size=n_samples)
    })

def test_gradient_boost_model():
    """Run comprehensive test of gradient boost model"""
    logger = setup_test_logging()
    report_gen = ReportGenerator()
    
    try:
        logger.info("Starting Gradient Boost Model Test")
        
        # Initialize model
        model = HRGradientBoostModel()
        
        # Generate test data
        test_data = generate_test_data()
        logger.info(f"Generated test data with shape: {test_data.shape}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = model.prepare_data(test_data)
        logger.info("Data preparation completed")
        
        # Train model
        model.train(X_train, y_train)
        logger.info("Model training completed")
        
        # Evaluate model
        results = model.evaluate(X_test, y_test)
        logger.info("Model evaluation completed")
        
        # Generate detailed report
        report = report_gen.generate_baseline_report(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=model.feature_columns,
            candidates_df=test_data
        )
        
        # Save report
        report_path = Path('reports/model_tests')
        report_path.mkdir(parents=True, exist_ok=True)
        report_file = report_path / f'gradient_boost_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_file, 'w') as f:
            f.write("=== Gradient Boost Model Test Report ===\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("=== Classification Report ===\n")
            f.write(results['classification_report'])
            f.write("\n\n=== Feature Importance ===\n")
            f.write(results['feature_importance'].to_string())
            f.write("\n\n=== Detailed Analysis ===\n")
            f.write(report)
        
        logger.info(f"Test report saved to: {report_file}")
        return True, report_file
        
    except Exception as e:
        logger.error(f"Error in gradient boost test: {str(e)}")
        return False, str(e)

def main():
    """Run gradient boost model test suite"""
    success, result = test_gradient_boost_model()
    if success:
        print(f"\nTest completed successfully. Report saved to: {result}")
    else:
        print(f"\nTest failed with error: {result}")

if __name__ == "__main__":
    main() 