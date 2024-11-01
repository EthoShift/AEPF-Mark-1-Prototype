from performance_classifier import PerformanceBasedClassifier
from summary_report import SummarizedReportGenerator
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

def setup_logging():
    """Set up logging for test"""
    log_dir = Path('logs/test_summary')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f'test_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_summary_test():
    """Run complete test of classification and summary generation"""
    logger = setup_logging()
    
    try:
        logger.info("Starting summary test...")
        
        # 1. Load Data
        logger.info("Loading HR dataset...")
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        logger.info(f"Loaded dataset from: {latest_file}")
        logger.info(f"Dataset shape: {df.shape}")
        
        # 2. Initialize and Run Classifier
        logger.info("Initializing performance classifier...")
        classifier = PerformanceBasedClassifier()
        
        # Prepare data
        X, y = classifier.prepare_data(df)
        logger.info(f"Prepared data shape: {X.shape}")
        
        # Train classifier
        logger.info("Training classifier...")
        classifier.train(X, y)
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = classifier.predict_performance(X)
        
        # 3. Generate Summary Report
        logger.info("Generating summary report...")
        report_gen = SummarizedReportGenerator()
        
        # Create candidate data with necessary information
        candidate_data = df[['EngagementSurvey', 'EmpSatisfaction', 
                           'SpecialProjectsCount', 'DaysLateLast30', 'Absences']]
        
        # Add a Name column if not present (using index as placeholder)
        if 'Name' not in df.columns:
            candidate_data['Name'] = [f"Candidate_{i}" for i in range(len(df))]
        else:
            candidate_data['Name'] = df['Name']
        
        # Generate summary
        summary = report_gen.generate_summary(predictions, candidate_data)
        
        # 4. Display Results
        print("\nSummary Report Generated:")
        print("=" * 50)
        print(summary)
        print("=" * 50)
        
        logger.info("Test completed successfully")
        
        return predictions, summary
        
    except Exception as e:
        logger.error(f"Error in summary test: {str(e)}")
        raise

if __name__ == "__main__":
    run_summary_test() 