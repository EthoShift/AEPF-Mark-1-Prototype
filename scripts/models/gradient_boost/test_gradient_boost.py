from .gradient_boost import HRGradientBoostModel
from .generate_report import ReportGenerator
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

def setup_logging():
    """Set up logging configuration"""
    log_dir = Path('logs/gradient_boost')
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

def get_key_strength(emp_details):
    """Determine key strength based on employee metrics"""
    if emp_details['EngagementSurvey'] >= 4.5:
        return "High Engagement"
    elif emp_details['EmpSatisfaction'] >= 4.5:
        return "High Satisfaction"
    elif emp_details['SpecialProjectsCount'] >= 3:
        return "Project Leadership"
    elif emp_details['PerformanceScore'] == "Exceeds":
        return "Outstanding Performance"
    elif emp_details['Absences'] <= 1:
        return "Strong Reliability"
    else:
        return "Balanced Performance"

def test_gradient_boost_model():
    """Run test to identify top 15 candidates"""
    logger = setup_logging()
    report_gen = ReportGenerator()
    
    try:
        logger.info("Starting Gradient Boost Model Test for Top 15 Candidates")
        
        # Load the HR dataset
        data_path = Path("data/datasets/hr_analytics")
        hr_files = list(data_path.glob("*HRDataset_v14*.csv"))
        if not hr_files:
            raise FileNotFoundError("HRDataset v14 not found")
        
        dataset = pd.read_csv(hr_files[0])
        total_employees = len(dataset)
        
        # Prepare and run model (existing code)
        features = [
            'Salary', 'EngagementSurvey', 'EmpSatisfaction', 
            'SpecialProjectsCount', 'DaysLateLast30', 'Absences',
            'PerformanceScore', 'Department', 'Position'
        ]
        
        data_subset = dataset[features + ['Termd']]
        all_indices = np.arange(len(data_subset))
        
        model = HRGradientBoostModel()
        X_train, X_test, y_train, y_test = model.prepare_data(data_subset, target_col='Termd')
        model.train(X_train, y_train)
        predictions, probabilities = model.predict(X_test)
        results = model.evaluate(X_test, y_test)
        
        # Get top 15 candidates
        candidate_scores = pd.DataFrame({
            'Score': probabilities[:, 1],
            'Original_Index': all_indices[-len(X_test):]
        })
        top_15 = candidate_scores.nlargest(15, 'Score')
        
        # Generate formatted report
        report_path = Path('reports/gradient_boost')
        report_path.mkdir(parents=True, exist_ok=True)
        report_file = report_path / f'top_15_candidates_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_file, 'w') as f:
            # Title and Date
            f.write("TOP 15 CANDIDATES REPORT\n")
            f.write("=======================\n\n")
            f.write(f"Generated: {datetime.now().strftime('%B %d, %Y')}\n\n")
            
            # Model Summary
            f.write("MODEL SUMMARY\n")
            f.write("-------------\n")
            f.write(f"Model Type: Gradient Boost Classifier\n")
            f.write(f"Dataset Size: {total_employees} employees\n")
            f.write(f"Model Accuracy: {float(results['classification_report'].split()[-8]):.1%}\n\n")
            
            # Overview Statement
            f.write("EXECUTIVE SUMMARY\n")
            f.write("----------------\n")
            f.write("This report presents the top 15 candidates recommended for interview consideration, ")
            f.write("ranked by their suitability scores based on performance, engagement, and reliability metrics.\n\n")
            
            # Recommended Candidates Table
            f.write("RECOMMENDED CANDIDATES\n")
            f.write("---------------------\n")
            f.write("Rank | ID      | Department      | Position          | Perf Score | Engagement | Satisfaction | Key Strength\n")
            f.write("-" * 100 + "\n")
            
            for rank, (_, row) in enumerate(top_15.iterrows(), 1):
                emp_details = dataset.iloc[int(row['Original_Index'])]
                key_strength = get_key_strength(emp_details)
                
                f.write(f"{rank:2d}   | {emp_details['EmpID']:<7} | ")
                f.write(f"{emp_details['Department']:<14} | ")
                f.write(f"{emp_details['Position'][:15]:<15} | ")
                f.write(f"{emp_details['PerformanceScore']:<10} | ")
                f.write(f"{emp_details['EngagementSurvey']:>9.1f} | ")
                f.write(f"{emp_details['EmpSatisfaction']:>11.1f} | ")
                f.write(f"{key_strength}\n")
            
            f.write("\nINTERVIEW RECOMMENDATIONS\n")
            f.write("-----------------------\n")
            f.write("Focus Areas:\n")
            f.write("1. Leadership and Project Management\n")
            f.write("   - Discuss special project experiences and outcomes\n")
            f.write("   - Explore team collaboration and leadership style\n\n")
            f.write("2. Engagement and Commitment\n")
            f.write("   - Review engagement survey responses\n")
            f.write("   - Discuss career goals and growth opportunities\n\n")
            f.write("3. Performance and Reliability\n")
            f.write("   - Examine performance review highlights\n")
            f.write("   - Discuss attendance and punctuality record\n\n")
            
            f.write("CLOSING NOTES\n")
            f.write("------------\n")
            f.write("These recommendations are based on objective metrics and model analysis. ")
            f.write("The interview process should focus on validating the identified strengths ")
            f.write("and exploring areas for potential growth. Final decisions should consider ")
            f.write("both these recommendations and standard interview assessments.\n")
        
        logger.info(f"Test report saved to: {report_file}")
        return True, report_file
        
    except Exception as e:
        logger.error(f"Error in gradient boost test: {str(e)}")
        return False, str(e)

if __name__ == "__main__":
    success, result = test_gradient_boost_model()
    if success:
        print(f"\nTest completed successfully. Report saved to: {result}")
        
        # Display the report contents
        with open(result, 'r') as f:
            print("\nReport Contents:")
            print("=" * 80)
            print(f.read())
    else:
        print(f"\nTest failed with error: {result}") 