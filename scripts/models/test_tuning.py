from hyperparameter_tuning import ModelTuner
import pandas as pd
from pathlib import Path
from datetime import datetime

def examine_dataset(df):
    """Print dataset information"""
    print("\nDataset Information:")
    print("-" * 50)
    print("\nColumns in dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    print(f"\nTotal rows: {len(df)}")
    print("-" * 50)

def main():
    try:
        print("Starting model tuning test...")
        
        # Load data
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        
        df = pd.read_csv(latest_file)
        print(f"Loaded dataset from {latest_file}")
        
        # First, let's look at the data
        examine_dataset(df)
        
        # Wait for user input
        print("\nReview the columns above and press Enter to continue...")
        input()
        
        # Now ask for target column
        target_column = input("\nEnter the target column name exactly as shown above: ")
        
        if target_column not in df.columns:
            print(f"Error: '{target_column}' is not a valid column name.")
            print("Available columns are:")
            for col in df.columns:
                print(f"- {col}")
            return
        
        # Prepare data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Convert categorical columns to numeric
        categorical_columns = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nPrepared data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
        
        # Initialize and run tuner
        tuner = ModelTuner()
        best_model = tuner.tune_model(X_train, y_train)
        
        # Evaluate best model
        evaluation_results = tuner.evaluate_best_model(X_test, y_test)
        
        # Generate report
        report = generate_tuning_report(tuner, evaluation_results)
        print("\nTuning Report:")
        print(report)
        
    except Exception as e:
        print(f"Error in test execution: {str(e)}")
        raise

def generate_tuning_report(tuner, evaluation_results, output_dir='reports/tuning'):
    """Generate a detailed tuning and evaluation report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report = f"""
========== Hyperparameter Tuning Report ==========
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Model Configuration
------------------
Best Parameters Found:
{pd.Series(tuner.best_params).to_string()}

Cross-Validation Results
-----------------------
Best CV Score: {tuner.best_score:.4f}
Mean Test CV Score: {evaluation_results['cv_scores'].mean():.4f}
Std Test CV Score: {evaluation_results['cv_scores'].std():.4f}

Test Set Performance
-------------------
Accuracy: {evaluation_results['accuracy']:.4f}

Detailed Classification Report
----------------------------
{evaluation_results['classification_report']}

Parameter Search Space
--------------------
{pd.Series(tuner.param_grid).to_string()}

========== End of Report ==========
"""
    
    # Save report
    report_path = report_dir / f'tuning_report_{timestamp}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {report_path}")
    return report

if __name__ == "__main__":
    main() 