from bias_analysis import BiasAnalyzer
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import logging

def run_bias_test():
    """Run comprehensive bias analysis test"""
    try:
        print("Starting Bias Analysis Test...")
        
        # Load data
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        
        print(f"\nLoading dataset from: {latest_file}")
        df = pd.read_csv(latest_file)
        
        # Display available columns
        print("\nAvailable columns in dataset:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
        
        # Get sensitive features
        print("\nIdentifying sensitive features...")
        sensitive_features = []
        for col in df.columns:
            if any(term in col.lower() for term in ['gender', 'race', 'age', 'ethnic']):
                sensitive_features.append(col)
                print(f"Found sensitive feature: {col}")
        
        if not sensitive_features:
            print("No standard sensitive features found. Please specify columns to check for bias:")
            user_input = input("Enter column names (comma-separated): ")
            sensitive_features = [col.strip() for col in user_input.split(',')]
        
        # Get target column
        target_column = input("\nEnter the target column name: ")
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset")
        
        # Initialize analyzer with specified sensitive features
        analyzer = BiasAnalyzer()
        analyzer.sensitive_features = sensitive_features
        
        print(f"\nAnalyzing bias for features: {', '.join(sensitive_features)}")
        
        # Prepare data
        X = df.drop(sensitive_features + [target_column], axis=1)
        y = df[target_column]
        sensitive_data = df[sensitive_features]
        
        # Handle categorical variables
        X = pd.get_dummies(X)
        
        # Split data
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X, y, sensitive_data, test_size=0.2, random_state=42
        )
        
        # Train model
        print("\nTraining model...")
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Perform bias analysis
        print("\nPerforming bias analysis...")
        bias_metrics, report = analyzer.analyze_bias(model, X_test, y_test, sens_test)
        
        # Display results
        print("\nBias Analysis Results:")
        print("=" * 50)
        print(report)
        
        print("\nBias analysis completed successfully!")
        print("Check the 'plots' and 'reports' directories for detailed results.")
        
        return bias_metrics, report
        
    except Exception as e:
        print(f"Error in bias analysis test: {str(e)}")
        raise

if __name__ == "__main__":
    run_bias_test() 