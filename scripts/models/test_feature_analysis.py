from feature_analysis import FeatureAnalyzer
import pandas as pd
from pathlib import Path
import logging

def load_and_prepare_data():
    """Load and prepare the HR dataset"""
    try:
        # Load data
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        
        print(f"Loading dataset from: {latest_file}")
        df = pd.read_csv(latest_file)
        
        # Display initial dataset info
        print("\nDataset Information:")
        print("-" * 50)
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print("\nColumns:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def main():
    try:
        print("Starting Feature Analysis Test...")
        
        # Load data
        df = load_and_prepare_data()
        
        # Initialize analyzer
        analyzer = FeatureAnalyzer()
        print("\nFeature Analyzer initialized successfully")
        
        # Prepare features (example with dummy model)
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        
        # Convert categorical columns to numeric
        df_encoded = pd.get_dummies(df)
        
        # Split features and target (assuming 'Attrition' is target)
        target_column = input("\nEnter the target column name from the list above: ")
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset")
        
        X = df_encoded.drop(target_column, axis=1)
        y = df_encoded[target_column]
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a simple model
        print("\nTraining model for feature importance analysis...")
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Analyze feature importance
        print("\nAnalyzing feature importance...")
        feature_importance = analyzer.analyze_feature_importance(model, X.columns)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        analyzer.plot_feature_importance(feature_importance)
        
        # Generate and display report
        print("\nGenerating feature analysis report...")
        report = analyzer.generate_feature_report(feature_importance)
        print("\nFeature Analysis Report:")
        print(report)
        
        print("\nFeature analysis completed successfully!")
        print("Check the 'plots' and 'reports' directories for output files.")
        
    except Exception as e:
        print(f"Error in feature analysis test: {str(e)}")
        raise

if __name__ == "__main__":
    main() 