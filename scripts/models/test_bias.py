from bias_analysis import BiasAnalyzer
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

def test_bias_analysis():
    try:
        print("\n=== Starting Bias Analysis Test ===\n")
        
        # 1. Load Data
        print("Loading HR dataset...")
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        # 2. Display Dataset Info
        print(f"\nDataset loaded: {latest_file.name}")
        print(f"Total records: {len(df)}")
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}. {col}")
        
        # 3. Get User Input
        print("\n=== Configuration ===")
        target_col = input("\nEnter target column name: ")
        sensitive_cols = input("Enter sensitive feature columns (comma-separated): ").split(',')
        sensitive_cols = [col.strip() for col in sensitive_cols]
        
        # 4. Validate Inputs
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        for col in sensitive_cols:
            if col not in df.columns:
                raise ValueError(f"Sensitive feature '{col}' not found in dataset")
        
        # 5. Prepare Data
        print("\n=== Preparing Data ===")
        X = df.drop(sensitive_cols + [target_col], axis=1)
        y = df[target_col]
        sensitive_features = df[sensitive_cols]
        
        # Handle categorical variables
        print("Converting categorical variables...")
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = pd.Categorical(X[col]).codes
        
        # Convert target to numeric if needed
        if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
            print("Converting target to numeric...")
            y = pd.Categorical(y).codes
        
        # 6. Split Data
        print("Splitting data...")
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X, y, sensitive_features, test_size=0.2, random_state=42, stratify=y
        )
        
        # 7. Train Model
        print("\n=== Training Model ===")
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            validation_fraction=0.2,
            early_stopping=True,
            verbose=1
        )
        
        print("Training model with early stopping...")
        model.fit(X_train, y_train)
        
        # Print training results
        print("\nTraining completed:")
        print(f"Best iteration: {model.n_iter_}")
        print(f"Training score: {model.score(X_train, y_train):.4f}")
        print(f"Validation score: {model.score(X_test, y_test):.4f}")
        
        # 8. Run Bias Analysis
        print("\n=== Running Bias Analysis ===")
        analyzer = BiasAnalyzer()
        analyzer.sensitive_features = sensitive_cols
        
        print("Analyzing bias...")
        bias_metrics, report = analyzer.analyze_bias(model, X_test, y_test, sens_test)
        
        # 9. Display Results
        print("\n=== Bias Analysis Results ===")
        print(report)
        
        # 10. Save Model Performance Metrics
        performance_metrics = {
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test),
            'n_iterations': model.n_iter_,
            'feature_names': list(X.columns),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save metrics
        metrics_dir = Path('metrics')
        metrics_dir.mkdir(exist_ok=True)
        pd.Series(performance_metrics).to_json(
            metrics_dir / f'model_metrics_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        print("\n=== Analysis Complete ===")
        print("\nResults saved in:")
        print("- plots/bias_analysis/: Visualization plots")
        print("- reports/bias_analysis/: Detailed reports")
        print("- logs/bias_analysis/: Analysis logs")
        print("- metrics/: Model performance metrics")
        
        return bias_metrics, report, model, performance_metrics, X_test, y_test
        
    except Exception as e:
        print(f"\nError during bias analysis: {str(e)}")
        raise

if __name__ == "__main__":
    test_bias_analysis() 