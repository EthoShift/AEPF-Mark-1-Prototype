from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json

class ModelTuner:
    def __init__(self):
        """Initialize model tuner with logging"""
        self.setup_logging()
        self.best_params = None
        self.best_score = None
        self.best_model = None
        
        # Define parameter grid
        self.param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs/tuning')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def tune_model(self, X_train, y_train, cv=5):
        """Perform grid search for hyperparameter tuning"""
        try:
            self.logger.info("Starting hyperparameter tuning...")
            self.logger.info(f"Parameter grid: {json.dumps(self.param_grid, indent=2)}")
            
            # Initialize base model
            base_model = GradientBoostingClassifier(random_state=42)
            
            # Set up GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=self.param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2
            )
            
            # Perform grid search
            self.logger.info("Running grid search...")
            grid_search.fit(X_train, y_train)
            
            # Store best results
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            self.best_model = grid_search.best_estimator_
            
            # Log results
            self.logger.info("\nGrid Search Results:")
            self.logger.info(f"Best parameters: {json.dumps(self.best_params, indent=2)}")
            self.logger.info(f"Best cross-validation score: {self.best_score:.4f}")
            
            # Save results
            self.save_tuning_results()
            
            return self.best_model
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise
    
    def evaluate_best_model(self, X_test, y_test):
        """Evaluate the best model on test data"""
        try:
            if self.best_model is None:
                raise ValueError("Model must be tuned before evaluation")
            
            # Make predictions
            y_pred = self.best_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            
            # Log evaluation results
            self.logger.info("\nBest Model Evaluation Results:")
            self.logger.info(f"Test Accuracy: {accuracy:.4f}")
            self.logger.info("\nClassification Report:\n" + class_report)
            
            # Perform cross-validation on best model
            cv_scores = cross_val_score(
                self.best_model, X_test, y_test, cv=5, scoring='accuracy'
            )
            
            self.logger.info("\nCross-validation scores:")
            self.logger.info(f"Mean CV Score: {cv_scores.mean():.4f}")
            self.logger.info(f"Std CV Score: {cv_scores.std():.4f}")
            
            return {
                'accuracy': accuracy,
                'classification_report': class_report,
                'cv_scores': cv_scores
            }
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            raise
    
    def save_tuning_results(self):
        """Save tuning results to file"""
        try:
            results_dir = Path('results/tuning')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'best_parameters': self.best_params,
                'best_score': float(self.best_score),
                'parameter_grid': self.param_grid
            }
            
            # Save as JSON
            results_file = results_dir / f'tuning_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Tuning results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving tuning results: {str(e)}")
            raise

def main():
    """Main function to demonstrate hyperparameter tuning"""
    try:
        # Load data
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        
        df = pd.read_csv(latest_file)
        print(f"Loaded dataset from {latest_file}")
        
        # Prepare data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X = df.drop('Attrition', axis=1)  # Adjust column name if different
        y = df['Attrition']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize tuner
        tuner = ModelTuner()
        
        # Perform tuning
        best_model = tuner.tune_model(X_train, y_train)
        
        # Evaluate best model
        evaluation_results = tuner.evaluate_best_model(X_test, y_test)
        
        print("\nTuning and evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 