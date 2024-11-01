from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

class ModelValidator:
    def __init__(self, model, n_folds=5):
        """Initialize model validator"""
        self.model = model
        self.n_folds = n_folds
        self.setup_logging()
        
        # Define scoring metrics
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs/validation')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def perform_cross_validation(self, X, y):
        """Perform k-fold cross-validation with multiple metrics"""
        try:
            self.logger.info(f"Starting {self.n_folds}-fold cross-validation...")
            
            # Perform cross-validation with multiple metrics
            cv_results = cross_validate(
                self.model,
                X, y,
                cv=self.n_folds,
                scoring=self.scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Calculate and log results
            validation_stats = self._calculate_validation_stats(cv_results)
            self._log_validation_results(validation_stats)
            
            return validation_stats
            
        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise
    
    def evaluate_test_set(self, X_test, y_test):
        """Evaluate model on test set"""
        try:
            self.logger.info("Evaluating model on test set...")
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            test_results = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            self._log_test_results(test_results)
            return test_results
            
        except Exception as e:
            self.logger.error(f"Error during test set evaluation: {str(e)}")
            raise
    
    def _calculate_validation_stats(self, cv_results):
        """Calculate statistics from cross-validation results"""
        stats = {}
        
        for metric in self.scoring.keys():
            train_scores = cv_results[f'train_{metric}']
            test_scores = cv_results[f'test_{metric}']
            
            stats[metric] = {
                'train': {
                    'mean': train_scores.mean(),
                    'std': train_scores.std(),
                    'scores': train_scores
                },
                'test': {
                    'mean': test_scores.mean(),
                    'std': test_scores.std(),
                    'scores': test_scores
                }
            }
        
        return stats
    
    def _log_validation_results(self, validation_stats):
        """Log cross-validation results"""
        self.logger.info("\nCross-validation Results:")
        
        for metric, results in validation_stats.items():
            self.logger.info(f"\n{metric.capitalize()} Scores:")
            self.logger.info("-" * 40)
            
            # Training scores
            self.logger.info("Training:")
            self.logger.info(f"Mean: {results['train']['mean']:.4f}")
            self.logger.info(f"Std: {results['train']['std']:.4f}")
            self.logger.info("Individual fold scores:")
            for i, score in enumerate(results['train']['scores'], 1):
                self.logger.info(f"Fold {i}: {score:.4f}")
            
            # Test scores
            self.logger.info("\nValidation:")
            self.logger.info(f"Mean: {results['test']['mean']:.4f}")
            self.logger.info(f"Std: {results['test']['std']:.4f}")
            self.logger.info("Individual fold scores:")
            for i, score in enumerate(results['test']['scores'], 1):
                self.logger.info(f"Fold {i}: {score:.4f}")
    
    def _log_test_results(self, test_results):
        """Log test set results"""
        self.logger.info("\nTest Set Results:")
        self.logger.info("-" * 40)
        
        for metric, value in test_results.items():
            self.logger.info(f"{metric.capitalize()}: {value:.4f}")
    
    def generate_validation_report(self, validation_stats, test_results):
        """Generate comprehensive validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path('reports/validation')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report = f"""
========== Model Validation Report ==========
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Cross-validation Results ({self.n_folds}-fold)
--------------------------------------------
"""
        
        # Add cross-validation results
        for metric, results in validation_stats.items():
            report += f"\n{metric.capitalize()}:\n"
            report += "-" * 40 + "\n"
            
            # Training results
            report += "Training:\n"
            report += f"Mean: {results['train']['mean']:.4f}\n"
            report += f"Std:  {results['train']['std']:.4f}\n"
            report += "Fold scores: "
            report += ", ".join([f"{score:.4f}" for score in results['train']['scores']])
            report += "\n"
            
            # Validation results
            report += "\nValidation:\n"
            report += f"Mean: {results['test']['mean']:.4f}\n"
            report += f"Std:  {results['test']['std']:.4f}\n"
            report += "Fold scores: "
            report += ", ".join([f"{score:.4f}" for score in results['test']['scores']])
            report += "\n"
        
        # Add test set results
        report += "\nTest Set Results\n"
        report += "-" * 40 + "\n"
        for metric, value in test_results.items():
            report += f"{metric.capitalize()}: {value:.4f}\n"
        
        report += "\n========== End of Report ==========\n"
        
        # Save report
        report_path = report_dir / f'validation_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"\nValidation report saved to {report_path}")
        return report

def main():
    """Main function to demonstrate model validation"""
    try:
        # Load your tuned model and data
        from test_tuning import main as load_tuned_model
        model, X_train, X_test, y_train, y_test = load_tuned_model()
        
        # Initialize validator
        validator = ModelValidator(model)
        
        # Perform cross-validation
        validation_stats = validator.perform_cross_validation(X_train, y_train)
        
        # Evaluate on test set
        test_results = validator.evaluate_test_set(X_test, y_test)
        
        # Generate and display report
        report = validator.generate_validation_report(validation_stats, test_results)
        print("\nValidation Report:")
        print(report)
        
    except Exception as e:
        print(f"Error in validation: {str(e)}")

if __name__ == "__main__":
    main() 