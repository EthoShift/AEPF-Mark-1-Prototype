from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

class HoldoutValidator:
    def __init__(self, model, holdout_size=0.2):
        """Initialize holdout validator"""
        self.model = model
        self.holdout_size = holdout_size
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs/holdout_validation')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'holdout_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary directories"""
        self.plots_dir = Path('plots/holdout_validation')
        self.reports_dir = Path('reports/holdout_validation')
        
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def split_holdout_set(self, X, y):
        """Split data into training and holdout sets"""
        try:
            # First split: training + validation vs holdout
            X_temp, X_holdout, y_temp, y_holdout = train_test_split(
                X, y,
                test_size=self.holdout_size,
                random_state=42,
                stratify=y
            )
            
            # Second split: training vs validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=0.2,
                random_state=42,
                stratify=y_temp
            )
            
            self.logger.info(f"Data split complete:")
            self.logger.info(f"Training set: {X_train.shape}")
            self.logger.info(f"Validation set: {X_val.shape}")
            self.logger.info(f"Holdout set: {X_holdout.shape}")
            
            return X_train, X_val, X_holdout, y_train, y_val, y_holdout
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def evaluate_performance(self, X, y, set_name=""):
        """Evaluate model performance on a dataset"""
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'roc_auc': roc_auc_score(y, y_prob)
            }
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y, y_pred)
            
            return metrics, conf_matrix, y_prob
            
        except Exception as e:
            self.logger.error(f"Error evaluating {set_name} performance: {str(e)}")
            raise
    
    def compare_performance(self, val_metrics, holdout_metrics):
        """Compare validation and holdout performance"""
        try:
            comparison = pd.DataFrame({
                'Validation': val_metrics,
                'Holdout': holdout_metrics
            }).round(4)
            
            # Calculate differences
            comparison['Difference'] = (
                comparison['Holdout'] - comparison['Validation']
            ).round(4)
            
            # Calculate percent change
            comparison['% Change'] = (
                (comparison['Difference'] / comparison['Validation']) * 100
            ).round(2)
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing performance: {str(e)}")
            raise
    
    def plot_confusion_matrices(self, val_conf_matrix, holdout_conf_matrix):
        """Plot confusion matrices for comparison"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Validation confusion matrix
            sns.heatmap(
                val_conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax1
            )
            ax1.set_title('Validation Set\nConfusion Matrix')
            
            # Holdout confusion matrix
            sns.heatmap(
                holdout_conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax2
            )
            ax2.set_title('Holdout Set\nConfusion Matrix')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / f'confusion_matrices_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Confusion matrices plot saved to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error plotting confusion matrices: {str(e)}")
            raise
    
    def generate_validation_report(self, comparison_df, val_conf_matrix, holdout_conf_matrix):
        """Generate detailed validation report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            report = f"""
========== Holdout Validation Report ==========
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Performance Comparison
--------------------
{comparison_df.to_string()}

Analysis
--------
1. Stability Assessment:
{self._assess_stability(comparison_df)}

2. Overfitting Analysis:
{self._assess_overfitting(comparison_df)}

3. Model Reliability:
{self._assess_reliability(comparison_df)}

Recommendations
-------------
{self._generate_recommendations(comparison_df)}

Confusion Matrix Analysis
-----------------------
Validation Set:
{val_conf_matrix}

Holdout Set:
{holdout_conf_matrix}

========== End of Report ==========
"""
            
            # Save report
            report_path = self.reports_dir / f'holdout_validation_{timestamp}.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Validation report saved to {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating validation report: {str(e)}")
            raise
    
    def _assess_stability(self, comparison_df):
        """Assess model stability based on performance differences"""
        assessments = []
        for metric, row in comparison_df.iterrows():
            diff = abs(row['% Change'])
            if diff < 5:
                assessments.append(f"- {metric}: Stable (±{diff:.1f}%)")
            elif diff < 10:
                assessments.append(f"- {metric}: Moderate variation (±{diff:.1f}%)")
            else:
                assessments.append(f"- {metric}: High variation (±{diff:.1f}%) - Requires attention")
        
        return "\n".join(assessments)
    
    def _assess_overfitting(self, comparison_df):
        """Assess potential overfitting"""
        if (comparison_df['Validation'] > comparison_df['Holdout']).all():
            return "Potential overfitting detected - Model performs consistently better on validation set"
        elif abs(comparison_df['% Change']).mean() > 10:
            return "Some overfitting concerns - Large performance variations between sets"
        else:
            return "No significant overfitting detected - Performance is consistent across sets"
    
    def _assess_reliability(self, comparison_df):
        """Assess model reliability"""
        reliability_score = (100 - abs(comparison_df['% Change']).mean()) / 100
        
        if reliability_score >= 0.9:
            return f"High reliability (Score: {reliability_score:.2f}) - Model performance is very consistent"
        elif reliability_score >= 0.8:
            return f"Good reliability (Score: {reliability_score:.2f}) - Model performance is reasonably consistent"
        else:
            return f"Moderate reliability (Score: {reliability_score:.2f}) - Consider model refinement"
    
    def _generate_recommendations(self, comparison_df):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check for large performance differences
        if abs(comparison_df['% Change']).max() > 10:
            recommendations.append(
                "- Consider collecting more training data or adjusting model complexity"
            )
        
        # Check for consistent underperformance
        if (comparison_df['Holdout'] < comparison_df['Validation']).all():
            recommendations.append(
                "- Review feature selection and consider feature engineering"
            )
        
        # Check specific metrics
        if comparison_df.loc['precision', '% Change'] < -5:
            recommendations.append(
                "- Focus on improving model precision through threshold adjustment"
            )
        
        if comparison_df.loc['recall', '% Change'] < -5:
            recommendations.append(
                "- Consider adjusting model to improve recall on unseen data"
            )
        
        if not recommendations:
            recommendations.append("- Model performs well - Continue monitoring performance")
        
        return "\n".join(recommendations)

def main():
    """Main function to demonstrate holdout validation"""
    try:
        # Load your trained model and data
        from test_feature_analysis import load_and_prepare_data
        
        # Load data
        df = load_and_prepare_data()
        
        # Get target column
        target_column = input("\nEnter the target column name from the list above: ")
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset")
        
        # Prepare data
        df_encoded = pd.get_dummies(df)
        X = df_encoded.drop(target_column, axis=1)
        y = df_encoded[target_column]
        
        # Initialize validator with a new model
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=42)
        validator = HoldoutValidator(model)
        
        # Split data
        X_train, X_val, X_holdout, y_train, y_val, y_holdout = validator.split_holdout_set(X, y)
        
        # Train model
        print("\nTraining model...")
        model.fit(X_train, y_train)
        
        # Evaluate on validation and holdout sets
        print("\nEvaluating performance...")
        val_metrics, val_conf_matrix, val_prob = validator.evaluate_performance(X_val, y_val, "validation")
        holdout_metrics, holdout_conf_matrix, holdout_prob = validator.evaluate_performance(X_holdout, y_holdout, "holdout")
        
        # Compare performance
        comparison = validator.compare_performance(val_metrics, holdout_metrics)
        
        # Plot confusion matrices
        validator.plot_confusion_matrices(val_conf_matrix, holdout_conf_matrix)
        
        # Generate and display report
        report = validator.generate_validation_report(comparison, val_conf_matrix, holdout_conf_matrix)
        print("\nValidation Report:")
        print(report)
        
    except Exception as e:
        print(f"Error in holdout validation: {str(e)}")

if __name__ == "__main__":
    main() 