from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple

class PerformanceBasedClassifier:
    def __init__(self):
        """Initialize performance-based classifier"""
        self.setup_logging()
        self.model = self._configure_model()
        self.scaler = StandardScaler()
        
        # Define performance metrics and their weights
        self.performance_metrics = {
            'EngagementSurvey': {'weight': 0.25, 'optimal_direction': 'high'},
            'EmpSatisfaction': {'weight': 0.25, 'optimal_direction': 'high'},
            'SpecialProjectsCount': {'weight': 0.20, 'optimal_direction': 'high'},
            'DaysLateLast30': {'weight': 0.15, 'optimal_direction': 'low'},
            'Absences': {'weight': 0.15, 'optimal_direction': 'low'}
        }
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs/performance_classifier')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'performance_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _configure_model(self) -> HistGradientBoostingClassifier:
        """Configure the gradient boosting classifier"""
        return HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
            validation_fraction=0.2,
            early_stopping=True,
            verbose=1
        )
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data focusing on performance metrics"""
        try:
            self.logger.info("Preparing performance-based classification data...")
            
            # Validate required columns
            missing_columns = [col for col in self.performance_metrics.keys() if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Select only performance metrics and target
            X = df[list(self.performance_metrics.keys())].copy()
            y = df['Termd']
            
            # Scale performance metrics
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            self.logger.info(f"Data prepared with {len(self.performance_metrics)} performance metrics")
            return X_scaled, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the performance-based classifier"""
        try:
            self.logger.info("Training performance-based classifier...")
            
            # Log class distribution
            class_dist = pd.Series(y).value_counts(normalize=True)
            self.logger.info(f"Class distribution:\n{class_dist}")
            
            # Fit the model
            self.model.fit(X, y)
            
            # Log performance metric importance
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = dict(zip(X.columns, self.model.feature_importances_))
                self.logger.info("\nPerformance metric importance:")
                for metric, importance in sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    self.logger.info(f"{metric}: {importance:.4f}")
            
            self.logger.info("Training completed")
            
        except Exception as e:
            self.logger.error(f"Error training classifier: {str(e)}")
            raise
    
    def predict_performance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict performance-based suitability"""
        try:
            self.logger.info("Generating performance-based predictions...")
            
            # Scale input data
            X_scaled = pd.DataFrame(
                self.scaler.transform(X[list(self.performance_metrics.keys())]),
                columns=X.columns,
                index=X.index
            )
            
            # Get base predictions
            base_probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            # Calculate weighted performance score
            performance_scores = self._calculate_performance_scores(X)
            
            # Combine model predictions with performance scores
            final_scores = 0.7 * base_probabilities + 0.3 * performance_scores
            
            # Create results DataFrame
            results = pd.DataFrame({
                'model_score': base_probabilities,
                'performance_score': performance_scores,
                'final_score': final_scores,
                'suitability_rating': pd.qcut(
                    final_scores,
                    q=3,
                    labels=['Low', 'Medium', 'High']
                )
            })
            
            self.logger.info("Predictions generated successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def _calculate_performance_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate weighted performance scores"""
        scores = np.zeros(len(X))
        
        for metric, config in self.performance_metrics.items():
            values = X[metric].values
            
            # Normalize values
            if config['optimal_direction'] == 'high':
                normalized = (values - values.min()) / (values.max() - values.min())
            else:  # 'low'
                normalized = 1 - (values - values.min()) / (values.max() - values.min())
            
            # Apply weight
            scores += normalized * config['weight']
        
        # Normalize final scores
        return (scores - scores.min()) / (scores.max() - scores.min())
    
    def generate_performance_report(self, predictions: pd.DataFrame) -> str:
        """Generate detailed performance report"""
        report = "Performance-Based Classification Report\n"
        report += "=" * 50 + "\n\n"
        
        # Distribution of suitability ratings
        report += "Suitability Distribution:\n"
        report += f"{predictions['suitability_rating'].value_counts().to_string()}\n\n"
        
        # Score statistics
        report += "Score Statistics:\n"
        for score_type in ['model_score', 'performance_score', 'final_score']:
            report += f"\n{score_type}:\n"
            stats = predictions[score_type].describe()
            report += f"  Mean: {stats['mean']:.3f}\n"
            report += f"  Std: {stats['std']:.3f}\n"
            report += f"  Min: {stats['min']:.3f}\n"
            report += f"  Max: {stats['max']:.3f}\n"
        
        return report

def main():
    """Test the performance-based classifier"""
    try:
        # Load data
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        # Initialize classifier
        classifier = PerformanceBasedClassifier()
        
        # Prepare data
        X, y = classifier.prepare_data(df)
        
        # Train classifier
        classifier.train(X, y)
        
        # Generate predictions
        predictions = classifier.predict_performance(X)
        
        # Generate and print report
        print("\nPerformance Report:")
        print(classifier.generate_performance_report(predictions))
        
    except Exception as e:
        print(f"Error in performance classifier test: {str(e)}")

if __name__ == "__main__":
    main() 