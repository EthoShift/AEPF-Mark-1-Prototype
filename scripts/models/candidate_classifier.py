from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple

class CandidateSuitabilityClassifier:
    def __init__(self):
        """Initialize candidate suitability classifier"""
        self.setup_logging()
        self.model = self._configure_model()
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.sensitivity_weights = self._get_sensitivity_weights()
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs/classifier')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
    
    def _get_sensitivity_weights(self) -> Dict[str, float]:
        """Define sensitivity weights for features"""
        return {
            # Demographic features
            'GenderID': 0.15,
            'RaceDesc': 0.15,
            'MaritalStatusID': 0.10,
            'Sex': 0.15,
            'HispanicLatino': 0.15,
            
            # Organizational features
            'DeptID': 0.10,
            'State': 0.05,
            'RecruitmentSource': 0.10,
            
            # Performance indicators
            'EngagementSurvey': 0.20,
            'EmpSatisfaction': 0.20,
            'SpecialProjectsCount': 0.15,
            'DaysLateLast30': 0.25,
            'Absences': 0.20
        }
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for classification"""
        try:
            self.logger.info("Preparing data for classification...")
            
            # Separate target
            X = df.drop('Termd', axis=1)
            y = df['Termd']
            
            # Handle categorical variables
            categorical_features = X.select_dtypes(include=['object']).columns
            X_encoded = pd.get_dummies(X, columns=categorical_features)
            
            # Scale numerical features
            numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
            X_encoded[numerical_features] = self.scaler.fit_transform(X[numerical_features])
            
            self.logger.info(f"Data prepared: {X_encoded.shape[1]} features")
            return X_encoded, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the classifier"""
        try:
            self.logger.info("Training candidate suitability classifier...")
            
            # Fit the model
            self.model.fit(X, y)
            
            # Store feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
                
                # Log top features
                top_features = sorted(
                    self.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                self.logger.info("Top 10 important features:")
                for feature, importance in top_features:
                    self.logger.info(f"{feature}: {importance:.4f}")
            
            self.logger.info("Training completed")
            
        except Exception as e:
            self.logger.error(f"Error training classifier: {str(e)}")
            raise
    
    def predict_suitability(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict candidate suitability with sensitivity adjustments"""
        try:
            self.logger.info("Generating suitability predictions...")
            
            # Get base predictions
            base_probabilities = self.model.predict_proba(X)[:, 1]
            
            # Apply sensitivity adjustments
            adjusted_scores = self._apply_sensitivity_adjustments(X, base_probabilities)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'base_score': base_probabilities,
                'adjusted_score': adjusted_scores,
                'suitability_class': ['High' if s >= 0.7 else 'Medium' if s >= 0.4 else 'Low' 
                                    for s in adjusted_scores]
            })
            
            self.logger.info("Predictions generated successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def _apply_sensitivity_adjustments(self, X: pd.DataFrame, base_scores: np.ndarray) -> np.ndarray:
        """Apply sensitivity adjustments to base scores"""
        adjusted_scores = base_scores.copy()
        
        for feature, weight in self.sensitivity_weights.items():
            if feature in X.columns:
                feature_values = X[feature]
                
                # Calculate feature-specific adjustment
                if feature in ['DaysLateLast30', 'Absences']:
                    # Lower values are better
                    adjustment = -1 * weight * (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
                else:
                    # Higher values are better
                    adjustment = weight * (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
                
                # Apply adjustment
                adjusted_scores += adjustment
        
        # Normalize scores to [0, 1]
        adjusted_scores = (adjusted_scores - adjusted_scores.min()) / (adjusted_scores.max() - adjusted_scores.min())
        
        return adjusted_scores
    
    def get_feature_importance_report(self) -> str:
        """Generate feature importance report"""
        if not self.feature_importance:
            return "No feature importance available. Model needs to be trained first."
        
        report = "Feature Importance Report\n"
        report += "=" * 40 + "\n\n"
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Add feature importance details
        for feature, importance in sorted_features:
            report += f"{feature:30} {importance:.4f}\n"
        
        return report

def main():
    """Test the candidate suitability classifier"""
    try:
        # Load data
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        # Initialize classifier
        classifier = CandidateSuitabilityClassifier()
        
        # Prepare data
        X, y = classifier.prepare_data(df)
        
        # Train classifier
        classifier.train(X, y)
        
        # Generate predictions
        predictions = classifier.predict_suitability(X)
        
        # Print feature importance report
        print("\nFeature Importance Report:")
        print(classifier.get_feature_importance_report())
        
        # Print prediction summary
        print("\nPrediction Summary:")
        print(predictions['suitability_class'].value_counts())
        
    except Exception as e:
        print(f"Error in classifier test: {str(e)}")

if __name__ == "__main__":
    main() 