import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class PercentileClassifier(BaseEstimator, ClassifierMixin):
    """Classifier that categorizes candidates based on score thresholds"""
    
    def __init__(self):
        self.high_threshold = 0.75
        self.medium_threshold = 0.50
    
    def classify_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify candidates into suitability categories based on scores
        
        Args:
            df: DataFrame with 'score' column
            
        Returns:
            DataFrame with added 'suitability' column
        """
        # Create copy to avoid modifying original
        results = df.copy()
        
        # Add suitability classification
        conditions = [
            (results['score'] > self.high_threshold),
            (results['score'] > self.medium_threshold) & (results['score'] <= self.high_threshold),
            (results['score'] <= self.medium_threshold)
        ]
        choices = ['High', 'Medium', 'Low']
        
        results['suitability'] = np.select(conditions, choices, default='Low')
        
        # Ensure required columns
        if 'CandidateID' not in results.columns:
            results['CandidateID'] = [f'CAND_{i:03d}' for i in range(1, len(results) + 1)]
        
        # Select and order columns
        return results[['CandidateID', 'suitability', 'score']]
    
    def fit(self, X, y=None):
        """Placeholder fit method to maintain sklearn compatibility"""
        return self
    
    def predict(self, X):
        """Predict suitability categories"""
        scores = X.mean(axis=1)
        conditions = [
            (scores > self.high_threshold),
            (scores > self.medium_threshold) & (scores <= self.high_threshold)
        ]
        choices = [2, 1]  # 2=High, 1=Medium, 0=Low
        return np.select(conditions, choices, default=0)