from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

class DataPipeline:
    def __init__(self):
        """Initialize data pipeline with logging"""
        self.setup_logging()
        self.pipeline = None
        self.numeric_features = None
        self.categorical_features = None
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs/pipeline')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_pipeline(self, X):
        """Create preprocessing and model pipeline"""
        try:
            # Identify numeric and categorical columns
            self.numeric_features = X.select_dtypes(
                include=['int64', 'float64']
            ).columns.tolist()
            
            self.categorical_features = X.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
            
            self.logger.info(f"Numeric features: {len(self.numeric_features)}")
            self.logger.info(f"Categorical features: {len(self.categorical_features)}")
            
            # Create preprocessing pipelines for numeric and categorical data
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_features),
                    ('cat', categorical_transformer, self.categorical_features)
                ]
            )
            
            # Create full pipeline
            self.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ])
            
            self.logger.info("Pipeline created successfully")
            return self.pipeline
            
        except Exception as e:
            self.logger.error(f"Error creating pipeline: {str(e)}")
            raise
    
    def fit_pipeline(self, X_train, y_train):
        """Fit the pipeline on training data"""
        try:
            self.logger.info("Fitting pipeline...")
            self.logger.info(f"Training data shape: {X_train.shape}")
            
            # Fit pipeline
            self.pipeline.fit(X_train, y_train)
            
            self.logger.info("Pipeline fitted successfully")
            
            # Log feature names after preprocessing
            self._log_feature_names()
            
            return self.pipeline
            
        except Exception as e:
            self.logger.error(f"Error fitting pipeline: {str(e)}")
            raise
    
    def _log_feature_names(self):
        """Log feature names after preprocessing"""
        try:
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            # Get feature names for numeric columns
            numeric_features = self.numeric_features
            
            # Get feature names for categorical columns (after one-hot encoding)
            cat_features = []
            if self.categorical_features:
                onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = onehot.get_feature_names_out(self.categorical_features)
            
            all_features = numeric_features + list(cat_features)
            
            self.logger.info(f"Total features after preprocessing: {len(all_features)}")
            self.logger.info("Feature names:")
            for i, feature in enumerate(all_features, 1):
                self.logger.info(f"{i}. {feature}")
            
        except Exception as e:
            self.logger.error(f"Error logging feature names: {str(e)}")
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        try:
            if self.pipeline is None:
                raise ValueError("Pipeline not fitted yet")
            
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            # Get feature names for numeric and categorical columns
            numeric_features = self.numeric_features
            cat_features = []
            
            if self.categorical_features:
                onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = list(onehot.get_feature_names_out(self.categorical_features))
            
            return numeric_features + cat_features
            
        except Exception as e:
            self.logger.error(f"Error getting feature names: {str(e)}")
            raise

def main():
    """Main function to demonstrate pipeline usage"""
    try:
        # Load data
        from test_feature_analysis import load_and_prepare_data
        df = load_and_prepare_data()
        
        # Get target column
        target_column = input("\nEnter the target column name from the list above: ")
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset")
        
        # Split features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and fit pipeline
        pipeline = DataPipeline()
        pipeline.create_pipeline(X)
        pipeline.fit_pipeline(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.pipeline.predict(X_test)
        
        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest accuracy: {accuracy:.4f}")
        
        # Get feature names
        feature_names = pipeline.get_feature_names()
        print(f"\nTotal features after preprocessing: {len(feature_names)}")
        
    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")

if __name__ == "__main__":
    main() 