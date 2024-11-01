from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRGradientBoostModel:
    def __init__(self, random_state=42):
        """Initialize the Gradient Boosting model for HR analytics"""
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, data_path: str = "data/datasets/hr_analytics/"):
        """Load the HR dataset"""
        try:
            # Find the most recent HR dataset
            data_dir = Path(data_path)
            hr_files = list(data_dir.glob("*HRDataset*.csv"))
            if not hr_files:
                raise FileNotFoundError("No HR dataset found")
            
            latest_file = max(hr_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading dataset from: {latest_file}")
            
            # Load the dataset
            df = pd.read_csv(latest_file)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Attrition'):
        """Prepare data for training"""
        try:
            # Handle categorical variables
            df_encoded = pd.get_dummies(df, drop_first=True)
            
            # Separate features and target
            X = df_encoded.drop(columns=[target_col])
            y = df_encoded[target_col]
            
            # Store feature columns for future use
            self.feature_columns = X.columns
            self.target_column = target_col
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info("Data preparation completed successfully")
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train(self, X_train, y_train):
        """Train the model"""
        try:
            logger.info("Starting model training...")
            self.model.fit(X_train, y_train)
            logger.info("Model training completed")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Generate evaluation metrics
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            feature_imp = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results = {
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'feature_importance': feature_imp
            }
            
            logger.info("Model evaluation completed")
            logger.info("\nClassification Report:\n" + report)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def predict(self, X):
        """Make predictions on new data"""
        try:
            # Scale the input data
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

def main():
    """Main function to demonstrate model usage"""
    try:
        # Initialize model
        model = HRGradientBoostModel()
        
        # Load data
        df = model.load_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test = model.prepare_data(df)
        
        # Train model
        model.train(X_train, y_train)
        
        # Evaluate model
        results = model.evaluate(X_test, y_test)
        
        # Print feature importance
        print("\nTop 10 Most Important Features:")
        print(results['feature_importance'].head(10))
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 