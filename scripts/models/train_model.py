from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, random_state=42):
        """Initialize the model trainer"""
        self.model = GradientBoostingClassifier(random_state=random_state)
        self.training_start = None
        self.training_end = None
        
    def train_model(self, X_train, y_train):
        """Train the model and log progress"""
        try:
            logger.info("Starting model training...")
            self.training_start = datetime.now()
            
            # Fit the model
            self.model.fit(X_train, y_train)
            
            self.training_end = datetime.now()
            training_duration = (self.training_end - self.training_start).total_seconds()
            
            logger.info(f"Model training completed in {training_duration:.2f} seconds")
            logger.info(f"Model parameters: {self.model.get_params()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return False
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance with multiple metrics"""
        try:
            logger.info("Evaluating model performance...")
            
            # Generate predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Log results
            logger.info("\nModel Evaluation Results:")
            logger.info(f"Accuracy Score: {accuracy:.4f}")
            logger.info("\nClassification Report:\n" + class_report)
            logger.info("\nConfusion Matrix:\n" + str(conf_matrix))
            
            # Store metrics
            self.evaluation_metrics = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred
            }
            
            return self.evaluation_metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            return None
    
    def generate_predictions(self, X_test):
        """Generate predictions for test data"""
        try:
            logger.info("Generating predictions...")
            
            # Generate predictions and probabilities
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)
            
            logger.info(f"Generated predictions for {len(y_pred)} samples")
            
            return y_pred, y_prob
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return None, None
    
    def log_training_summary(self):
        """Generate and log training summary"""
        try:
            if not hasattr(self, 'evaluation_metrics'):
                logger.warning("No evaluation metrics available")
                return
            
            summary = f"""
            ========== Model Training Summary ==========
            Training Start: {self.training_start}
            Training End: {self.training_end}
            Duration: {(self.training_end - self.training_start).total_seconds():.2f} seconds
            
            Model Parameters:
            {self.model.get_params()}
            
            Performance Metrics:
            - Accuracy: {self.evaluation_metrics['accuracy']:.4f}
            
            Classification Report:
            {self.evaluation_metrics['classification_report']}
            
            Confusion Matrix:
            {self.evaluation_metrics['confusion_matrix']}
            =========================================
            """
            
            logger.info(summary)
            
            # Save summary to file
            summary_path = Path('logs/training_summaries')
            summary_path.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path / f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
                f.write(summary)
                
        except Exception as e:
            logger.error(f"Error generating training summary: {str(e)}")

def main():
    """Main function to demonstrate model training and evaluation"""
    try:
        # Load data (assuming data is already prepared)
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        
        df = pd.read_csv(latest_file)
        logger.info(f"Loaded dataset from {latest_file}")
        
        # Prepare data (simplified for demonstration)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X = df.drop('Attrition', axis=1)  # Adjust column name if different
        y = df['Attrition']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
        trainer = ModelTrainer()
        
        # Train
        if trainer.train_model(X_train, y_train):
            # Evaluate
            metrics = trainer.evaluate_model(X_test, y_test)
            
            # Generate predictions
            y_pred, y_prob = trainer.generate_predictions(X_test)
            
            # Log summary
            trainer.log_training_summary()
            
            logger.info("Model training and evaluation completed successfully")
        else:
            logger.error("Model training failed")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 