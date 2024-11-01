import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    def __init__(self):
        """Initialize report generator"""
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        self.reports_dir = Path('reports/baseline')
        self.plots_dir = Path('reports/baseline/plots')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_baseline_report(self, model, X_test, y_test, feature_names, candidates_df):
        """Generate report with basic metrics"""
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred)
        
        # Get feature importance
        importance = self._get_feature_importance(model, feature_names)
        
        # Create report
        report = "MODEL PERFORMANCE REPORT\n"
        report += "=====================\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "CLASSIFICATION METRICS\n"
        report += "---------------------\n"
        report += class_report
        report += "\n\nFEATURE IMPORTANCE\n"
        report += "------------------\n"
        report += importance.to_string()
        
        return report
    
    def _get_feature_importance(self, model, feature_names):
        """Extract and format feature importance"""
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).set_index('feature').sort_values('importance', ascending=False)
        
        return importance
    
    def _create_visualization(self, metrics, feature_importance, recommendations):
        """Create visualization summary"""
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        
        # Feature importance plot
        feature_importance.head(10).plot(
            kind='barh',
            ax=axes[0],
            title='Top 10 Important Features'
        )
        axes[0].set_xlabel('Importance Score')
        
        # Score distribution
        if 'score' in recommendations.columns:
            sns.histplot(
                data=recommendations['score'],
                bins=20,
                ax=axes[1]
            )
            axes[1].set_title('Distribution of Scores')
            axes[1].set_xlabel('Score')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'baseline_summary_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        plt.close()

def main():
    """Generate baseline report for the model"""
    try:
        # Load data
        print("\nLoading dataset...")
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        candidates_df = pd.read_csv(latest_file)
        
        # Run bias analysis to get model and splits
        from test_bias import test_bias_analysis
        result = test_bias_analysis()
        
        # Unpack results
        bias_metrics, report, model, performance_metrics, X_test, y_test = result
        
        # Generate baseline report
        print("\nGenerating baseline report...")
        report_gen = ReportGenerator()
        baseline_report = report_gen.generate_baseline_report(
            model=model,
            X_test=X_test,
            y_test=y_test,
            feature_names=performance_metrics['feature_names'],
            candidates_df=candidates_df
        )
        
        print("\nBaseline Report Generated:")
        print(baseline_report)
        
    except Exception as e:
        print(f"Error generating baseline report: {str(e)}")

if __name__ == "__main__":
    main() 