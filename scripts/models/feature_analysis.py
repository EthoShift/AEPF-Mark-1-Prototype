import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging

class FeatureAnalyzer:
    def __init__(self):
        """Initialize feature analyzer"""
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs/feature_analysis')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'feature_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary directories"""
        self.plots_dir = Path('plots/feature_importance')
        self.reports_dir = Path('reports/feature_analysis')
        
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_feature_importance(self, model, feature_names):
        """Extract and analyze feature importance"""
        try:
            # Get feature importance scores
            importance = model.feature_importances_
            
            # Create DataFrame with feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            
            # Sort by importance
            feature_importance = feature_importance.sort_values(
                'importance', ascending=False
            ).reset_index(drop=True)
            
            # Calculate cumulative importance
            feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
            
            # Add rank
            feature_importance['rank'] = range(1, len(feature_importance) + 1)
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {str(e)}")
            raise
    
    def plot_feature_importance(self, feature_importance, top_n=20):
        """Create visualizations for feature importance"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Plot top N features
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=feature_importance.head(top_n),
                x='importance',
                y='feature',
                palette='viridis'
            )
            plt.title(f'Top {top_n} Most Important Features')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.plots_dir / f'feature_importance_{timestamp}.png'
            plt.savefig(plot_path)
            self.logger.info(f"Feature importance plot saved to {plot_path}")
            
            # Cumulative importance plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(1, len(feature_importance) + 1),
                feature_importance['cumulative_importance'],
                'b-'
            )
            plt.xlabel('Number of Features')
            plt.ylabel('Cumulative Importance')
            plt.title('Cumulative Feature Importance')
            plt.grid(True)
            
            # Save cumulative plot
            cumulative_plot_path = self.plots_dir / f'cumulative_importance_{timestamp}.png'
            plt.savefig(cumulative_plot_path)
            self.logger.info(f"Cumulative importance plot saved to {cumulative_plot_path}")
            
            plt.close('all')
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}")
            raise
    
    def generate_feature_report(self, feature_importance, threshold=0.01):
        """Generate detailed feature importance report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            report = f"""
========== Feature Importance Analysis Report ==========
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Top 10 Most Important Features
-----------------------------
{feature_importance.head(10).to_string()}

Feature Importance Statistics
---------------------------
Total Features: {len(feature_importance)}
Features with importance > {threshold}: {len(feature_importance[feature_importance['importance'] > threshold])}
Cumulative importance of top 10: {feature_importance.head(10)['importance'].sum():.4f}

Key Insights
-----------
1. Primary Drivers:
   {self._format_key_features(feature_importance.head(5))}

2. Secondary Factors:
   {self._format_key_features(feature_importance.iloc[5:10])}

3. Low Impact Features:
   {self._format_key_features(feature_importance[feature_importance['importance'] < threshold].head(5))}

Recommendations
--------------
{self._generate_recommendations(feature_importance, threshold)}

========== End of Report ==========
"""
            
            # Save report
            report_path = self.reports_dir / f'feature_analysis_{timestamp}.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Feature analysis report saved to {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating feature report: {str(e)}")
            raise
    
    def _format_key_features(self, features):
        """Format feature information for report"""
        return "\n   ".join([
            f"- {row['feature']}: {row['importance']:.4f}"
            for _, row in features.iterrows()
        ])
    
    def _generate_recommendations(self, feature_importance, threshold):
        """Generate recommendations based on feature importance"""
        recommendations = []
        
        # High importance features
        top_features = feature_importance.head(3)
        recommendations.append(
            "Focus on top drivers:\n" +
            "\n".join([f"- Monitor and optimize {f}" for f in top_features['feature']])
        )
        
        # Low importance features
        low_imp_features = feature_importance[feature_importance['importance'] < threshold]
        if not low_imp_features.empty:
            recommendations.append(
                "Consider reviewing or removing low-impact features:\n" +
                "\n".join([f"- Evaluate necessity of {f}" for f in low_imp_features.head(3)['feature']])
            )
        
        # Feature reduction potential
        cumulative_95 = feature_importance[feature_importance['cumulative_importance'] <= 0.95]
        if len(cumulative_95) < len(feature_importance):
            recommendations.append(
                f"Model simplification potential:\n"
                f"- {len(cumulative_95)} features account for 95% of importance\n"
                f"- Consider reducing feature set to top {len(cumulative_95)} features"
            )
        
        return "\n\n".join(recommendations)

def main():
    """Main function to demonstrate feature analysis"""
    try:
        # Load your trained model and data
        from test_tuning import main as load_model
        model, X_train, X_test, y_train, y_test = load_model()
        
        # Initialize analyzer
        analyzer = FeatureAnalyzer()
        
        # Analyze feature importance
        feature_importance = analyzer.analyze_feature_importance(model, X_train.columns)
        
        # Create visualizations
        analyzer.plot_feature_importance(feature_importance)
        
        # Generate and display report
        report = analyzer.generate_feature_report(feature_importance)
        print("\nFeature Analysis Report:")
        print(report)
        
    except Exception as e:
        print(f"Error in feature analysis: {str(e)}")

if __name__ == "__main__":
    main() 