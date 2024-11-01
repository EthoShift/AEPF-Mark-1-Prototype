from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class BiasAnalyzer:
    def __init__(self):
        """Initialize bias analyzer"""
        self.setup_logging()
        self.setup_directories()
        self.sensitive_features = ['GenderID', 'RaceDesc']  # Add more as needed
        self.metrics = {}
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs/bias_analysis')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'bias_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary directories"""
        self.plots_dir = Path('plots/bias_analysis')
        self.reports_dir = Path('reports/bias_analysis')
        
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_bias(self, model, X, y, sensitive_features_data):
        """Analyze bias across sensitive feature groups"""
        try:
            self.logger.info("Starting bias analysis...")
            
            bias_metrics = {}
            for feature in self.sensitive_features:
                if feature in sensitive_features_data.columns:
                    bias_metrics[feature] = self._analyze_feature_bias(
                        model, X, y, sensitive_features_data[feature], feature
                    )
            
            # Generate visualizations
            self._plot_bias_metrics(bias_metrics)
            
            # Generate report
            report = self._generate_bias_report(bias_metrics)
            
            return bias_metrics, report
            
        except Exception as e:
            self.logger.error(f"Error in bias analysis: {str(e)}")
            raise
    
    def _calculate_confusion_matrix_metrics(self, y_true, y_pred):
        """Calculate metrics from confusion matrix with proper handling"""
        try:
            # Check if both classes are present in the true labels
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                self.logger.warning(f"Only class {unique_classes[0]} present in true labels")
                # Return safe default values
                return 0, 0, 0, 0
            
            # Generate confusion matrix with explicit labels
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            # Handle different matrix shapes
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                self.logger.warning("Unexpected confusion matrix shape")
                tn = cm[0, 0] if cm.shape == (1, 1) else cm[0, 0]
                fp = cm[0, 1] if cm.shape == (1, 2) else 0
                fn = cm[1, 0] if cm.shape == (2, 1) else 0
                tp = cm[1, 1] if cm.shape == (2, 2) else 0
            
            return tn, fp, fn, tp
            
        except Exception as e:
            self.logger.error(f"Error calculating confusion matrix metrics: {str(e)}")
            raise
    
    def _calculate_accuracy(self, y_true, y_pred):
        """Calculate accuracy with proper handling"""
        try:
            if len(y_true) == 0:
                return 0
            
            tn, fp, fn, tp = self._calculate_confusion_matrix_metrics(y_true, y_pred)
            return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating accuracy: {str(e)}")
            raise
    
    def _calculate_fpr(self, y_true, y_pred):
        """Calculate false positive rate with proper handling"""
        try:
            tn, fp, fn, tp = self._calculate_confusion_matrix_metrics(y_true, y_pred)
            return fp / (fp + tn) if (fp + tn) > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating false positive rate: {str(e)}")
            raise
    
    def _calculate_fnr(self, y_true, y_pred):
        """Calculate false negative rate with proper handling"""
        try:
            tn, fp, fn, tp = self._calculate_confusion_matrix_metrics(y_true, y_pred)
            return fn / (fn + tp) if (fn + tp) > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating false negative rate: {str(e)}")
            raise
    
    def _analyze_feature_bias(self, model, X, y, feature_values, feature_name):
        """Analyze bias for a specific feature with improved error handling"""
        try:
            # Get predictions
            y_pred = model.predict(X)
            
            # Calculate metrics for each group
            groups = feature_values.unique()
            group_metrics = {}
            
            for group in groups:
                mask = feature_values == group
                group_size = sum(mask)
                
                if group_size < 10:  # Minimum group size threshold
                    self.logger.warning(
                        f"Small group size ({group_size}) for {feature_name}={group}"
                    )
                
                if group_size > 0:
                    # Check class balance in group
                    group_classes = np.unique(y[mask])
                    if len(group_classes) < 2:
                        self.logger.warning(
                            f"Only class {group_classes[0]} present in group {group}"
                        )
                    
                    group_metrics[group] = {
                        'size': group_size,
                        'accuracy': self._calculate_accuracy(y[mask], y_pred[mask]),
                        'false_positive_rate': self._calculate_fpr(y[mask], y_pred[mask]),
                        'false_negative_rate': self._calculate_fnr(y[mask], y_pred[mask]),
                        'prediction_rate': sum(y_pred[mask]) / group_size if group_size > 0 else 0,
                        'class_distribution': {
                            str(c): sum(y[mask] == c) / group_size 
                            for c in np.unique(y)
                        }
                    }
            
            # Calculate disparate impact with error handling
            reference_group = list(group_metrics.keys())[0]
            ref_pred_rate = group_metrics[reference_group]['prediction_rate']
            
            for group in group_metrics:
                if group != reference_group:
                    if ref_pred_rate > 0:
                        group_metrics[group]['disparate_impact'] = (
                            group_metrics[group]['prediction_rate'] / ref_pred_rate
                        )
                    else:
                        group_metrics[group]['disparate_impact'] = 0
                        self.logger.warning(
                            f"Zero prediction rate for reference group {reference_group}"
                        )
            
            return group_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing bias for {feature_name}: {str(e)}")
            raise
    
    def _plot_bias_metrics(self, bias_metrics):
        """Create visualizations for bias metrics"""
        try:
            for feature, metrics in bias_metrics.items():
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Bias Metrics for {feature}')
                
                # Prepare data for plotting
                groups = list(metrics.keys())
                accuracies = [metrics[g]['accuracy'] for g in groups]
                fprs = [metrics[g]['false_positive_rate'] for g in groups]
                fnrs = [metrics[g]['false_negative_rate'] for g in groups]
                pred_rates = [metrics[g]['prediction_rate'] for g in groups]
                
                # Plot accuracy
                axes[0,0].bar(groups, accuracies)
                axes[0,0].set_title('Accuracy by Group')
                axes[0,0].set_ylim(0, 1)
                
                # Plot FPR
                axes[0,1].bar(groups, fprs)
                axes[0,1].set_title('False Positive Rate by Group')
                axes[0,1].set_ylim(0, 1)
                
                # Plot FNR
                axes[1,0].bar(groups, fnrs)
                axes[1,0].set_title('False Negative Rate by Group')
                axes[1,0].set_ylim(0, 1)
                
                # Plot prediction rate
                axes[1,1].bar(groups, pred_rates)
                axes[1,1].set_title('Prediction Rate by Group')
                axes[1,1].set_ylim(0, 1)
                
                plt.tight_layout()
                
                # Save plot
                plot_path = self.plots_dir / f'bias_metrics_{feature}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(plot_path)
                plt.close()
                
                self.logger.info(f"Bias metrics plot for {feature} saved to {plot_path}")
                
        except Exception as e:
            self.logger.error(f"Error plotting bias metrics: {str(e)}")
            raise
    
    def _generate_bias_report(self, bias_metrics):
        """Generate detailed bias analysis report with additional checks"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            report = f"""
========== Bias Analysis Report ==========
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Data Quality Checks
-----------------
"""
            # Add data quality information
            for feature, metrics in bias_metrics.items():
                report += f"\n{feature}:\n"
                
                # Check group sizes
                small_groups = [
                    (group, m['size']) 
                    for group, m in metrics.items() 
                    if m['size'] < 10
                ]
                if small_groups:
                    report += "WARNING: Small group sizes detected:\n"
                    for group, size in small_groups:
                        report += f"- {group}: {size} samples\n"
                
                # Check class distribution
                for group, m in metrics.items():
                    if 'class_distribution' in m:
                        report += f"\nClass distribution for {group}:\n"
                        for class_label, prop in m['class_distribution'].items():
                            report += f"- Class {class_label}: {prop:.2%}\n"
            
            # Add the rest of the report
            report += "\nMetrics Analysis\n---------------\n"
            
            for feature, metrics in bias_metrics.items():
                report += f"\nAnalysis for {feature}:\n"
                report += "-" * 40 + "\n"
                
                # Calculate disparities
                accuracies = [m['accuracy'] for m in metrics.values()]
                max_acc_diff = max(accuracies) - min(accuracies)
                
                report += f"Group Sizes:\n"
                for group, group_metrics in metrics.items():
                    report += f"- {group}: {group_metrics['size']} samples\n"
                
                report += f"\nAccuracy Disparity: {max_acc_diff:.4f}\n"
                
                report += "\nPrediction Rates:\n"
                for group, group_metrics in metrics.items():
                    report += f"- {group}: {group_metrics['prediction_rate']:.4f}\n"
                
                if 'disparate_impact' in list(metrics.values())[1]:
                    report += "\nDisparate Impact Ratios:\n"
                    reference_group = list(metrics.keys())[0]
                    for group, group_metrics in metrics.items():
                        if group != reference_group:
                            report += f"- {group} vs {reference_group}: {group_metrics['disparate_impact']:.4f}\n"
                
                report += "\nRecommendations:\n"
                report += self._generate_recommendations(metrics)
                report += "\n"
            
            report += "\n========== End of Report ==========\n"
            
            # Save report
            report_path = self.reports_dir / f'bias_analysis_{timestamp}.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Bias analysis report saved to {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating bias report: {str(e)}")
            raise
    
    def _generate_recommendations(self, group_metrics):
        """Generate recommendations based on bias analysis"""
        recommendations = []
        
        # Check accuracy disparities
        accuracies = [m['accuracy'] for m in group_metrics.values()]
        max_acc_diff = max(accuracies) - min(accuracies)
        
        if max_acc_diff > 0.1:
            recommendations.append(
                "- High accuracy disparity detected. Consider:\n"
                "  * Collecting more data for underperforming groups\n"
                "  * Applying group-specific weights during training"
            )
        
        # Check prediction rate disparities
        pred_rates = [m['prediction_rate'] for m in group_metrics.values()]
        max_pred_diff = max(pred_rates) - min(pred_rates)
        
        if max_pred_diff > 0.1:
            recommendations.append(
                "- Significant prediction rate disparity detected. Consider:\n"
                "  * Adjusting decision thresholds for different groups\n"
                "  * Implementing post-processing bias mitigation"
            )
        
        # Check for small group sizes
        sizes = [m['size'] for m in group_metrics.values()]
        min_size = min(sizes)
        
        if min_size < 100:
            recommendations.append(
                "- Small group size detected. Consider:\n"
                "  * Collecting more data for underrepresented groups\n"
                "  * Using data augmentation techniques"
            )
        
        if not recommendations:
            recommendations.append("- No significant bias issues detected")
        
        return "\n".join(recommendations)

def main():
    """Main function to demonstrate bias analysis"""
    try:
        # Load data and model
        from test_feature_analysis import load_and_prepare_data
        df = load_and_prepare_data()
        
        # Get target column
        target_column = input("\nEnter the target column name from the list above: ")
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset")
        
        # Initialize analyzer
        analyzer = BiasAnalyzer()
        
        # Extract sensitive features
        sensitive_features = df[analyzer.sensitive_features]
        
        # Prepare data (excluding sensitive features from training)
        X = df.drop(analyzer.sensitive_features + [target_column], axis=1)
        y = df[target_column]
        
        # Train a simple model
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Perform bias analysis
        bias_metrics, report = analyzer.analyze_bias(
            model, X_test, y_test,
            sensitive_features.iloc[y_test.index]
        )
        
        print("\nBias Analysis Report:")
        print(report)
        
    except Exception as e:
        print(f"Error in bias analysis: {str(e)}")

if __name__ == "__main__":
    main() 