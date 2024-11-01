from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class SuitabilityReportGenerator:
    def __init__(self):
        """Initialize report generator"""
        self.reports_dir = Path('reports/suitability')
        self.plots_dir = self.reports_dir / 'plots'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, predictions: pd.DataFrame, performance_metrics: Dict) -> str:
        """Generate structured suitability report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report = f"""
CANDIDATE SUITABILITY ASSESSMENT REPORT
Generated: {timestamp}

OBJECTIVE
---------
This baseline classification report evaluates candidate suitability using objective 
performance metrics including engagement levels, satisfaction scores, project 
participation, attendance, and punctuality. The assessment aims to identify candidates 
with the strongest performance indicators for interview recommendations.

SUITABILITY SUMMARY
------------------
Total Candidates Evaluated: {len(predictions)}

Suitability Categories:
• High Suitability: {len(predictions[predictions['suitability_rating'] == 'High'])} candidates
  - Consistently strong performance across all metrics
  - High engagement and satisfaction levels
  - Excellent project participation and attendance

• Medium Suitability: {len(predictions[predictions['suitability_rating'] == 'Medium'])} candidates
  - Good overall performance with some variation
  - Moderate to high engagement levels
  - Satisfactory project participation and attendance

• Low Suitability: {len(predictions[predictions['suitability_rating'] == 'Low'])} candidates
  - Inconsistent performance metrics
  - Lower engagement or satisfaction scores
  - Limited project participation or attendance concerns

INTERVIEW RECOMMENDATIONS
------------------------
1. Recommended for Immediate Interview ({self._get_high_priority_count(predictions)} candidates):
{self._format_high_priority_candidates(predictions)}

2. Consider with Follow-Up ({self._get_medium_priority_count(predictions)} candidates):
{self._format_medium_priority_candidates(predictions)}

3. Not Recommended at This Time ({self._get_low_priority_count(predictions)} candidates):
   * Candidates requiring significant improvement in performance metrics
   * Recommended for reassessment after performance improvement

PERFORMANCE STATISTICS
--------------------
Model Score (Algorithm-based assessment):
{self._format_statistics(predictions['model_score'])}

Performance Score (Metric-based evaluation):
{self._format_statistics(predictions['performance_score'])}

Final Score (Combined assessment):
{self._format_statistics(predictions['final_score'])}

KEY PERFORMANCE INDICATORS
------------------------
{self._format_performance_metrics(performance_metrics)}

NOTES AND CONSIDERATIONS
----------------------
• This assessment is based on objective performance data and should be used as a 
  supporting tool for interview decisions.
• Individual circumstances and qualitative factors should be considered alongside 
  these recommendations.
• Regular reassessment is recommended as performance metrics are updated.

VISUALIZATION
------------
{self._generate_visualizations(predictions)}

Report generated by Performance-Based Classification System
For internal use in interview candidate selection process
"""
        
        # Save report
        report_path = self.reports_dir / f'suitability_report_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report
    
    def _format_statistics(self, series: pd.Series) -> str:
        """Format statistical summary"""
        stats = series.describe()
        return f"""
   Mean Score: {stats['mean']:.3f}
   Std Dev:   {stats['std']:.3f}
   Minimum:   {stats['min']:.3f}
   Maximum:   {stats['max']:.3f}"""
    
    def _get_high_priority_count(self, predictions: pd.DataFrame) -> int:
        """Get count of high priority candidates"""
        return len(predictions[
            (predictions['suitability_rating'] == 'High') & 
            (predictions['final_score'] >= 0.8)
        ])
    
    def _get_medium_priority_count(self, predictions: pd.DataFrame) -> int:
        """Get count of medium priority candidates"""
        return len(predictions[
            (predictions['suitability_rating'] == 'Medium') & 
            (predictions['final_score'] >= 0.6)
        ])
    
    def _get_low_priority_count(self, predictions: pd.DataFrame) -> int:
        """Get count of low priority candidates"""
        return len(predictions[predictions['suitability_rating'] == 'Low'])
    
    def _format_high_priority_candidates(self, predictions: pd.DataFrame) -> str:
        """Format high priority candidate recommendations"""
        high_priority = predictions[
            (predictions['suitability_rating'] == 'High') & 
            (predictions['final_score'] >= 0.8)
        ].sort_values('final_score', ascending=False)
        
        if len(high_priority) == 0:
            return "   * No candidates currently meet high priority criteria"
        
        result = ""
        for idx, row in high_priority.head(5).iterrows():
            result += f"""   * Candidate {idx}: 
     - Final Score: {row['final_score']:.3f}
     - Strong performance across all metrics
     - Recommended for immediate consideration\n"""
        
        if len(high_priority) > 5:
            result += f"   * Plus {len(high_priority) - 5} additional qualified candidates\n"
        
        return result
    
    def _format_medium_priority_candidates(self, predictions: pd.DataFrame) -> str:
        """Format medium priority candidate recommendations"""
        medium_priority = predictions[
            (predictions['suitability_rating'] == 'Medium') & 
            (predictions['final_score'] >= 0.6)
        ].sort_values('final_score', ascending=False)
        
        if len(medium_priority) == 0:
            return "   * No candidates currently in medium priority category"
        
        result = ""
        for idx, row in medium_priority.head(3).iterrows():
            result += f"""   * Candidate {idx}:
     - Final Score: {row['final_score']:.3f}
     - Shows potential with room for development
     - Recommended for secondary review\n"""
        
        if len(medium_priority) > 3:
            result += f"   * Plus {len(medium_priority) - 3} additional candidates for consideration\n"
        
        return result
    
    def _format_performance_metrics(self, metrics: Dict) -> str:
        """Format performance metrics summary"""
        result = "Most Influential Factors:\n"
        
        sorted_metrics = sorted(
            metrics.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )
        
        for metric, config in sorted_metrics:
            result += f"""   * {metric}:
     - Weight: {config['weight']:.2f}
     - Direction: {config['optimal_direction'].capitalize()}\n"""
        
        return result
    
    def _generate_visualizations(self, predictions: pd.DataFrame) -> str:
        """Generate and save visualization plots"""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Score distribution plot
        sns.histplot(data=predictions, x='final_score', hue='suitability_rating', 
                    multiple="stack", ax=ax1)
        ax1.set_title('Distribution of Final Scores by Suitability Rating')
        ax1.set_xlabel('Final Score')
        ax1.set_ylabel('Count')
        
        # Suitability rating counts
        sns.countplot(data=predictions, x='suitability_rating', ax=ax2)
        ax2.set_title('Count of Candidates by Suitability Rating')
        ax2.set_xlabel('Suitability Rating')
        ax2.set_ylabel('Count')
        
        # Save plot
        plot_path = self.plots_dir / f'suitability_visualization_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        return f"Visualizations saved to: {plot_path}"

def main():
    """Test report generation"""
    try:
        # Load predictions from performance classifier
        from performance_classifier import PerformanceBasedClassifier
        
        # Load data and generate predictions
        classifier = PerformanceBasedClassifier()
        predictions = pd.read_csv('path_to_predictions.csv')  # You'll need to save predictions first
        
        # Generate report
        report_gen = SuitabilityReportGenerator()
        report = report_gen.generate_report(
            predictions=predictions,
            performance_metrics=classifier.performance_metrics
        )
        
        print("Report generated successfully!")
        print(report)
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")

if __name__ == "__main__":
    main() 