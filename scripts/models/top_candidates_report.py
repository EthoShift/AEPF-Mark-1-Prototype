from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

class TopCandidateReporter:
    def __init__(self, threshold: float = 0.8):
        """Initialize top candidate reporter"""
        self.threshold = threshold
        self.reports_dir = Path('reports/top_candidates')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_top_candidates_report(
        self, 
        predictions: pd.DataFrame, 
        candidate_data: pd.DataFrame,
        max_candidates: int = 15
    ) -> str:
        """Generate focused report on top candidates"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Merge predictions with candidate data
            analysis_df = predictions.join(candidate_data)
            
            # Filter top candidates
            top_candidates = self._get_top_candidates(
                analysis_df, 
                max_candidates=max_candidates
            )
            
            report = f"""
TOP CANDIDATES INTERVIEW RECOMMENDATIONS
Generated: {timestamp}

SUMMARY
-------
• Selected Candidates: {len(top_candidates)} of {len(predictions)} total
• Minimum Score Threshold: {self.threshold:.2f}
• Selection Criteria: High suitability and performance metrics

RECOMMENDED CANDIDATES
--------------------
{self._format_top_candidates(top_candidates)}

PERFORMANCE METRICS SUMMARY
------------------------
{self._generate_metrics_summary(top_candidates)}

SELECTION NOTES
-------------
• All listed candidates exceed {self.threshold*100:.0f}% suitability threshold
• Recommendations based on objective performance metrics
• Candidates listed in priority order
• Individual strengths highlighted for interview focus

Note: This report focuses on top {max_candidates} highest-scoring candidates 
meeting or exceeding the suitability threshold.
"""
            
            # Save report
            report_path = self.reports_dir / f'top_candidates_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Top candidates report saved to {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating top candidates report: {str(e)}")
            raise
    
    def _get_top_candidates(
        self, 
        df: pd.DataFrame, 
        max_candidates: int
    ) -> pd.DataFrame:
        """Get top candidates meeting threshold"""
        # Filter by threshold and sort
        qualified = df[
            (df['final_score'] >= self.threshold) & 
            (df['suitability_rating'] == 'High')
        ].sort_values('final_score', ascending=False)
        
        return qualified.head(max_candidates)
    
    def _format_top_candidates(self, candidates: pd.DataFrame) -> str:
        """Format top candidates list with strengths"""
        if len(candidates) == 0:
            return "No candidates currently meet the high suitability threshold."
        
        result = ""
        for rank, (_, candidate) in enumerate(candidates.iterrows(), 1):
            result += self._format_candidate_entry(candidate, rank)
        
        return result
    
    def _format_candidate_entry(self, candidate: pd.Series, rank: int) -> str:
        """Format individual candidate entry"""
        strengths = self._identify_key_strengths(candidate)
        metrics = self._get_key_metrics(candidate)
        
        return f"""
{rank}. {candidate['Name']} (Score: {candidate['final_score']:.2f})
   Strengths: {strengths}
   Key Metrics: {metrics}
"""
    
    def _identify_key_strengths(self, candidate: pd.Series) -> str:
        """Identify key strengths based on metrics"""
        strengths = []
        
        # Engagement and Satisfaction
        if candidate['EngagementSurvey'] >= 4.5:
            strengths.append("exceptional engagement")
        elif candidate['EngagementSurvey'] >= 4.0:
            strengths.append("high engagement")
            
        if candidate['EmpSatisfaction'] >= 4.5:
            strengths.append("very high satisfaction")
        elif candidate['EmpSatisfaction'] >= 4.0:
            strengths.append("good satisfaction")
        
        # Project Participation
        if candidate['SpecialProjectsCount'] >= 4:
            strengths.append("outstanding project participation")
        elif candidate['SpecialProjectsCount'] >= 3:
            strengths.append("strong project involvement")
        
        # Attendance and Punctuality
        if candidate['DaysLateLast30'] == 0 and candidate['Absences'] <= 1:
            strengths.append("excellent attendance record")
        elif candidate['DaysLateLast30'] <= 1 and candidate['Absences'] <= 2:
            strengths.append("good attendance")
        
        return ", ".join(strengths).capitalize()
    
    def _get_key_metrics(self, candidate: pd.Series) -> str:
        """Format key performance metrics"""
        return (f"Engagement: {candidate['EngagementSurvey']:.1f}, "
                f"Projects: {candidate['SpecialProjectsCount']}, "
                f"Attendance: {100 - (candidate['Absences']/30*100):.0f}%")
    
    def _generate_metrics_summary(self, candidates: pd.DataFrame) -> str:
        """Generate summary of performance metrics"""
        metrics = {
            'Engagement': candidates['EngagementSurvey'].describe(),
            'Satisfaction': candidates['EmpSatisfaction'].describe(),
            'Projects': candidates['SpecialProjectsCount'].describe(),
            'Attendance': (100 - (candidates['Absences']/30*100)).describe()
        }
        
        summary = "Average Metrics for Selected Candidates:\n"
        for metric, stats in metrics.items():
            summary += f"• {metric:12}: {stats['mean']:6.1f} "
            summary += f"(Range: {stats['min']:.1f} - {stats['max']:.1f})\n"
        
        return summary

def main():
    """Test top candidates report generation"""
    try:
        # Load predictions and candidate data
        from performance_classifier import PerformanceBasedClassifier
        
        # Load and process data
        classifier = PerformanceBasedClassifier()
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        # Prepare data and generate predictions
        X, y = classifier.prepare_data(df)
        classifier.train(X, y)
        predictions = classifier.predict_performance(X)
        
        # Create candidate data
        candidate_data = df[['EngagementSurvey', 'EmpSatisfaction', 
                           'SpecialProjectsCount', 'DaysLateLast30', 'Absences']]
        candidate_data['Name'] = [f"Candidate_{i}" for i in range(len(df))]
        
        # Generate top candidates report
        reporter = TopCandidateReporter(threshold=0.8)
        report = reporter.generate_top_candidates_report(
            predictions=predictions,
            candidate_data=candidate_data,
            max_candidates=15
        )
        
        print("\nTop Candidates Report:")
        print(report)
        
    except Exception as e:
        print(f"Error generating top candidates report: {str(e)}")

if __name__ == "__main__":
    main() 