from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

class Top15Reporter:
    def __init__(self):
        """Initialize top 15 candidate reporter"""
        self.reports_dir = Path('reports/top_15')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_top_15_report(
        self, 
        predictions: pd.DataFrame, 
        candidate_data: pd.DataFrame
    ) -> str:
        """Generate focused report on top 15 candidates"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Merge predictions with candidate data
            analysis_df = predictions.join(candidate_data)
            
            # Get top 15 candidates
            top_15 = self._get_top_15_candidates(analysis_df)
            
            # Calculate score ranges for context
            score_ranges = self._calculate_score_ranges(top_15)
            
            report = f"""
TOP 15 CANDIDATES FOR INTERVIEW
Generated: {timestamp}

SELECTION SUMMARY
---------------
• Total Candidates Evaluated: {len(predictions)}
• Score Range: {score_ranges['min_score']:.2f} to {score_ranges['max_score']:.2f}
• Selection Criteria: Overall performance and suitability scores

PRIORITIZED RECOMMENDATIONS
------------------------
{self._format_candidate_list(top_15)}

PERFORMANCE HIGHLIGHTS
-------------------
{self._generate_highlights(top_15)}

INTERVIEW FOCUS AREAS
------------------
{self._generate_focus_areas(top_15)}

Note: Candidates are listed in strict priority order based on comprehensive 
performance evaluation. Each candidate's standout metrics are highlighted 
for targeted interview discussions.
"""
            
            # Save report
            report_path = self.reports_dir / f'top_15_report_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Top 15 report saved to {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating top 15 report: {str(e)}")
            raise
    
    def _get_top_15_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get top 15 candidates by final score"""
        return df.nlargest(15, 'final_score')
    
    def _calculate_score_ranges(self, top_15: pd.DataFrame) -> Dict[str, float]:
        """Calculate score ranges for top 15"""
        return {
            'min_score': top_15['final_score'].min(),
            'max_score': top_15['final_score'].max(),
            'avg_score': top_15['final_score'].mean()
        }
    
    def _format_candidate_list(self, candidates: pd.DataFrame) -> str:
        """Format prioritized candidate list"""
        result = ""
        for rank, (_, candidate) in enumerate(candidates.iterrows(), 1):
            result += self._format_candidate_entry(candidate, rank)
        return result
    
    def _format_candidate_entry(self, candidate: pd.Series, rank: int) -> str:
        """Format individual candidate entry with standout metrics only"""
        standout_metrics = self._get_standout_metrics(candidate)
        strengths = self._identify_key_strengths(candidate)
        
        entry = f"""
{rank}. {candidate['Name']} (Score: {candidate['final_score']:.2f})
   Primary Strengths: {strengths}"""
        
        if standout_metrics:
            entry += f"\n   Standout Metrics: {standout_metrics}"
        
        return entry
    
    def _get_standout_metrics(self, candidate: pd.Series) -> str:
        """Get only standout metrics for candidate"""
        standouts = []
        
        # Check each metric for standout performance
        if candidate['EngagementSurvey'] >= 4.5:
            standouts.append(f"Engagement: {candidate['EngagementSurvey']:.1f}")
        
        if candidate['EmpSatisfaction'] >= 4.5:
            standouts.append(f"Satisfaction: {candidate['EmpSatisfaction']:.1f}")
        
        if candidate['SpecialProjectsCount'] >= 4:
            standouts.append(f"Projects: {candidate['SpecialProjectsCount']}")
        
        if candidate['DaysLateLast30'] == 0:
            standouts.append("Perfect Attendance")
        
        if candidate['Absences'] <= 1:
            standouts.append("Minimal Absences")
        
        return ", ".join(standouts) if standouts else "Consistent across metrics"
    
    def _identify_key_strengths(self, candidate: pd.Series) -> str:
        """Identify key strengths focusing on outstanding areas"""
        strengths = []
        
        # Only include truly outstanding metrics
        if candidate['EngagementSurvey'] >= 4.5:
            strengths.append("exceptional engagement")
        
        if candidate['SpecialProjectsCount'] >= 4:
            strengths.append("project leader")
        
        if candidate['DaysLateLast30'] == 0 and candidate['Absences'] <= 1:
            strengths.append("perfect attendance record")
        
        if not strengths:
            strengths.append("strong overall performer")
        
        return ", ".join(strengths).capitalize()
    
    def _generate_highlights(self, candidates: pd.DataFrame) -> str:
        """Generate key performance highlights"""
        highlights = [
            f"• {len(candidates[candidates['EngagementSurvey'] >= 4.5])} candidates show exceptional engagement",
            f"• {len(candidates[candidates['SpecialProjectsCount'] >= 4])} candidates lead in project participation",
            f"• {len(candidates[candidates['DaysLateLast30'] == 0])} candidates have perfect attendance"
        ]
        
        return "\n".join(highlights)
    
    def _generate_focus_areas(self, candidates: pd.DataFrame) -> str:
        """Generate suggested interview focus areas"""
        # Identify common strengths among top candidates
        common_strengths = {
            'Engagement': len(candidates[candidates['EngagementSurvey'] >= 4.0]),
            'Projects': len(candidates[candidates['SpecialProjectsCount'] >= 3]),
            'Attendance': len(candidates[candidates['DaysLateLast30'] <= 1])
        }
        
        # Generate focus areas based on common strengths
        focus_areas = [
            "Suggested areas for interview discussions:",
            "• Project experience and contributions",
            "• Team engagement and collaboration",
            "• Professional development goals",
            "• Attendance and reliability track record"
        ]
        
        return "\n".join(focus_areas)

def main():
    """Test top 15 report generation"""
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
        
        # Generate top 15 report
        reporter = Top15Reporter()
        report = reporter.generate_top_15_report(
            predictions=predictions,
            candidate_data=candidate_data
        )
        
        print("\nTop 15 Candidates Report:")
        print(report)
        
    except Exception as e:
        print(f"Error generating top 15 report: {str(e)}")

if __name__ == "__main__":
    main() 