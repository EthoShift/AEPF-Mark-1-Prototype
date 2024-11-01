from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from scripts.models.percentile_classifier import PercentileClassifier

class Top15Selector:
    def __init__(self):
        """Initialize top 15 candidate selector"""
        self.setup_logging()
        self.classifier = PercentileClassifier()
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs/top_15_selector')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def select_top_candidates(self, df: pd.DataFrame) -> str:
        """Select and report on top 15 candidates"""
        try:
            self.logger.info("Starting top 15 candidate selection...")
            
            # Classify all candidates
            results = self.classifier.classify_candidates(df)
            
            # Select top 15 from high suitability category
            top_candidates = self._get_top_15(results, df)
            
            # Generate report
            report = self._generate_selection_report(top_candidates)
            
            # Save report
            self._save_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in candidate selection: {str(e)}")
            raise
    
    def _get_top_15(self, results: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Get top 15 candidates with their metrics"""
        try:
            # Filter high suitability candidates
            high_suitability = results[results['suitability_rating'] == 'High'].copy()
            
            # Sort by composite score
            high_suitability = high_suitability.sort_values('composite_score', ascending=False)
            
            # Select top 15
            top_15 = high_suitability.head(15)
            
            # Add original metrics
            for col in original_df.columns:
                top_15[f'original_{col}'] = original_df.loc[top_15.index, col]
            
            # Add explanatory notes
            top_15['recommendation_reason'] = top_15.apply(
                self._generate_recommendation_reason, axis=1
            )
            
            return top_15
            
        except Exception as e:
            self.logger.error(f"Error selecting top 15 candidates: {str(e)}")
            raise
    
    def _generate_recommendation_reason(self, row: pd.Series) -> str:
        """Generate concise recommendation reason"""
        try:
            reasons = []
            
            # Check engagement
            if row['original_EngagementSurvey'] >= 4.5:
                reasons.append("exceptional engagement")
            elif row['original_EngagementSurvey'] >= 4.0:
                reasons.append("strong engagement")
            
            # Check satisfaction
            if row['original_EmpSatisfaction'] >= 4.5:
                reasons.append("high satisfaction")
            
            # Check projects
            if row['original_SpecialProjectsCount'] >= 4:
                reasons.append("project leader")
            elif row['original_SpecialProjectsCount'] >= 3:
                reasons.append("active project participant")
            
            # Check attendance
            if (row['original_DaysLateLast30'] == 0 and 
                row['original_Absences'] <= 1):
                reasons.append("perfect attendance")
            
            # Select primary reason (most impressive)
            if reasons:
                return reasons[0].capitalize()
            return "Consistently strong performer"
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation reason: {str(e)}")
            return "Strong overall metrics"
    
    def _generate_selection_report(self, top_candidates: pd.DataFrame) -> str:
        """Generate detailed selection report"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            report = f"""
TOP 15 CANDIDATES RECOMMENDED FOR INTERVIEW
Generated: {timestamp}

SELECTION CRITERIA
----------------
• Candidates selected from High Suitability category
• Sorted by overall performance score
• Individual strengths highlighted for each candidate

RECOMMENDED CANDIDATES
--------------------
"""
            # Add each candidate's details
            for rank, (idx, candidate) in enumerate(top_candidates.iterrows(), 1):
                report += f"""
{rank}. Candidate ID: {idx}
   Score: {candidate['composite_score']:.3f} ({candidate['percentile_rank']:.1%} percentile)
   Primary Strength: {candidate['recommendation_reason']}
   Key Metrics:
   - Engagement: {candidate['original_EngagementSurvey']:.1f}
   - Satisfaction: {candidate['original_EmpSatisfaction']:.1f}
   - Projects: {candidate['original_SpecialProjectsCount']}
   - Attendance: {100 - (candidate['original_Absences']/30*100):.0f}%
"""
            
            # Add summary statistics
            report += f"""
SELECTION SUMMARY
---------------
• Score Range: {top_candidates['composite_score'].min():.3f} to {top_candidates['composite_score'].max():.3f}
• Average Engagement: {top_candidates['original_EngagementSurvey'].mean():.1f}
• Average Projects: {top_candidates['original_SpecialProjectsCount'].mean():.1f}
• Perfect Attendance: {len(top_candidates[top_candidates['original_DaysLateLast30'] == 0])} candidates

INTERVIEW RECOMMENDATIONS
----------------------
1. Focus on demonstrated strengths in:
   • Project leadership and participation
   • Team engagement and collaboration
   • Professional reliability and commitment

2. Explore specific achievements in:
   • Special project contributions
   • Team involvement
   • Professional development initiatives

Note: All selected candidates represent top performers with proven track records
in key performance areas. Individual strengths should guide interview focus.
"""
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating selection report: {str(e)}")
            raise
    
    def _save_report(self, report: str):
        """Save selection report"""
        try:
            report_dir = Path('reports/top_15_selection')
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = report_dir / f'top_15_selection_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
            
            with open(report_path, 'w') as f:
                f.write(report)
                
            self.logger.info(f"Selection report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
            raise

def main():
    """Test top 15 selection"""
    try:
        # Load data
        data_path = Path("data/datasets/hr_analytics")
        latest_file = max(data_path.glob("*HRDataset*.csv"), key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        # Initialize selector
        selector = Top15Selector()
        
        # Generate selection report
        report = selector.select_top_candidates(df)
        
        # Display report
        print(report)
        
    except Exception as e:
        print(f"Error in top 15 selection: {str(e)}")

if __name__ == "__main__":
    main() 