from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class SummarizedReportGenerator:
    def __init__(self):
        """Initialize summary report generator"""
        self.reports_dir = Path('reports/summaries')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_summary(self, predictions: pd.DataFrame, candidate_data: pd.DataFrame) -> str:
        """Generate concise summary report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Merge predictions with candidate data
        analysis_df = predictions.join(candidate_data)
        
        summary = f"""
CANDIDATE SUITABILITY SUMMARY REPORT
Generated: {timestamp}

OVERVIEW
--------
This report identifies and prioritizes candidates for interviews based on objective 
performance metrics and suitability scores. The analysis aims to streamline the 
interview process by highlighting candidates with strong performance indicators.

SUITABILITY BREAKDOWN
-------------------
Total Candidates Analyzed: {len(predictions)}

• High Suitability:   {self._count_category(predictions, 'High')} candidates
  - Priority candidates for immediate interview consideration
  - Consistently strong performance across key metrics
  
• Medium Suitability: {self._count_category(predictions, 'Medium')} candidates
  - Secondary candidates for consideration
  - Good performance with specific areas for review
  
• Low Suitability:    {self._count_category(predictions, 'Low')} candidates
  - Not recommended for current interview round
  - Performance metrics below target thresholds

RECOMMENDED CANDIDATES
--------------------
Immediate Interview Recommendations:
{self._format_high_priority_candidates(analysis_df)}

Secondary Considerations:
{self._format_medium_priority_candidates(analysis_df)}

STATISTICAL SUMMARY
-----------------
{self._generate_statistics_table(predictions)}

KEY INSIGHTS
-----------
• {self._get_key_insights(predictions)}

CONCLUSION
---------
This summary identifies {self._count_category(predictions, 'High')} high-priority 
candidates recommended for immediate interviews, with an additional 
{self._count_category(predictions, 'Medium')} candidates suitable for secondary 
consideration. Recommendations are based on objective performance metrics and 
model-driven suitability assessments.

Note: Individual circumstances and qualitative factors should complement these 
data-driven recommendations in the final selection process.
"""
        
        # Save summary
        summary_path = self.reports_dir / f'summary_report_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        return summary
    
    def _count_category(self, predictions: pd.DataFrame, category: str) -> int:
        """Count candidates in specific category"""
        return len(predictions[predictions['suitability_rating'] == category])
    
    def _format_high_priority_candidates(self, df: pd.DataFrame) -> str:
        """Format high priority candidate list"""
        high_priority = df[
            (df['suitability_rating'] == 'High') & 
            (df['final_score'] >= 0.8)
        ].sort_values('final_score', ascending=False)
        
        if len(high_priority) == 0:
            return "No candidates currently meet high priority criteria.\n"
        
        result = ""
        for _, row in high_priority.iterrows():
            result += self._format_candidate_entry(row)
        
        return result
    
    def _format_medium_priority_candidates(self, df: pd.DataFrame) -> str:
        """Format medium priority candidate list"""
        medium_priority = df[
            (df['suitability_rating'] == 'Medium') & 
            (df['final_score'] >= 0.6)
        ].sort_values('final_score', ascending=False)
        
        if len(medium_priority) == 0:
            return "No candidates currently in medium priority category.\n"
        
        result = ""
        for _, row in medium_priority.head(5).iterrows():
            result += self._format_candidate_entry(row)
        
        if len(medium_priority) > 5:
            result += f"\n* Plus {len(medium_priority) - 5} additional candidates available for review\n"
        
        return result
    
    def _format_candidate_entry(self, row: pd.Series) -> str:
        """Format individual candidate entry"""
        strengths = self._identify_candidate_strengths(row)
        return f"""• {row['Name']} (Score: {row['final_score']:.2f})
  - {strengths}
"""
    
    def _identify_candidate_strengths(self, row: pd.Series) -> str:
        """Generate strength description based on metrics"""
        strengths = []
        
        if row.get('EngagementSurvey', 0) >= 4.0:
            strengths.append("high engagement")
        if row.get('EmpSatisfaction', 0) >= 4.0:
            strengths.append("strong satisfaction")
        if row.get('SpecialProjectsCount', 0) >= 3:
            strengths.append("exceeds project goals")
        if row.get('DaysLateLast30', 0) <= 1:
            strengths.append("excellent attendance")
        if row.get('Absences', 0) <= 2:
            strengths.append("consistent presence")
        
        if not strengths:
            strengths = ["meets performance expectations"]
        
        return ", ".join(strengths).capitalize()
    
    def _generate_statistics_table(self, predictions: pd.DataFrame) -> str:
        """Generate formatted statistics table"""
        stats_table = """
Score Type        |  Mean  |  Std Dev  |   Min   |   Max   
-----------------|--------|-----------|---------|--------"""
        
        for score_type in ['model_score', 'performance_score', 'final_score']:
            stats = predictions[score_type].describe()
            stats_table += f"\n{score_type:15} | {stats['mean']:6.3f} | {stats['std']:9.3f} | {stats['min']:7.3f} | {stats['max']:7.3f}"
        
        return stats_table
    
    def _get_key_insights(self, predictions: pd.DataFrame) -> str:
        """Generate key insights from the data"""
        insights = []
        
        # Distribution insight
        high_ratio = len(predictions[predictions['suitability_rating'] == 'High']) / len(predictions)
        if high_ratio >= 0.2:
            insights.append(f"{(high_ratio * 100):.1f}% of candidates show high suitability")
        
        # Score spread insight
        score_range = predictions['final_score'].max() - predictions['final_score'].min()
        if score_range > 0.5:
            insights.append("Wide range of candidate performance observed")
        else:
            insights.append("Consistent performance levels across candidates")
        
        # High performers insight
        top_performers = len(predictions[predictions['final_score'] >= 0.8])
        if top_performers > 0:
            insights.append(f"{top_performers} candidates demonstrate exceptional performance")
        
        return " • ".join(insights)

def main():
    """Test summary report generation"""
    try:
        # Load predictions and candidate data
        predictions = pd.read_csv('path_to_predictions.csv')  # You'll need to save predictions first
        candidate_data = pd.read_csv('path_to_candidate_data.csv')  # You'll need candidate data
        
        # Generate summary
        report_gen = SummarizedReportGenerator()
        summary = report_gen.generate_summary(predictions, candidate_data)
        
        print("Summary report generated successfully!")
        print(summary)
        
    except Exception as e:
        print(f"Error generating summary report: {str(e)}")

if __name__ == "__main__":
    main() 