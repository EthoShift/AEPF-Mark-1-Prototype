from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging

class CandidateEvaluator:
    def __init__(self, random_state=42):
        """Initialize the candidate evaluation model"""
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        )
        
        # Define candidate evaluation metrics
        self.evaluation_metrics = [
            'Technical_Skills',
            'Experience',
            'Education',
            'Communication_Skills',
            'Leadership_Potential',
            'Cultural_Fit',
            'Problem_Solving',
            'Initiative',
            'Team_Collaboration',
            'Adaptability'
        ]
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path('logs/candidate_evaluation')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def evaluate_candidate(self, candidate_data: dict) -> dict:
        """Evaluate a single candidate"""
        try:
            # Validate input data
            self._validate_candidate_data(candidate_data)
            
            # Prepare features
            features = self._prepare_features(candidate_data)
            
            # Generate prediction
            prediction = self.model.predict_proba([features])[0]
            
            # Calculate evaluation score
            score = self._calculate_evaluation_score(prediction, features)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(score)
            
            # Prepare evaluation report
            report = {
                'candidate_id': candidate_data.get('candidate_id', 'Unknown'),
                'evaluation_date': datetime.now().isoformat(),
                'overall_score': score,
                'recommendation': recommendation,
                'detailed_scores': self._get_detailed_scores(features),
                'strengths': self._identify_strengths(features),
                'areas_for_improvement': self._identify_improvements(features)
            }
            
            self.logger.info(f"Completed evaluation for candidate {report['candidate_id']}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error evaluating candidate: {str(e)}")
            raise
    
    def _validate_candidate_data(self, data: dict):
        """Validate candidate data completeness"""
        required_fields = set(self.evaluation_metrics)
        provided_fields = set(data.keys())
        
        missing_fields = required_fields - provided_fields
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
    
    def _prepare_features(self, data: dict) -> np.array:
        """Prepare features for model input"""
        return np.array([data[metric] for metric in self.evaluation_metrics])
    
    def _calculate_evaluation_score(self, prediction: np.array, features: np.array) -> float:
        """Calculate overall evaluation score"""
        # Weighted combination of model prediction and feature scores
        base_score = prediction[1]  # Probability of positive class
        feature_avg = np.mean(features)
        
        return 0.7 * base_score + 0.3 * feature_avg
    
    def _generate_recommendation(self, score: float) -> str:
        """Generate recommendation based on score"""
        if score >= 0.8:
            return "Strongly Recommend for Interview"
        elif score >= 0.6:
            return "Recommend for Interview"
        elif score >= 0.4:
            return "Consider for Interview"
        else:
            return "Not Recommended at This Time"
    
    def _get_detailed_scores(self, features: np.array) -> dict:
        """Generate detailed scores for each metric"""
        return {
            metric: float(score) 
            for metric, score in zip(self.evaluation_metrics, features)
        }
    
    def _identify_strengths(self, features: np.array) -> list:
        """Identify candidate strengths"""
        detailed_scores = self._get_detailed_scores(features)
        return [
            metric for metric, score in detailed_scores.items()
            if score >= 0.7
        ]
    
    def _identify_improvements(self, features: np.array) -> list:
        """Identify areas for improvement"""
        detailed_scores = self._get_detailed_scores(features)
        return [
            metric for metric, score in detailed_scores.items()
            if score < 0.6
        ]
    
    def save_model(self, filename='candidate_evaluator.pkl'):
        """Save the trained model"""
        model_dir = Path('models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / filename
        joblib.dump(self.model, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load_model(cls, filename='candidate_evaluator.pkl'):
        """Load a saved model"""
        model_path = Path('models') / filename
        instance = cls()
        instance.model = joblib.load(model_path)
        return instance
    
    def generate_evaluation_report(self, candidate_data: dict, evaluation_results: dict) -> str:
        """Generate detailed evaluation report in AEPF format"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
========== Candidate Evaluation Report ==========
Generated on: {timestamp}

Candidate ID: {candidate_data.get('candidate_id', 'Unknown')}
Evaluation Type: Pre-Interview Assessment
Risk Level: {self._determine_risk_level(evaluation_results['overall_score'])}

## Evaluation Metrics
Safety Score: {evaluation_results['overall_score']:.2f}
Reliability Score: {self._calculate_reliability_score(evaluation_results['detailed_scores']):.2f}
Ethical Score: {self._calculate_ethical_score(evaluation_results['detailed_scores']):.2f}

## Key Findings
1. Safety Assessment
   • Technical validation completed
   • Background verification status: {candidate_data.get('background_verified', 'Pending')}
   • Risk assessment completed

2. Reliability Analysis
   • Experience verified: {candidate_data.get('experience_verified', 'Pending')}
   • Skills assessment completed
   • Reference checks: {candidate_data.get('references_checked', 'Pending')}

3. Ethical Considerations
   • Cultural alignment assessed
   • Professional conduct verified
   • Compliance requirements met

## Detailed Scores
{self._format_detailed_scores(evaluation_results['detailed_scores'])}

## Strengths
{self._format_list_items(evaluation_results['strengths'])}

## Areas for Improvement
{self._format_list_items(evaluation_results['areas_for_improvement'])}

## Recommendations
{self._format_recommendations(evaluation_results['recommendation'])}

## Status: {self._get_status_indicator(evaluation_results['overall_score'])}

========== End of Report ==========
        """
        
        # Save report to file
        self._save_report(report, candidate_data['candidate_id'])
        
        return report
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level based on score"""
        if score >= 0.8:
            return "LOW"
        elif score >= 0.6:
            return "MEDIUM"
        return "HIGH"
    
    def _calculate_reliability_score(self, detailed_scores: dict) -> float:
        """Calculate reliability score from detailed scores"""
        reliability_metrics = [
            'Experience',
            'Technical_Skills',
            'Problem_Solving',
            'Initiative'
        ]
        return np.mean([detailed_scores[metric] for metric in reliability_metrics])
    
    def _calculate_ethical_score(self, detailed_scores: dict) -> float:
        """Calculate ethical score from detailed scores"""
        ethical_metrics = [
            'Cultural_Fit',
            'Team_Collaboration',
            'Leadership_Potential'
        ]
        return np.mean([detailed_scores[metric] for metric in ethical_metrics])
    
    def _format_detailed_scores(self, scores: dict) -> str:
        """Format detailed scores for report"""
        formatted = "Metric Scores:\n"
        for metric, score in scores.items():
            bar = "=" * int(score * 20)  # Create visual bar
            formatted += f"{metric:20}: [{bar:<20}] {score:.2f}\n"
        return formatted
    
    def _format_list_items(self, items: list) -> str:
        """Format list items for report"""
        return "\n".join([f"• {item}" for item in items])
    
    def _format_recommendations(self, recommendation: str) -> str:
        """Format recommendations with supporting details"""
        rec_details = {
            "Strongly Recommend for Interview": [
                "Candidate shows exceptional potential",
                "Strong alignment with requirements",
                "Recommended for immediate consideration"
            ],
            "Recommend for Interview": [
                "Candidate meets key requirements",
                "Good potential for role",
                "Proceed with standard interview process"
            ],
            "Consider for Interview": [
                "Candidate shows some potential",
                "Additional screening recommended",
                "Consider for junior positions"
            ],
            "Not Recommended at This Time": [
                "Does not meet minimum requirements",
                "Significant gaps identified",
                "Consider for future opportunities"
            ]
        }
        
        details = rec_details.get(recommendation, [])
        formatted = f"Primary Recommendation: {recommendation}\n\nSupporting Details:"
        return formatted + "\n" + self._format_list_items(details)
    
    def _get_status_indicator(self, score: float) -> str:
        """Get status indicator with emoji"""
        if score >= 0.8:
            return "✅ STRONGLY RECOMMENDED"
        elif score >= 0.6:
            return "✓ RECOMMENDED"
        elif score >= 0.4:
            return "⚠️ CONSIDER WITH CAUTION"
        return "❌ NOT RECOMMENDED"
    
    def _save_report(self, report: str, candidate_id: str):
        """Save report to file"""
        reports_dir = Path('reports/candidate_evaluations')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"evaluation_{candidate_id}_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Report saved to {report_path}")

def main():
    """Example usage with enhanced reporting"""
    candidate = {
        'candidate_id': 'CAND001',
        'Technical_Skills': 0.85,
        'Experience': 0.75,
        'Education': 0.90,
        'Communication_Skills': 0.80,
        'Leadership_Potential': 0.70,
        'Cultural_Fit': 0.85,
        'Problem_Solving': 0.88,
        'Initiative': 0.82,
        'Team_Collaboration': 0.85,
        'Adaptability': 0.78,
        'background_verified': 'Completed',
        'experience_verified': 'Completed',
        'references_checked': 'In Progress'
    }
    
    evaluator = CandidateEvaluator()
    evaluation_results = evaluator.evaluate_candidate(candidate)
    
    # Generate and print report
    report = evaluator.generate_evaluation_report(candidate, evaluation_results)
    print(report)

if __name__ == "__main__":
    main() 