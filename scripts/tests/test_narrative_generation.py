import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from scripts.context_engine import ContextEngine
from scripts.ethical_governor import EthicalGovernor
from scripts.decision_analysis.probability_scorer import ProbabilityScorer, ProbabilityBand
from scripts.decision_analysis.narrative_generator import OutcomeNarrative
from scripts.report_templates import TestReport, ReportManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_narrative_generation():
    """Test narrative generation for different decision scenarios"""
    
    # Initialize components
    context_engine = ContextEngine()
    ethical_governor = EthicalGovernor()
    probability_scorer = ProbabilityScorer()
    narrative_generator = OutcomeNarrative()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Privacy-Focused Decision",
            "action": "implement_data_collection",
            "context": {
                "privacy_level": "high",
                "region": "EU",
                "compliance_requirements": ["GDPR"],
                "risk_level": "high",
                "prism_scores": {
                    "human": 0.7,
                    "privacy": 0.8,
                    "innovation": -0.2
                }
            }
        },
        {
            "name": "Innovation Initiative",
            "action": "deploy_ai_system",
            "context": {
                "innovation_impact": "high",
                "region": "US-CA",
                "compliance_requirements": ["AI Transparency"],
                "risk_level": "moderate",
                "prism_scores": {
                    "human": 0.5,
                    "innovation": 0.9,
                    "eco": 0.3
                }
            }
        },
        {
            "name": "Environmental Impact",
            "action": "optimize_cloud_resources",
            "context": {
                "environmental_priority": "high",
                "region": "DE",
                "compliance_requirements": ["Green IT"],
                "risk_level": "low",
                "prism_scores": {
                    "eco": 0.8,
                    "innovation": 0.4,
                    "human": 0.6
                }
            }
        }
    ]
    
    # Initialize report
    report = TestReport.create(
        test_name="narrative_generation_test",
        components=["OutcomeNarrative", "ProbabilityScorer"]
    )
    
    results = []
    for scenario in test_scenarios:
        logger.info(f"\nTesting narrative generation for: {scenario['name']}")
        
        # Get probability score
        probability_score = probability_scorer.calculate_probability(
            scenario['context']['prism_scores'],
            scenario['context'],
            {'compliance': scenario['context']['compliance_requirements']},
            'high_impact'
        )
        
        # Generate narrative
        narrative = narrative_generator.generate_narrative(
            probability_score.adjusted_score,
            probability_score.band,
            scenario['context']['prism_scores'],
            scenario['context'],
            probability_score.confidence_level
        )
        
        logger.info("\nGenerated Narrative:")
        logger.info(narrative)
        
        # Record results
        results.append({
            "scenario": scenario['name'],
            "probability_score": probability_score,
            "narrative": narrative,
            "context": scenario['context']
        })
        
        # Add to report
        report.results[scenario['name']] = {
            "status": "success",
            "probability_band": probability_score.band.value,
            "adjusted_score": probability_score.adjusted_score,
            "narrative": narrative
        }
    
    # Add performance metrics
    report.performance_metrics = {
        "scenarios_tested": len(results),
        "average_score": sum(r["probability_score"].adjusted_score for r in results) / len(results),
        "average_confidence": sum(r["probability_score"].confidence_level for r in results) / len(results)
    }
    
    report.status = "completed"
    
    # Save report
    report_manager = ReportManager()
    report.test_id = report_manager.get_next_id()
    
    try:
        json_path, text_path = report_manager.save_report(report)
        logger.info(f"\nReport saved to: {text_path}")
        
        # Display report content
        with open(text_path, 'r', encoding='cp1252', errors='ignore') as f:
            print("\nTest Results:")
            print("=" * 80)
            print(f.read())
            print("=" * 80)
            
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
    
    return report

if __name__ == "__main__":
    test_narrative_generation() 