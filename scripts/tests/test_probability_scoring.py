import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from scripts.context_engine import ContextEngine
from scripts.ethical_governor import EthicalGovernor
from scripts.decision_analysis.probability_scorer import ProbabilityScorer, ProbabilityBand
from scripts.report_templates import TestReport, ReportManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_probability_scoring():
    """Test probability scoring across different scenarios"""
    
    # Ensure report directory exists
    report_dir = Path(__file__).parent.parent.parent / "reports" / "probability_tests"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    context_engine = ContextEngine()
    ethical_governor = EthicalGovernor()
    probability_scorer = ProbabilityScorer()
    
    # Test scenarios with varying ethical implications
    test_scenarios = [
        {
            "name": "High Privacy Impact",
            "action": "collect_user_data",
            "context": {
                "privacy_level": "high",
                "data_sensitivity": "personal",
                "region": "EU",
                "compliance_requirements": ["GDPR"],
                "privacy_emphasis": "very_high",
                "risk_level": "high"
            },
            "expected_band": ProbabilityBand.LOW
        },
        {
            "name": "Innovation Focus",
            "action": "implement_ai_optimization",
            "context": {
                "privacy_level": "moderate",
                "innovation_impact": "high",
                "region": "US-CA",
                "compliance_requirements": ["AI Transparency"],
                "innovation_tolerance": "progressive",
                "risk_level": "moderate"
            },
            "expected_band": ProbabilityBand.HIGH
        },
        {
            "name": "Environmental Impact",
            "action": "deploy_cloud_infrastructure",
            "context": {
                "energy_efficiency": "high",
                "resource_impact": "moderate",
                "region": "DE",
                "compliance_requirements": ["Green IT Standards"],
                "environmental_priority": "high",
                "risk_level": "low"
            },
            "expected_band": ProbabilityBand.MEDIUM
        }
    ]
    
    results = []
    for scenario in test_scenarios:
        logger.info(f"\nTesting scenario: {scenario['name']}")
        
        # Get ethical decision
        decision = ethical_governor.evaluate_action(
            scenario['action'],
            scenario['context']
        )
        
        # Calculate probability scores
        probability_score = probability_scorer.calculate_probability(
            decision.prism_scores,
            scenario['context'],
            {'compliance': scenario['context']['compliance_requirements']},
            decision.category.value
        )
        
        # Record results
        results.append({
            "scenario": scenario['name'],
            "action": scenario['action'],
            "ethical_decision": decision,
            "probability_score": probability_score,
            "expected_band": scenario['expected_band'],
            "matches_expected": probability_score.band == scenario['expected_band']
        })
        
        # Log results
        logger.info(f"Probability Band: {probability_score.band.value}")
        logger.info(f"Raw Score: {probability_score.raw_score:.2f}")
        logger.info(f"Adjusted Score: {probability_score.adjusted_score:.2f}")
        logger.info(f"Confidence Level: {probability_score.confidence_level:.2f}")
        
        if probability_score.band == scenario['expected_band']:
            logger.info("[PASS] Matches expected probability band")
        else:
            logger.warning(f"[FAIL] Expected {scenario['expected_band'].value}, got {probability_score.band.value}")
    
    # Generate report
    report = generate_probability_report(results)
    
    # Save report
    report_manager = ReportManager()
    report.test_id = report_manager.get_next_id()
    
    try:
        json_path, text_path = report_manager.save_report(report)
        logger.info(f"\nReport saved to: {text_path}")
        
        # Display report content using Windows default encoding
        with open(text_path, 'r', encoding='cp1252', errors='ignore') as f:
            print("\nTest Results:")
            print("=" * 80)
            print(f.read())
            print("=" * 80)
            
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
    
    return report

def generate_probability_report(results: List[Dict]) -> TestReport:
    """Generate test report for probability scoring"""
    report = TestReport.create(
        test_name="probability_scoring_test",
        components=["EthicalGovernor", "ProbabilityScorer"]
    )
    
    # Add results for each scenario
    for result in results:
        scenario_name = result['scenario']
        status_marker = "[PASS]" if result['matches_expected'] else "[FAIL]"
        report.results[scenario_name] = {
            "status": "success" if result['matches_expected'] else "warning",
            "action": result['action'],
            "ethical_decision": str(result['ethical_decision'].recommendation),
            "probability_score": {
                "band": result['probability_score'].band.value,
                "raw_score": result['probability_score'].raw_score,
                "adjusted_score": result['probability_score'].adjusted_score,
                "confidence": result['probability_score'].confidence_level
            },
            "expected_band": result['expected_band'].value,
            "matches_expected": result['matches_expected'],
            "status_marker": status_marker
        }
    
    # Add performance metrics
    report.performance_metrics = {
        "scenarios_tested": len(results),
        "successful_predictions": sum(1 for r in results if r['matches_expected']),
        "average_confidence": sum(r['probability_score'].confidence_level for r in results) / len(results)
    }
    
    report.status = "completed"
    return report

if __name__ == "__main__":
    test_probability_scoring() 