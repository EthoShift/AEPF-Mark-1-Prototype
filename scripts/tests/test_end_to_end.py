import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import json
import time

from scripts.context_engine import ContextEngine
from scripts.ethical_governor import EthicalGovernor
from scripts.decision_analysis.probability_scorer import ProbabilityScorer, ProbabilityBand
from scripts.decision_analysis.narrative_generator import OutcomeNarrative
from scripts.report_templates import TestReport, ReportManager
from scripts.context_models import StakeholderData, RealTimeMetrics, ContextEntry, StakeholderRole

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_scenarios() -> Dict[str, List[Dict]]:
    """Load test scenarios from generated files"""
    scenarios = {}
    test_data_dir = Path("test_data")
    
    for scenario_file in test_data_dir.glob("*_scenarios.json"):
        with open(scenario_file, 'r') as f:
            scenario_type = scenario_file.stem.split('_')[0]
            scenarios[scenario_type] = json.load(f)
            
    return scenarios

def test_end_to_end() -> TestReport:
    """Test system with generated scenarios"""
    
    # Initialize report
    report = TestReport.create(
        test_name="end_to_end_system_test",
        components=["ContextEngine", "EthicalGovernor", "ProbabilityScorer", "OutcomeNarrative"]
    )
    
    start_time = time.time()
    
    try:
        # Load test scenarios
        scenarios = load_test_scenarios()
        logger.info(f"Loaded {len(scenarios)} scenario types")
        
        # Initialize components
        context_engine = ContextEngine()
        ethical_governor = EthicalGovernor()
        narrative_generator = OutcomeNarrative()
        
        results = []
        for scenario_type, scenario_list in scenarios.items():
            logger.info(f"\nProcessing {scenario_type} scenarios...")
            
            for scenario in scenario_list:
                logger.info(f"\nEvaluating: {scenario['name']}")
                
                # Create context from scenario
                context = {
                    "stakeholder": StakeholderData(**scenario['stakeholders'][0]),
                    "context_type": scenario_type,
                    "description": scenario['description'],
                    "compliance_requirements": scenario['compliance_requirements'],
                    "risk_level": scenario['risk_level'],
                    "regional_context": scenario['regional_context']
                }
                
                # Add to context engine
                context_entry = ContextEntry(
                    entry_type="assessment",
                    data=context
                )
                context_engine.add_context_entry(context_entry)
                
                # Evaluate through ethical governor
                decision = ethical_governor.evaluate_action(scenario['action'], context)
                
                # Generate narrative
                narrative = narrative_generator.generate_narrative(
                    decision.probability_score.adjusted_score,
                    decision.probability_score.band,
                    decision.prism_scores,
                    context,
                    decision.confidence_score
                )
                
                # Record results
                results.append({
                    "scenario_type": scenario_type,
                    "scenario_name": scenario['name'],
                    "decision": decision,
                    "narrative": narrative,
                    "matches_expected": decision.recommendation.value == scenario['expected_outcome']
                })
                
                # Log results
                logger.info(f"Decision: {decision.recommendation}")
                logger.info(f"Probability Band: {decision.probability_score.band}")
                logger.info(f"Confidence Score: {decision.confidence_score}")
                logger.info("\nNarrative:")
                logger.info(narrative)
        
        # Add results to report
        for result in results:
            report.results[f"{result['scenario_type']}_{result['scenario_name']}"] = {
                "status": "success" if result['matches_expected'] else "warning",
                "decision": str(result['decision'].recommendation),
                "probability_band": result['decision'].probability_score.band.value,
                "confidence_score": result['decision'].confidence_score,
                "narrative": result['narrative'],
                "matches_expected": result['matches_expected']
            }
        
        # Add performance metrics
        report.performance_metrics = {
            "scenarios_tested": len(results),
            "successful_predictions": sum(1 for r in results if r['matches_expected']),
            "average_confidence": sum(r['decision'].confidence_score for r in results) / len(results),
            "execution_time": time.time() - start_time
        }
        
        report.status = "completed"
        
    except Exception as e:
        report.status = "failed"
        report.errors.append(str(e))
        logger.error(f"Error during end-to-end test: {str(e)}", exc_info=True)
    
    return report

if __name__ == "__main__":
    # Initialize report manager
    report_manager = ReportManager()
    
    # Run test and get report
    test_report = test_end_to_end()
    test_report.test_id = report_manager.get_next_id()
    
    # Save report
    try:
        json_path, text_path = report_manager.save_report(test_report)
        logger.info(f"\nReport saved to: {text_path}")
        
        # Display report content
        with open(text_path, 'r', encoding='cp1252', errors='ignore') as f:
            print("\nTest Results:")
            print("=" * 80)
            print(f.read())
            print("=" * 80)
            
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}") 