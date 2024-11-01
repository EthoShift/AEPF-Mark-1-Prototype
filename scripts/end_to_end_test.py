import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from scripts.context_engine import ContextEngine
from scripts.ethical_governor import EthicalGovernor
from scripts.decision_analysis.probability_scorer import ProbabilityScorer, ProbabilityBand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Structure for evaluation results"""
    scenario_id: str
    context_type: str
    initial_score: float
    bayesian_posterior: float
    fuzzy_memberships: Dict[str, float]
    probability_band: ProbabilityBand
    final_decision: str
    expected_band: str
    confidence: float

class EndToEndTest:
    def __init__(self):
        """Initialize AEPF framework components"""
        self.context_engine = ContextEngine()
        self.ethical_governor = EthicalGovernor()
        self.probability_scorer = ProbabilityScorer()
        
        # Create reports directory if it doesn't exist
        self.reports_dir = Path("reports/end_tests")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def load_mock_data(self) -> List[Dict[str, Any]]:
        """Load mock dataset for testing"""
        logger.info("Loading mock test scenarios")
        
        scenarios = [
            {
                "id": "TEST001",
                "context_type": "medical",
                "action": "implement_ai_diagnosis",
                "stakeholder_consensus": "high",
                "metrics": {
                    "safety_score": 0.8,
                    "efficacy_score": 0.9
                },
                "risk_level": "medium",
                "compliance_data": {
                    "regulatory_approval": True,
                    "data_protection": "high"
                },
                "expected_band": "HIGH"
            },
            {
                "id": "TEST002",
                "context_type": "environmental",
                "action": "green_datacenter_migration",
                "environmental_priority": "high",
                "sustainability_focus": True,
                "metrics": {
                    "emissions_reduction": 0.7,
                    "energy_efficiency": 0.8
                },
                "risk_level": "low",
                "compliance_data": {
                    "environmental_standards": True,
                    "regulatory_compliance": True
                },
                "expected_band": "HIGH"
            },
            {
                "id": "TEST003",
                "context_type": "cultural",
                "action": "implement_privacy_controls",
                "cultural_context": {
                    "privacy_emphasis": "high",
                    "innovation_tolerance": "conservative"
                },
                "stakeholder_consensus": "medium",
                "risk_level": "medium",
                "compliance_data": {
                    "privacy_standards": True,
                    "cultural_assessment": "completed"
                },
                "expected_band": "MEDIUM"
            }
        ]
        
        logger.info(f"Loaded {len(scenarios)} test scenarios")
        return scenarios
    
    def evaluate_scenario(self, scenario: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single scenario through the framework"""
        logger.debug(f"Evaluating scenario {scenario['id']}")
        
        try:
            # Context evaluation
            context_result = self.context_engine.evaluate_context(scenario)
            
            # Prepare evaluation data
            evaluation_data = {
                'context_type': scenario['context_type'],
                'action': scenario['action'],
                'metrics': scenario.get('metrics', {}),
                'compliance_data': scenario.get('compliance_data', {}),
                'risk_level': scenario.get('risk_level', 'medium'),
                'probability_score': context_result['updated_score'],
                'fuzzy_memberships': context_result['fuzzy_memberships']
            }
            
            # Get ethical governor decision
            decision = self.ethical_governor.evaluate_action(
                action=scenario['action'],
                context=evaluation_data
            )
            
            # Determine probability band
            max_membership = max(
                context_result['fuzzy_memberships'].items(),
                key=lambda x: x[1]
            )
            band = ProbabilityBand[max_membership[0].upper()]
            
            return EvaluationResult(
                scenario_id=scenario['id'],
                context_type=scenario['context_type'],
                initial_score=context_result['base_score'],
                bayesian_posterior=context_result['updated_score'],
                fuzzy_memberships=context_result['fuzzy_memberships'],
                probability_band=band,
                final_decision=str(decision),
                expected_band=scenario['expected_band'],
                confidence=context_result['confidence']
            )
            
        except Exception as e:
            logger.error(f"Error evaluating scenario {scenario['id']}: {str(e)}")
            raise
    
    def generate_report(self, results: List[EvaluationResult]) -> str:
        """Generate detailed evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = []
        
        report.append("=" * 80)
        report.append("AEPF End-to-End Test Report")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Scenarios Evaluated: {len(results)}")
        report.append("\n")
        
        for result in results:
            report.append(f"Scenario: {result.scenario_id}")
            report.append("-" * 40)
            report.append(f"Context Type: {result.context_type}")
            report.append(f"Initial Score: {result.initial_score:.4f}")
            report.append(f"Bayesian Posterior: {result.bayesian_posterior:.4f}")
            report.append(f"Confidence: {result.confidence:.4f}")
            report.append("\nFuzzy Memberships:")
            for band, score in result.fuzzy_memberships.items():
                report.append(f"  {band}: {score:.4f}")
            report.append(f"\nProbability Band: {result.probability_band}")
            report.append(f"Final Decision: {result.final_decision}")
            report.append(f"Expected Band: {result.expected_band}")
            report.append("\n")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def run_tests(self, generate_report: bool = True) -> None:
        """Run end-to-end tests"""
        logger.info("Starting end-to-end tests")
        
        try:
            # Load test data
            scenarios = self.load_mock_data()
            
            # Evaluate scenarios
            results = []
            for scenario in scenarios:
                result = self.evaluate_scenario(scenario)
                results.append(result)
                logger.info(f"Evaluated scenario {scenario['id']}")
            
            # Generate and save report
            if generate_report:
                report_content = self.generate_report(results)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = self.reports_dir / f"end_to_end_report_{timestamp}.txt"
                
                report_path.write_text(report_content)
                logger.info(f"Report saved to: {report_path}")
                
        except Exception as e:
            logger.error(f"Error in end-to-end test: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Run AEPF end-to-end tests")
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate detailed test report')
    
    args = parser.parse_args()
    
    test_runner = EndToEndTest()
    test_runner.run_tests(generate_report=args.generate_report)

if __name__ == "__main__":
    main() 