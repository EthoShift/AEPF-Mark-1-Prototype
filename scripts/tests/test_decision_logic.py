from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json
import sys

from scripts.ethical_governor import EthicalGovernor, DecisionOutcome
from scripts.decision_analysis.probability_scorer import ProbabilityBand
from scripts.context_models import StakeholderData, StakeholderRole

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('debug_decision_logic.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class DecisionTestCase:
    """Test case for decision logic evaluation"""
    scenario_id: str
    description: str
    action: str
    context: Dict[str, Any]
    expected_outcome: DecisionOutcome
    expected_probability_band: ProbabilityBand
    expected_secondary_review: bool
    expected_confidence_range: Tuple[float, float]

@dataclass
class TestResult:
    """Results from a decision logic test"""
    scenario_id: str
    initial_outcome: DecisionOutcome
    final_outcome: DecisionOutcome
    probability_band: ProbabilityBand
    confidence_score: float
    secondary_review_activated: bool
    secondary_effects: Dict[str, float]
    narrative: str
    passed: bool
    timestamp: datetime

class DecisionLogicTestSuite:
    """Test suite for evaluating refined decision logic"""
    
    def __init__(self, output_dir: str = "reports/decision_tests"):
        self.governor = EthicalGovernor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_cases = self._generate_test_cases()
        
    def _generate_test_cases(self) -> List[DecisionTestCase]:
        """Generate comprehensive test cases"""
        return [
            # DL001: High human impact with low initial probability
            DecisionTestCase(
                scenario_id="DL001",
                description="High human impact with low initial probability",
                action="implement_ai_medical_diagnosis",
                context={
                    "stakeholder": StakeholderData(
                        id=1,
                        name="Medical Center",
                        role=StakeholderRole.USER,
                        region="US-CA",
                        priority_level=1,
                        impact_score=95.0
                    ),
                    "context_type": "medical",
                    "urgency_level": "high",
                    "risk_level": "high",
                    "human_welfare_priority": "high",
                    "compliance_requirements": ["HIPAA", "FDA"],
                    "system_metrics": {"reliability": 0.95},
                    "mitigation_strategies": ["phased_rollout", "human_oversight"],
                    "technical_readiness": "high",
                    "uncertainty_level": "medium",
                    "stakeholder_consensus": "moderate"
                },
                expected_outcome=DecisionOutcome.REVIEW,
                expected_probability_band=ProbabilityBand.LOW,
                expected_secondary_review=True,
                expected_confidence_range=(0.4, 0.7)
            ),
            
            # DL002: Strong eco impact with moderate probability
            DecisionTestCase(
                scenario_id="DL002",
                description="Strong eco impact with moderate probability",
                action="implement_green_data_center",
                context={
                    "context_type": "environmental",
                    "environmental_priority": "high",
                    "sustainability_focus": True,
                    "resource_efficiency": 0.8,
                    "eco_impact": "significant",
                    "stakeholder_consensus": "high",
                    "system_metrics": {"energy_efficiency": 0.9},
                    "mitigation_strategies": ["efficiency_monitoring"],
                    "technical_readiness": "high",
                    "uncertainty_level": "low",
                    "stakeholder": StakeholderData(
                        id=2,
                        name="Data Center Ops",
                        role=StakeholderRole.MANAGER,
                        region="US-CA",
                        priority_level=2,
                        impact_score=85.0
                    )
                },
                expected_outcome=DecisionOutcome.APPROVE,
                expected_probability_band=ProbabilityBand.MEDIUM,
                expected_secondary_review=False,
                expected_confidence_range=(0.6, 0.9)
            ),
            
            # DL003: Cultural sensitivity with mixed impacts
            DecisionTestCase(
                scenario_id="DL003",
                description="Cultural sensitivity with mixed impacts",
                action="deploy_facial_recognition",
                context={
                    "context_type": "cultural",
                    "cultural_context": {
                        "privacy_emphasis": "very_high",
                        "innovation_tolerance": "conservative"
                    },
                    "stakeholder_consensus": "low",
                    "privacy_level": "high",
                    "technical_readiness": "high",
                    "uncertainty_level": "high",
                    "stakeholder": StakeholderData(
                        id=3,
                        name="Regional Office",
                        role=StakeholderRole.MANAGER,
                        region="JP-13",
                        priority_level=1,
                        impact_score=90.0
                    ),
                    "mitigation_strategies": ["cultural_assessment", "phased_deployment"],
                    "compliance_requirements": ["Privacy Act", "Cultural Protection"]
                },
                expected_outcome=DecisionOutcome.ESCALATE,
                expected_probability_band=ProbabilityBand.MEDIUM,
                expected_secondary_review=True,
                expected_confidence_range=(0.3, 0.6)
            )
        ]
    
    def _execute_test_case(self, test_case: DecisionTestCase) -> TestResult:
        """Execute a single test case"""
        try:
            # Initialize components
            governor = EthicalGovernor()
            
            # Evaluate action
            decision = governor.evaluate_action(test_case.action, test_case.context)
            
            # Extract results
            initial_outcome = getattr(decision.probability_score, 'initial_recommendation', None)
            final_outcome = decision.recommendation
            probability_band = decision.probability_score.band
            confidence_score = decision.confidence_score
            secondary_review = (initial_outcome != final_outcome)
            
            # Get secondary effects
            secondary_effects = {
                'human': decision.prism_scores.get('human', 0),
                'sentient': decision.prism_scores.get('sentient', 0),
                'eco': decision.prism_scores.get('eco', 0),
                'innovation': decision.prism_scores.get('innovation', 0)
            }
            
            # Generate narrative
            narrative = []
            narrative.append(f"Decision context: {test_case.context.get('context_type', '')} scenario")
            
            if final_outcome == DecisionOutcome.APPROVE:
                narrative.append("Analysis indicates positive ethical alignment")
            elif final_outcome == DecisionOutcome.REVIEW:
                narrative.append("Further review recommended due to ethical considerations")
            elif final_outcome == DecisionOutcome.ESCALATE:
                narrative.append("Escalation required due to ethical complexity")
            else:
                narrative.append("Ethical concerns prevent approval")
                
            # Add stakeholder impact if present
            if test_case.context.get('stakeholder'):
                narrative.append(f"Primary stakeholder impact: {test_case.context['stakeholder'].role}")
                
            # Add secondary review analysis if activated
            if secondary_review:
                narrative.append("\nSecondary Review Analysis:")
                if any(score > 2.0 for score in secondary_effects.values()):
                    total_impact = sum(abs(score) for score in secondary_effects.values())
                    narrative.append(
                        f"Secondary review triggered: High impact score ({total_impact:.2f}) "
                        "despite low probability"
                    )
                    narrative.extend([
                        "Recommendation updated based on:",
                        "- Significant positive impact potential",
                        "- Manageable risk factors",
                        "- Available mitigation strategies"
                    ])
            
            # Create test result
            result = TestResult(
                scenario_id=test_case.scenario_id,
                initial_outcome=initial_outcome,
                final_outcome=final_outcome,
                probability_band=probability_band,
                confidence_score=confidence_score,
                secondary_review_activated=secondary_review,
                secondary_effects=secondary_effects,
                narrative=narrative,
                passed=self._evaluate_test_case(decision, test_case),
                timestamp=datetime.now()
            )
            
            # Log results
            self._log_test_result(result, test_case)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during test execution: {str(e)}")
            # Return failed result
            return TestResult(
                scenario_id=test_case.scenario_id,
                initial_outcome=None,
                final_outcome=DecisionOutcome.REJECT,
                probability_band=ProbabilityBand.LOW,
                confidence_score=0.0,
                secondary_review_activated=False,
                secondary_effects={},
                narrative=[f"Error during test execution: {str(e)}"],
                passed=False,
                timestamp=datetime.now()
            )
    
    def run_tests(self) -> List[TestResult]:
        """Run all test cases"""
        results = []
        for test_case in self.test_cases:
            logger.info(f"\nTesting scenario: {test_case.description}")
            result = self._execute_test_case(test_case)
            results.append(result)
        return results
    
    def _evaluate_test_case(self, decision: Any, case: DecisionTestCase) -> bool:
        """Evaluate if test case passed"""
        confidence_in_range = (
            case.expected_confidence_range[0] <= 
            decision.confidence_score <= 
            case.expected_confidence_range[1]
        )
        
        # Handle potential None values in initial recommendation
        initial_recommendation = getattr(
            decision.probability_score,
            'initial_recommendation',
            decision.recommendation
        )
        
        secondary_review_correct = (
            case.expected_secondary_review == 
            (decision.recommendation != initial_recommendation)
        )
        
        return all([
            decision.recommendation == case.expected_outcome,
            decision.probability_score.band == case.expected_probability_band,
            confidence_in_range,
            secondary_review_correct
        ])
    
    def _log_test_result(self, result: TestResult, case: DecisionTestCase):
        """Log detailed test results"""
        logger.info(f"\nTest Results for {result.scenario_id}:")
        logger.info(f"Initial Outcome: {result.initial_outcome if result.initial_outcome else 'None'}")
        logger.info(f"Final Outcome: {result.final_outcome}")
        logger.info(f"Probability Band: {result.probability_band}")
        logger.info(f"Confidence Score: {result.confidence_score}")
        logger.info(f"Secondary Review Activated: {result.secondary_review_activated}")
        logger.info("\nSecondary Effects:")
        for effect, score in result.secondary_effects.items():
            logger.info(f"- {effect}: {score:.2f}")
        logger.info(f"\nNarrative:\n{result.narrative}")
        logger.info(f"\nTest Passed: {result.passed}")
        
        if not result.passed:
            logger.warning("\nExpected vs Actual:")
            logger.warning(f"Outcome: {case.expected_outcome} vs {result.final_outcome}")
            logger.warning(f"Band: {case.expected_probability_band} vs {result.probability_band}")
            logger.warning(
                f"Confidence Range: {case.expected_confidence_range} vs {result.confidence_score}"
            )
            logger.warning(
                f"Secondary Review: {case.expected_secondary_review} vs {result.secondary_review_activated}"
            )
    
    def generate_report(self, results: List[TestResult]) -> None:
        """Generate comprehensive test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"decision_logic_report_{timestamp}"
        report_path.mkdir(exist_ok=True)
        
        # Generate JSON report with safe handling of None values
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'total_scenarios': len(results),
            'results': [
                {
                    'scenario_id': r.scenario_id,
                    'timestamp': r.timestamp.isoformat(),
                    'initial_outcome': r.initial_outcome.value if r.initial_outcome else "NONE",
                    'final_outcome': r.final_outcome.value if r.final_outcome else "NONE",
                    'probability_band': r.probability_band.value if r.probability_band else "NONE",
                    'confidence_score': r.confidence_score,
                    'secondary_review': r.secondary_review_activated,
                    'passed': r.passed,
                    'secondary_effects': {
                        k: f"{v:.2f}" for k, v in r.secondary_effects.items()
                    } if r.secondary_effects else {},
                    'narrative': r.narrative if r.narrative else []
                }
                for r in results
            ]
        }
        
        # Save JSON report
        with open(report_path / 'detailed_results.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Generate text summary
        with open(report_path / 'summary.txt', 'w') as f:
            f.write("Decision Logic Test Results\n")
            f.write("=" * 80 + "\n\n")
            
            for result in results:
                f.write(f"\nScenario: {result.scenario_id}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Initial Outcome: {result.initial_outcome.value if result.initial_outcome else 'NONE'}\n")
                f.write(f"Final Outcome: {result.final_outcome.value if result.final_outcome else 'NONE'}\n")
                f.write(f"Probability Band: {result.probability_band.value if result.probability_band else 'NONE'}\n")
                f.write(f"Confidence Score: {result.confidence_score:.3f}\n")
                f.write(f"Secondary Review: {'Yes' if result.secondary_review_activated else 'No'}\n")
                f.write(f"Test Passed: {'Yes' if result.passed else 'No'}\n")
                
                if result.secondary_effects:
                    f.write("\nSecondary Effects:\n")
                    for effect, score in result.secondary_effects.items():
                        f.write(f"- {effect}: {score:.2f}\n")
                
                if result.narrative:
                    f.write("\nNarrative:\n")
                    for line in result.narrative:
                        f.write(f"- {line}\n")
                
                f.write("\n" + "=" * 80 + "\n")
    
def run_test_with_debug():
    """Run decision logic tests with enhanced debugging"""
    logger.info("Starting decision logic tests with debug logging")
    
    try:
        # Initialize test components
        logger.debug("Initializing test components")
        test_suite = DecisionLogicTestSuite()
        
        # Run tests with step tracking
        logger.debug("Beginning test execution")
        for test_case in test_suite.test_cases:
            logger.info(f"\nExecuting test case: {test_case.scenario_id}")
            logger.debug(f"Test context: {test_case.context}")
            
            try:
                # Track execution depth
                sys.setrecursionlimit(100)  # Lower limit to catch recursion earlier
                
                # Execute test with depth tracking
                result = test_suite._execute_test_case(test_case)
                
                logger.info(f"Test {test_case.scenario_id} completed successfully")
                logger.debug(f"Test result: {result}")
                
            except RecursionError as e:
                logger.error(f"Recursion detected in test {test_case.scenario_id}")
                logger.error(f"Error details: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error in test {test_case.scenario_id}: {str(e)}")
                
            finally:
                # Reset recursion limit
                sys.setrecursionlimit(1000)
    
    except Exception as e:
        logger.error(f"Critical error in test execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_test_with_debug()
    