import unittest
from typing import Dict, List
from scripts.ethical_governor import EthicalGovernor
from scripts.context_engine import ContextEngine
from scripts.decision_analysis.probability_scorer import ProbabilityScorer
from scripts.decision_analysis.feedback_loop import FeedbackLoop

class AEPFTestSuite(unittest.TestCase):
    """Comprehensive test suite for AEPF Mk1"""
    
    def setUp(self):
        self.governor = EthicalGovernor()
        self.context_engine = ContextEngine()
        self.probability_scorer = ProbabilityScorer()
        self.feedback_loop = FeedbackLoop()
    
    def test_probability_scoring(self):
        """Test probability scoring functionality"""
        # Test scenarios
        scenarios = [
            {
                "name": "High Privacy Impact",
                "action": "collect_user_data",
                "expected_band": "low"
            },
            {
                "name": "Innovation Focus",
                "action": "implement_ai_optimization",
                "expected_band": "high"
            }
        ]
        
        for scenario in scenarios:
            context = self.context_engine.get_decision_context(scenario['action'])
            decision = self.governor.evaluate_action(scenario['action'], context)
            
            self.assertIsNotNone(decision.probability_score)
            self.assertEqual(
                decision.probability_score.band.value,
                scenario['expected_band']
            )
    
    def test_feedback_loop(self):
        """Test feedback loop refinement"""
        # Test feedback loop convergence
        action = "deploy_system_update"
        context = self.context_engine.get_decision_context(action)
        decision = self.governor.evaluate_action(action, context)
        
        self.assertTrue(decision.feedback_result.convergence_achieved)
        self.assertLess(len(decision.feedback_result.iterations), 5) 