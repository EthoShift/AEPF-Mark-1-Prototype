import unittest
from typing import Dict
from scripts.context_engine import ContextEngine
import logging

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestContextEngine(unittest.TestCase):
    def setUp(self):
        """Initialize test components"""
        self.context_engine = ContextEngine()
    
    def test_fuzzy_scoring(self):
        """Test fuzzy membership scoring"""
        # Test case 1: Score at boundary between medium and high (0.65)
        memberships = self.context_engine.get_fuzzy_memberships(0.65)
        self.assertAlmostEqual(memberships['low'], 0.0, places=4)
        self.assertAlmostEqual(memberships['medium'], 0.5, places=2)
        self.assertAlmostEqual(memberships['high'], 0.5, places=2)
        self.assertAlmostEqual(sum(memberships.values()), 1.0, places=4)
        
        # Test case 2: Clear low score (0.2)
        memberships = self.context_engine.get_fuzzy_memberships(0.2)
        self.assertAlmostEqual(memberships['low'], 1.0, places=4)
        self.assertAlmostEqual(memberships['medium'], 0.0, places=4)
        self.assertAlmostEqual(memberships['high'], 0.0, places=4)
        
        # Test case 3: Clear medium score (0.5)
        memberships = self.context_engine.get_fuzzy_memberships(0.5)
        self.assertAlmostEqual(memberships['low'], 0.0, places=4)
        self.assertAlmostEqual(memberships['medium'], 1.0, places=4)
        self.assertAlmostEqual(memberships['high'], 0.0, places=4)
        
        # Test case 4: Clear high score (0.8)
        memberships = self.context_engine.get_fuzzy_memberships(0.8)
        self.assertAlmostEqual(memberships['low'], 0.0, places=4)
        self.assertAlmostEqual(memberships['medium'], 0.0, places=4)
        self.assertAlmostEqual(memberships['high'], 1.0, places=4)
        
        # Test case 5: Low-Medium boundary (0.35)
        memberships = self.context_engine.get_fuzzy_memberships(0.35)
        self.assertTrue(0.0 < memberships['low'] < 1.0)
        self.assertTrue(0.0 < memberships['medium'] < 1.0)
        self.assertAlmostEqual(memberships['high'], 0.0, places=4)
        self.assertAlmostEqual(sum(memberships.values()), 1.0, places=4)
    
    def test_bayesian_adjustment(self):
        """Test Bayesian posterior calculation"""
        # Test case 1: Standard case
        prior = 0.5
        likelihood = 0.7
        evidence = 0.6
        posterior = self.context_engine.bayesian_update(prior, likelihood, evidence)
        self.assertAlmostEqual(posterior, 0.5833, places=4)
        
        # Test case 2: Zero evidence
        posterior = self.context_engine.bayesian_update(0.5, 0.7, 0.0)
        self.assertAlmostEqual(posterior, 0.5, places=4)
        
        # Test case 3: High certainty
        posterior = self.context_engine.bayesian_update(0.9, 0.9, 0.9)
        self.assertAlmostEqual(posterior, 0.9, places=4)
        
        # Test case 4: Low certainty
        posterior = self.context_engine.bayesian_update(0.1, 0.1, 0.1)
        self.assertAlmostEqual(posterior, 0.1, places=4)
        
        # Test case 5: Boundary case
        posterior = self.context_engine.bayesian_update(1.0, 1.0, 0.5)
        self.assertAlmostEqual(posterior, 1.0, places=4)
    
    def test_context_evaluation(self):
        """Test complete context evaluation"""
        # Test medical context
        medical_context = {
            'context_type': 'medical',
            'stakeholder_consensus': 'high',
            'metrics': {
                'safety_score': 0.8,
                'efficacy_score': 0.9
            }
        }
        
        result = self.context_engine.evaluate_context(medical_context)
        
        # Verify result structure
        self.assertIn('base_score', result)
        self.assertIn('fuzzy_memberships', result)
        self.assertIn('updated_score', result)
        self.assertIn('confidence', result)
        
        # Verify value ranges
        self.assertTrue(0 <= result['base_score'] <= 1)
        self.assertTrue(0 <= result['updated_score'] <= 1)
        self.assertTrue(0 <= result['confidence'] <= 1)
        
        # Verify fuzzy memberships
        memberships = result['fuzzy_memberships']
        self.assertAlmostEqual(sum(memberships.values()), 1.0, places=4)

if __name__ == '__main__':
    unittest.main() 