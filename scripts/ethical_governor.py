from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time
from scripts.context_engine import ContextEngine
from scripts.context_models import ContextEntry
from scripts.prisms.human_centric import HumanCentricPrism
from scripts.prisms.sentient_first import SentientFirstPrism
from scripts.prisms.ecocentric import EcocentricPrism
from scripts.prisms.innovation_focused import InnovationFocusedPrism

class DecisionCategory(Enum):
    """Categories for ethical decisions"""
    CRITICAL = "critical"
    HIGH_IMPACT = "high_impact"
    MODERATE = "moderate"
    LOW_IMPACT = "low_impact"

class DecisionOutcome(Enum):
    """Possible decision outcomes"""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"

@dataclass
class EthicalDecision:
    """Represents an ethical decision made by the governor"""
    decision_id: str
    category: DecisionCategory
    recommendation: DecisionOutcome
    confidence_score: float
    prism_scores: Dict[str, float]
    context_snapshot: Dict
    reasoning: List[str]
    risk_factors: List[str]
    mitigation_steps: List[str]
    stakeholder_impact: Dict[str, float]
    timestamp: float

class EthicalGovernor:
    """
    Central ethical governing component for AEPF Mk1
    Coordinates ethical analysis across different ethical prisms
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.context_engine = ContextEngine()
        self.prisms = {
            'human': HumanCentricPrism(),
            'sentient': SentientFirstPrism(),
            'eco': EcocentricPrism(),
            'innovation': InnovationFocusedPrism()
        }
        # Define prism weights for different decision categories
        self.prism_weights = {
            DecisionCategory.CRITICAL: {
                'human': 0.4,
                'sentient': 0.3,
                'eco': 0.2,
                'innovation': 0.1
            },
            DecisionCategory.HIGH_IMPACT: {
                'human': 0.35,
                'sentient': 0.25,
                'eco': 0.25,
                'innovation': 0.15
            },
            DecisionCategory.MODERATE: {
                'human': 0.3,
                'sentient': 0.2,
                'eco': 0.2,
                'innovation': 0.3
            },
            DecisionCategory.LOW_IMPACT: {
                'human': 0.25,
                'sentient': 0.25,
                'eco': 0.25,
                'innovation': 0.25
            }
        }
        self.decisions_history: List[EthicalDecision] = []
        
    def categorize_decision(self, action: str, context: Dict) -> DecisionCategory:
        """Determine the category of the decision based on context and action"""
        # Extract relevant factors from context
        urgency_level = context.get('urgency_level', 'low')
        stakeholder_impact = self._calculate_stakeholder_impact(context)
        system_metrics = context.get('system_metrics', {})
        
        # Critical conditions
        if (urgency_level == 'critical' or 
            max(stakeholder_impact.values(), default=0) > 0.8 or
            'security' in action.lower()):
            return DecisionCategory.CRITICAL
            
        # High impact conditions
        elif (urgency_level == 'high' or 
              max(stakeholder_impact.values(), default=0) > 0.6):
            return DecisionCategory.HIGH_IMPACT
            
        # Moderate impact conditions
        elif (urgency_level == 'medium' or 
              max(stakeholder_impact.values(), default=0) > 0.4):
            return DecisionCategory.MODERATE
            
        # Default to low impact
        return DecisionCategory.LOW_IMPACT
    
    def _calculate_stakeholder_impact(self, context: Dict) -> Dict[str, float]:
        """Calculate impact scores for different stakeholder groups"""
        impact_scores = {}
        stakeholder = context.get('stakeholder')
        
        if stakeholder:
            if isinstance(stakeholder, ContextEntry):
                stakeholder_data = stakeholder.data
            else:
                stakeholder_data = stakeholder  # If it's already StakeholderData
                
            impact_scores[stakeholder_data.role] = stakeholder_data.impact_score / 100.0
            
        return impact_scores
    
    def _calculate_weighted_score(self, 
                                prism_scores: Dict[str, float], 
                                category: DecisionCategory) -> float:
        """Calculate weighted confidence score based on decision category"""
        weights = self.prism_weights[category]
        weighted_score = sum(
            prism_scores[prism] * weights[prism]
            for prism in prism_scores
        )
        return weighted_score
    
    def _generate_mitigation_steps(self, 
                                 risks: List[str], 
                                 category: DecisionCategory) -> List[str]:
        """Generate mitigation steps based on identified risks"""
        mitigation_steps = []
        for risk in risks:
            if 'privacy' in risk.lower():
                mitigation_steps.append("Implement enhanced privacy controls")
            elif 'security' in risk.lower():
                mitigation_steps.append("Conduct security audit before deployment")
            elif 'impact' in risk.lower():
                mitigation_steps.append("Develop impact monitoring framework")
            # Add more risk-specific mitigations
        return mitigation_steps
    
    def evaluate_action(self, action: str, context: Dict) -> EthicalDecision:
        """
        Evaluate an action across all ethical prisms
        
        Args:
            action: The proposed action to evaluate
            context: Current context information
            
        Returns:
            EthicalDecision containing the evaluation results
        """
        # Determine decision category
        category = self.categorize_decision(action, context)
        self.logger.info(f"Decision categorized as: {category.value}")
        
        # Initialize evaluation results
        prism_scores = {}
        reasoning = []
        risks = []
        
        # Evaluate through each prism
        for prism_name, prism in self.prisms.items():
            try:
                evaluation = prism.evaluate(action, context)
                prism_scores[prism_name] = getattr(evaluation, 'impact_score', 0.0)
                
                # Collect recommendations and risks
                if hasattr(evaluation, 'recommendations'):
                    reasoning.extend(evaluation.recommendations)
                if hasattr(evaluation, 'risks'):
                    risks.extend(evaluation.risks)
                    
            except Exception as e:
                self.logger.error(f"Error in {prism_name} evaluation: {str(e)}")
                reasoning.append(f"Error in {prism_name} evaluation: {str(e)}")
                prism_scores[prism_name] = 0.0
        
        # Calculate weighted confidence score
        confidence_score = self._calculate_weighted_score(prism_scores, category)
        
        # Generate mitigation steps
        mitigation_steps = self._generate_mitigation_steps(risks, category)
        
        # Calculate stakeholder impact
        stakeholder_impact = self._calculate_stakeholder_impact(context)
        
        # Determine recommendation based on category and score
        recommendation = self._determine_recommendation(
            confidence_score, 
            category,
            len(risks)
        )
        
        # Create decision record
        decision = EthicalDecision(
            decision_id=f"decision_{len(self.decisions_history) + 1}",
            category=category,
            recommendation=recommendation,
            confidence_score=confidence_score,
            prism_scores=prism_scores,
            context_snapshot=context,
            reasoning=reasoning,
            risk_factors=risks,
            mitigation_steps=mitigation_steps,
            stakeholder_impact=stakeholder_impact,
            timestamp=time.time()
        )
        
        # Store decision in history
        self.decisions_history.append(decision)
        
        # Log decision details
        self.logger.info(
            f"Decision made: {decision.recommendation.value} "
            f"(confidence: {confidence_score:.2f})"
        )
        
        return decision
    
    def _determine_recommendation(self, 
                                confidence_score: float, 
                                category: DecisionCategory,
                                risk_count: int) -> DecisionOutcome:
        """Determine recommendation based on score, category, and risks"""
        if category == DecisionCategory.CRITICAL:
            if confidence_score >= 0.8 and risk_count < 3:
                return DecisionOutcome.APPROVE
            elif confidence_score >= 0.6:
                return DecisionOutcome.ESCALATE
            else:
                return DecisionOutcome.REJECT
                
        elif category == DecisionCategory.HIGH_IMPACT:
            if confidence_score >= 0.75:
                return DecisionOutcome.APPROVE
            elif confidence_score >= 0.5:
                return DecisionOutcome.REVIEW
            else:
                return DecisionOutcome.REJECT
                
        else:  # MODERATE or LOW_IMPACT
            if confidence_score >= 0.7:
                return DecisionOutcome.APPROVE
            elif confidence_score >= 0.4:
                return DecisionOutcome.REVIEW
            else:
                return DecisionOutcome.REJECT
    
    def get_decision_history(self) -> List[EthicalDecision]:
        """Retrieve history of ethical decisions"""
        return self.decisions_history