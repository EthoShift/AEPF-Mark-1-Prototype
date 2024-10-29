from typing import Dict, List, Optional, Tuple, Union
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
from scripts.decision_analysis.probability_scorer import (
    ProbabilityScorer, 
    ProbabilityBand,
    ProbabilityScore
)
from scripts.decision_analysis.feedback_loop import FeedbackLoop, FeedbackLoopResult

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
    probability_score: Optional[ProbabilityScore] = None

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
                                 risks: List[Union[str, Dict[str, str]]], 
                                 category: DecisionCategory) -> List[str]:
        """Generate mitigation steps based on identified risks"""
        mitigation_steps = []
        
        for risk in risks:
            if isinstance(risk, dict):
                # Handle dictionary-type risks
                risk_description = risk.get('description', '').lower()
                risk_level = risk.get('level', '').lower()
                
                if 'privacy' in risk_description:
                    mitigation_steps.append("Implement enhanced privacy controls")
                elif 'security' in risk_description:
                    mitigation_steps.append("Conduct security audit before deployment")
                elif 'impact' in risk_description:
                    mitigation_steps.append("Develop impact monitoring framework")
                
                # Add specific mitigations based on risk level
                if risk_level == 'high':
                    mitigation_steps.append(f"High-priority mitigation required for: {risk['description']}")
                elif risk_level == 'critical':
                    mitigation_steps.append(f"URGENT: Immediate mitigation needed for: {risk['description']}")
            
            else:
                # Handle string-type risks
                risk_str = str(risk).lower()
                if 'privacy' in risk_str:
                    mitigation_steps.append("Implement privacy protection measures")
                elif 'security' in risk_str:
                    mitigation_steps.append("Implement security controls")
                elif 'impact' in risk_str:
                    mitigation_steps.append("Implement impact monitoring")
                else:
                    mitigation_steps.append(f"General mitigation needed for: {risk}")
        
        # Add category-specific mitigations
        if category == DecisionCategory.CRITICAL:
            mitigation_steps.append("Implement critical incident response plan")
        elif category == DecisionCategory.HIGH_IMPACT:
            mitigation_steps.append("Establish enhanced monitoring protocols")
        
        return mitigation_steps
    
    def evaluate_action(self, action: str, context: Dict) -> EthicalDecision:
        """Evaluate an action with probability scoring and feedback loop"""
        # Get initial decision
        initial_decision = self._make_initial_decision(action, context)
        
        # Initialize probability scorer and feedback loop
        probability_scorer = ProbabilityScorer()
        feedback_loop = FeedbackLoop()
        
        # Get initial probability score
        initial_score = probability_scorer.calculate_probability(
            initial_decision.prism_scores,
            context,
            {'compliance': context.get('compliance_requirements', [])},
            initial_decision.category.value
        )
        
        # Refine through feedback loop
        feedback_result = feedback_loop.refine_probability(
            initial_score,
            self._determine_expected_band(action, context),
            context
        )
        
        # Update decision with refined probability
        final_decision = self._update_decision_with_probability(
            initial_decision,
            feedback_result
        )
        
        # Set the probability score
        final_decision.probability_score = feedback_result.final_prediction
        
        return final_decision
    
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

    def _make_initial_decision(self, action: str, context: Dict) -> EthicalDecision:
        """Make initial decision before probability refinement"""
        # Categorize the decision
        category = self.categorize_decision(action, context)
        self.logger.info(f"Decision categorized as: {category.value}")
        
        # Get prism evaluations
        prism_scores = {}
        for name, prism in self.prisms.items():
            evaluation = prism.evaluate(action, context)
            if hasattr(evaluation, 'impact_score'):
                prism_scores[name] = evaluation.impact_score
        
        # Calculate weighted confidence score
        confidence_score = self._calculate_weighted_score(prism_scores, category)
        
        # Identify risks
        risks = self._identify_risks(action, context, prism_scores)
        
        # Generate mitigation steps
        mitigation_steps = self._generate_mitigation_steps(risks, category)
        
        # Calculate stakeholder impact
        stakeholder_impact = self._calculate_stakeholder_impact(context)
        
        # Determine recommendation
        recommendation = self._determine_recommendation(
            confidence_score,
            category,
            len(risks)
        )
        
        self.logger.info(f"Decision made: {recommendation.value} (confidence: {confidence_score:.2f})")
        
        # Create decision object
        decision = EthicalDecision(
            decision_id=str(len(self.decisions_history) + 1),
            category=category,
            recommendation=recommendation,
            confidence_score=confidence_score,
            prism_scores=prism_scores,
            context_snapshot=context,
            reasoning=[],  # Will be populated during probability refinement
            risk_factors=risks,
            mitigation_steps=mitigation_steps,
            stakeholder_impact=stakeholder_impact,
            timestamp=time.time()
        )
        
        return decision

    def _identify_risks(self, action: str, context: Dict, prism_scores: Dict[str, float]) -> List[str]:
        """Identify potential risks in the decision"""
        risks = []
        
        # Check for negative prism scores
        for prism_name, score in prism_scores.items():
            if score < -0.3:
                risks.append(f"Significant negative impact detected by {prism_name} prism")
            elif score < 0:
                risks.append(f"Minor negative impact detected by {prism_name} prism")
        
        # Check context-specific risks
        if 'privacy' in str(action).lower():
            risks.append("Privacy considerations required")
        if 'security' in str(action).lower():
            risks.append("Security implications detected")
        if context.get('urgency_level') == 'high':
            risks.append("High urgency may impact decision quality")
            
        return risks

    def _determine_expected_band(self, action: str, context: Dict) -> ProbabilityBand:
        """Determine expected probability band based on action and context"""
        # Check for privacy-focused actions
        if any(term in action.lower() for term in ['privacy', 'data', 'personal']):
            if context.get('privacy_level') == 'high' or 'GDPR' in str(context.get('compliance_requirements', [])):
                return ProbabilityBand.LOW  # High privacy concerns -> low probability
        
        # Check for innovation-focused actions
        if any(term in action.lower() for term in ['innovate', 'optimize', 'improve']):
            if context.get('innovation_impact') == 'high' or context.get('innovation_tolerance') == 'progressive':
                return ProbabilityBand.HIGH  # High innovation focus -> high probability
        
        # Check for environmental/sustainability actions
        if any(term in action.lower() for term in ['green', 'sustainable', 'efficiency']):
            if 'Green IT Standards' in str(context.get('compliance_requirements', [])):
                return ProbabilityBand.MEDIUM  # Environmental focus -> medium probability
        
        # Default to medium band if no specific conditions are met
        return ProbabilityBand.MEDIUM

    def _update_decision_with_probability(self, 
                                        decision: EthicalDecision,
                                        feedback_result: FeedbackLoopResult) -> EthicalDecision:
        """Update decision with probability refinement results"""
        # Add probability-related reasoning
        decision.reasoning.extend([
            f"Initial probability band: {feedback_result.initial_prediction.band.value}",
            f"Final probability band: {feedback_result.final_prediction.band.value}",
            f"Confidence level: {feedback_result.final_prediction.confidence_level:.2f}"
        ])
        
        # Add iteration details
        for iteration in feedback_result.iterations:
            decision.reasoning.append(
                f"Iteration {iteration.iteration_number}: {iteration.reasoning}"
            )
        
        # Update confidence score if probability assessment is more confident
        if feedback_result.final_prediction.confidence_level > decision.confidence_score:
            decision.confidence_score = feedback_result.final_prediction.confidence_level
        
        return decision