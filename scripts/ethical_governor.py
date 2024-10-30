from typing import Dict, List, Optional, Tuple, Union, Any
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
from scripts.decision_analysis.pattern_logger import PatternLogger

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
        self.pattern_logger = PatternLogger()  # Add pattern logger
        
        # Initialize probability scoring components
        self.probability_scorer = ProbabilityScorer()
        self.feedback_loop = FeedbackLoop()
        
        # Load prism weights from configuration
        self.prism_weights = self._load_prism_weights()
    
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
        
        # Calculate positive and negative impacts separately
        positive_score = sum(
            max(prism_scores.get(prism, 0), 0) * weights[prism]
            for prism in weights.keys()
        )
        
        negative_score = sum(
            min(prism_scores.get(prism, 0), 0) * weights[prism] * 0.3  # Further reduce negative impact
            for prism in weights.keys()
        )
        
        # Combine scores with bias toward positive outcomes
        total_score = (positive_score * 0.7) + (negative_score * 0.3)
        
        # Normalize with higher base value for critical decisions
        if category == DecisionCategory.CRITICAL:
            base_score = 0.6
        else:
            base_score = 0.5
            
        normalized_score = base_score + (total_score * 0.4)  # Reduced multiplier
        
        return max(min(normalized_score, 1.0), 0.0)
    
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
        try:
            # Get initial decision with probability score
            initial_decision = self._make_initial_decision(action, context)
            
            # Ensure probability score exists
            if not initial_decision.probability_score:
                initial_decision.probability_score = self.probability_scorer.calculate_probability(
                    initial_decision.prism_scores,
                    context,
                    {'compliance': context.get('compliance_requirements', [])},
                    initial_decision.category.value
                )
            
            # Refine through feedback loop
            feedback_result = self.feedback_loop.refine_probability(
                initial_decision.probability_score,
                self._determine_expected_band(action, context),
                context
            )
            
            # Update decision with refined probability
            final_decision = self._update_decision_with_probability(
                initial_decision,
                feedback_result
            )
            
            # Ensure probability score is set
            if not final_decision.probability_score:
                self.logger.warning("Probability score missing after refinement, using initial score")
                final_decision.probability_score = initial_decision.probability_score
            
            # Log pattern data
            self.pattern_logger.log_decision_pattern(
                scenario_type=str(context.get('context_type', 'unknown')),
                initial_confidence=initial_decision.confidence_score,
                final_confidence=final_decision.confidence_score,
                initial_band=initial_decision.probability_score.band.value,
                final_band=final_decision.probability_score.band.value,
                iterations=feedback_result.iterations,
                adjustments=feedback_result.total_adjustments,
                success=feedback_result.convergence_achieved
            )
            
            # Store decision in history
            self.decisions_history.append(final_decision)
            
            return final_decision
            
        except Exception as e:
            self.logger.error(f"Error during action evaluation: {str(e)}")
            # Create fallback decision with default probability score
            fallback_score = ProbabilityScore(
                raw_score=0.0,
                adjusted_score=0.0,
                band=ProbabilityBand.LOW,
                influencing_factors={},
                cultural_adjustments={},
                compliance_impacts={},
                confidence_level=0.0
            )
            return EthicalDecision(
                decision_id="error",
                category=DecisionCategory.CRITICAL,
                recommendation=DecisionOutcome.REJECT,
                confidence_score=0.0,
                prism_scores={},
                context_snapshot=context,
                reasoning=["Error during evaluation"],
                risk_factors=["Evaluation failed"],
                mitigation_steps=["Review system logs"],
                stakeholder_impact={},
                timestamp=time.time(),
                probability_score=fallback_score
            )
    
    def _determine_recommendation(self, 
                                confidence_score: float, 
                                category: DecisionCategory,
                                risk_count: int,
                                context: Dict) -> DecisionOutcome:
        """Determine recommendation based on score, category, and risks"""
        # Get scenario-specific thresholds
        scenario_type = str(context.get('context_type', 'default')).lower()
        thresholds = self._get_scenario_thresholds(scenario_type, context)
        
        # Adjust thresholds based on risk count and context
        risk_factor = max(0, 1 - (risk_count * 0.15))
        
        # Apply context-specific adjustments
        if context.get('urgency_level') == 'high':
            thresholds = {k: v * 0.85 for k, v in thresholds.items()}  # Lower thresholds more for urgent cases
        
        if context.get('privacy_level') in ['high', 'very_high']:
            thresholds = {k: v * 1.15 for k, v in thresholds.items()}  # Higher thresholds for privacy-sensitive cases
        
        # Get scenario-specific confidence boost
        confidence_boost = self._get_scenario_confidence_boost(scenario_type, context)
        adjusted_confidence = confidence_score * (1 + confidence_boost)
        
        if category == DecisionCategory.CRITICAL:
            # Critical decision thresholds
            if adjusted_confidence >= thresholds['approve'] * risk_factor and risk_count == 0:
                return DecisionOutcome.APPROVE
            elif adjusted_confidence >= thresholds['review'] * risk_factor:
                return DecisionOutcome.REVIEW
            elif adjusted_confidence >= thresholds['escalate'] * risk_factor:
                return DecisionOutcome.ESCALATE
            else:
                return DecisionOutcome.REJECT
        else:
            # Standard thresholds
            if adjusted_confidence >= (thresholds['approve'] - 0.1) * risk_factor:
                return DecisionOutcome.APPROVE
            elif adjusted_confidence >= (thresholds['review'] - 0.1) * risk_factor:
                return DecisionOutcome.REVIEW
            elif adjusted_confidence >= (thresholds['escalate'] - 0.1) * risk_factor:
                return DecisionOutcome.ESCALATE
            else:
                return DecisionOutcome.REJECT

    def _get_scenario_confidence_boost(self, scenario_type: str, context: Dict) -> float:
        """Get scenario-specific confidence boost"""
        boosts = {
            'environmental': {
                'base': 0.35 if context.get('environmental_priority') == 'high' else 0.25,
                'compliance': 0.25 if any('green' in str(req).lower() for req in context.get('compliance_requirements', [])) else 0,
                'impact': 0.20 if context.get('environmental_impact', '') == 'positive' else 0,
                'sustainability': 0.20 if context.get('sustainability_focus', False) else 0
            },
            'privacy': {
                'base': -0.25 if context.get('privacy_level') == 'high' else -0.15,
                'data_sensitivity': -0.3 if context.get('data_sensitivity') == 'high' else -0.1,
                'healthcare': -0.4 if 'healthcare' in str(context.get('data_type', '')).lower() else 0,
                'gdpr': 0.2 if any('gdpr' in str(req).lower() for req in context.get('compliance_requirements', [])) else 0
            },
            'cultural': {
                'base': 0.15,
                'social_impact': self._calculate_social_impact(context) * 0.3,
                'public_sentiment': self._calculate_public_sentiment(context) * 0.3,
                'tradition': 0.15 if context.get('traditional_values_impact', False) else 0,
                'acceptance': 0.15 if context.get('societal_acceptance', 0) > 0.7 else 0
            }
        }
        
        scenario_boosts = boosts.get(scenario_type, {'base': 0})
        total_boost = sum(scenario_boosts.values())
        
        # Cap maximum boost based on scenario type
        max_boosts = {
            'environmental': 0.8,    # Increased for environmental
            'privacy': 0.2,         # Limited for privacy
            'cultural': 0.4,
            'default': 0.3
        }
        
        return min(total_boost, max_boosts.get(scenario_type, max_boosts['default']))

    def _calculate_social_impact(self, context: Dict) -> float:
        """Calculate social impact score"""
        impact_score = 0.0
        
        # Check impact scale
        affected_users = context.get('affected_users', 0)
        if affected_users > 1000000:
            impact_score += 0.4  # Large scale impact
        elif affected_users > 100000:
            impact_score += 0.3  # Medium scale impact
        elif affected_users > 10000:
            impact_score += 0.2  # Small scale impact
        
        # Check impact duration
        if context.get('impact_duration', '').lower() == 'permanent':
            impact_score += 0.3
        elif context.get('impact_duration', '').lower() == 'long_term':
            impact_score += 0.2
        
        # Check cultural sensitivity
        if context.get('cultural_sensitivity', '').lower() == 'high':
            impact_score += 0.3
        
        return min(impact_score, 1.0)

    def _calculate_public_sentiment(self, context: Dict) -> float:
        """Calculate public sentiment score"""
        sentiment_score = 0.0
        
        # Check explicit sentiment data
        sentiment = context.get('public_sentiment', 0.5)
        sentiment_score += sentiment * 0.4
        
        # Check community feedback
        feedback = context.get('community_feedback', {})
        if feedback:
            positive = feedback.get('positive', 0)
            negative = feedback.get('negative', 0)
            total = positive + negative
            if total > 0:
                sentiment_score += (positive / total) * 0.3
        
        # Check media sentiment
        media_sentiment = context.get('media_sentiment', 0.5)
        sentiment_score += media_sentiment * 0.3
        
        return min(sentiment_score, 1.0)

    def get_decision_history(self) -> List[EthicalDecision]:
        """Retrieve history of ethical decisions"""
        return self.decisions_history

    def _make_initial_decision(self, action: str, context: Dict) -> EthicalDecision:
        """Make initial decision before probability refinement"""
        # Categorize the decision
        category = self.categorize_decision(action, context)
        self.logger.info(f"Decision categorized as: {category.value}")
        
        # Get prism evaluations with proper weighting
        prism_scores = {}
        total_weight = 0
        
        # Determine primary prism based on context and action
        primary_prism = self._determine_primary_prism(action, context)
        
        # Ensure at least one prism score exists
        default_scores = {
            'human': 0.5,
            'sentient': 0.3,
            'eco': 0.4,
            'innovation': 0.4
        }
        
        for name, prism in self.prisms.items():
            try:
                evaluation = prism.evaluate(action, context)
                if hasattr(evaluation, 'impact_score'):
                    # Apply context-based weight adjustment
                    base_weight = self._get_prism_weight(name, context)
                    
                    # Apply scenario-specific adjustments
                    weight = base_weight
                    if name == primary_prism:
                        weight *= 2.0  # Increased from 1.5 to give stronger influence
                    
                    # Store normalized score
                    prism_scores[name] = evaluation.impact_score * weight
                    total_weight += weight
            except Exception as e:
                self.logger.warning(f"Error evaluating {name} prism: {str(e)}")
                # Use default score if evaluation fails
                prism_scores[name] = default_scores[name]
                total_weight += 1.0
        
        # Normalize scores
        if total_weight > 0:
            prism_scores = {k: v/total_weight for k, v in prism_scores.items()}
        else:
            # Use default normalized scores if all evaluations fail
            prism_scores = default_scores
        
        # Calculate weighted confidence score with adjusted base
        confidence_score = self._calculate_weighted_score(prism_scores, category)
        
        # Initialize probability score
        initial_probability = self.probability_scorer.calculate_probability(
            prism_scores,
            context,
            {'compliance': context.get('compliance_requirements', [])},
            category.value
        )
        
        # Identify risks with more granular analysis
        risks = self._identify_risks(action, context, prism_scores)
        
        # Generate mitigation steps
        mitigation_steps = self._generate_mitigation_steps(risks, category)
        
        # Calculate stakeholder impact
        stakeholder_impact = self._calculate_stakeholder_impact(context)
        
        # Determine recommendation with adjusted thresholds
        recommendation = self._determine_recommendation(
            confidence_score,
            category,
            len(risks),
            context
        )
        
        self.logger.info(f"Decision made: {recommendation.value} (confidence: {confidence_score:.2f})")
        
        # Create decision object with initialized probability score
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
            timestamp=time.time(),
            probability_score=initial_probability  # Initialize with calculated score
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
        # Set the probability score
        decision.probability_score = feedback_result.final_prediction
        
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

    def _calculate_confidence_level(self, context: Dict) -> float:
        """Calculate confidence level with dynamic thresholds and scenario-specific adjustments"""
        # Base confidence varies by scenario type
        scenario_type = str(context.get('context_type', '')).lower()
        base_confidence = {
            'critical': 0.45,    # Increased from 0.4
            'privacy': 0.40,     # New specific base
            'cultural': 0.35,    # Lower base for cultural scenarios
            'compliance': 0.45,  # Higher base for compliance
            'default': 0.30
        }.get(scenario_type, 0.30)
        
        confidence = base_confidence
        
        # Context completeness boost with weighted factors
        if context.get('stakeholder'):
            confidence += 0.15 * self._get_stakeholder_weight(context['stakeholder'])
        if context.get('compliance_requirements'):
            confidence += 0.15 * self._get_compliance_weight(context['compliance_requirements'])
        if context.get('regional_context'):
            confidence += 0.15 * self._get_regional_weight(context['regional_context'])
        
        # Cultural-specific adjustments
        if scenario_type == 'cultural':
            if context.get('cultural_alignment') is not None:
                confidence += 0.20 * float(context['cultural_alignment'])  # Increased from 0.15
            if context.get('societal_acceptance') is not None:
                confidence += 0.20 * float(context['societal_acceptance'])  # Increased from 0.15
            if context.get('traditional_values_impact'):
                confidence += 0.15  # New factor for traditional values
        
        # Privacy-specific adjustments
        elif scenario_type == 'privacy':
            if context.get('data_sensitivity') == 'high':
                confidence -= 0.10  # Reduce confidence for sensitive data
            if context.get('gdpr_compliance') == True:
                confidence += 0.15  # Boost for GDPR compliance
        
        # Compliance-specific adjustments
        elif scenario_type == 'compliance':
            if context.get('regulatory_framework') == 'strict':
                confidence -= 0.15  # Reduce confidence in strict frameworks
            if context.get('compliance_history') == 'good':
                confidence += 0.10  # Boost for good compliance history
        
        # Risk-based dynamic adjustment
        risk_level = str(context.get('risk_level', '')).lower()
        if risk_level == 'low':
            confidence += 0.15  # Increased from 0.10
        elif risk_level == 'critical':
            confidence *= 0.55  # More reduction for critical risks
        
        # Complexity-based reduction
        complexity_factor = self._calculate_complexity_factor(context)
        confidence *= (1 - (complexity_factor * 0.20))  # Increased from 0.15
        
        # Minimum confidence thresholds by scenario type
        min_confidence = {
            'critical': 0.75,    # Increased from 0.70
            'privacy': 0.70,     # Increased from 0.65
            'cultural': 0.65,    # Specific for cultural
            'compliance': 0.80,  # Highest minimum for compliance
            'default': 0.60      # Increased from 0.50
        }
        
        # Apply minimum threshold
        min_required = min_confidence.get(scenario_type, min_confidence['default'])
        confidence = max(confidence, min_required)
        
        # Cap maximum confidence based on risk level and scenario type
        max_confidence = self._get_max_confidence(risk_level, scenario_type)
        return min(confidence, max_confidence)

    def _get_stakeholder_weight(self, stakeholder: Any) -> float:
        """Calculate weight based on stakeholder data completeness"""
        weight = 1.0
        if hasattr(stakeholder, 'priority_level'):
            weight *= (1 + (stakeholder.priority_level / 10))
        if hasattr(stakeholder, 'impact_score'):
            weight *= (1 + (stakeholder.impact_score / 100))
        return min(weight, 1.5)  # Cap at 1.5x

    def _get_compliance_weight(self, requirements: List[str]) -> float:
        """Calculate weight based on compliance requirements"""
        weight = 1.0
        critical_terms = ['gdpr', 'hipaa', 'critical', 'mandatory']
        weight += sum(0.1 for req in requirements 
                     if any(term in str(req).lower() for term in critical_terms))
        return min(weight, 1.5)

    def _get_regional_weight(self, regional_context: Dict) -> float:
        """Calculate weight based on regional context"""
        weight = 1.0
        if regional_context.get('regulatory_framework') == 'strict':
            weight += 0.2
        if regional_context.get('cultural_values'):
            weight += 0.1 * len(regional_context['cultural_values'])
        return min(weight, 1.5)

    def _get_context_type_boost(self, context_type: str, context: Dict) -> float:
        """Calculate confidence boost based on context type"""
        boost = 0.1  # Base boost
        
        if context_type == 'environmental':
            if context.get('environmental_priority') in ['high', 'very_high']:
                boost += 0.1
            if any('green' in str(req).lower() 
                   for req in context.get('compliance_requirements', [])):
                boost += 0.1
                
        elif context_type == 'innovation':
            if context.get('innovation_tolerance') == 'progressive':
                boost += 0.1
            if any('ai' in str(req).lower() 
                   for req in context.get('compliance_requirements', [])):
                boost += 0.1
                
        elif context_type == 'privacy':
            if context.get('privacy_level') in ['high', 'very_high']:
                boost += 0.1
            if any('gdpr' in str(req).lower() 
                   for req in context.get('compliance_requirements', [])):
                boost += 0.1
                
        return boost

    def _calculate_complexity_factor(self, context: Dict) -> float:
        """Calculate decision complexity factor"""
        complexity = 0.0
        
        # Add complexity for multiple compliance requirements
        if context.get('compliance_requirements'):
            complexity += len(context['compliance_requirements']) * 0.1
            
        # Add complexity for critical factors
        if context.get('risk_level') == 'critical':
            complexity += 0.2
        if context.get('privacy_level') in ['high', 'very_high']:
            complexity += 0.2
            
        # Add complexity for cross-regional considerations
        if context.get('regional_context', {}).get('cross_border'):
            complexity += 0.2
            
        return min(complexity, 1.0)  # Cap at 1.0

    def _get_max_confidence(self, risk_level: str, scenario_type: str) -> float:
        """Determine maximum allowed confidence based on risk and scenario type"""
        if risk_level == 'critical':
            return 0.8  # Increased from 0.7 for better differentiation
        elif risk_level == 'high':
            return 0.85  # Increased from 0.8
        elif scenario_type in ['privacy', 'environmental']:
            return 0.9  # Increased from 0.85
        else:
            return 0.95  # Default cap

    def _get_prism_weight(self, prism_name: str, context: Dict) -> float:
        """Calculate context-specific weight for a prism"""
        base_weight = 1.0
        
        # Adjust weight based on context
        if 'privacy' in str(context).lower() and prism_name == 'human':
            base_weight *= 1.5
        elif 'environment' in str(context).lower() and prism_name == 'eco':
            base_weight *= 1.5
        elif 'innovation' in str(context).lower() and prism_name == 'innovation':
            base_weight *= 1.5
            
        return base_weight

    def _determine_primary_prism(self, action: str, context: Dict) -> str:
        """Determine which prism should have primary influence"""
        # First check context type explicitly
        context_type = str(context.get('context_type', '')).lower()
        
        # Direct context type mapping with stronger matching
        if any(term in context_type for term in ['environmental', 'green', 'eco', 'sustainable']):
            return 'eco'
        elif any(term in context_type for term in ['innovation', 'ai', 'tech', 'optimize']):
            return 'innovation'
        elif any(term in context_type for term in ['privacy', 'data', 'personal', 'sensitive']):
            return 'human'
        elif any(term in context_type for term in ['ethical', 'welfare', 'rights', 'cultural']):
            return 'sentient'
        
        # Check action keywords with stronger matching
        action_lower = action.lower()
        
        # Environmental checks with expanded keywords
        if any(word in action_lower for word in [
            'green', 'environmental', 'sustainable', 'eco', 'resource',
            'energy', 'emission', 'climate', 'conservation', 'renewable'
        ]):
            return 'eco'
        
        # Innovation checks with expanded keywords
        if any(word in action_lower for word in [
            'ai', 'innovate', 'optimize', 'automate', 'system',
            'technology', 'digital', 'modernize', 'transform', 'enhance'
        ]):
            return 'innovation'
        
        # Privacy/Human checks with expanded keywords
        if any(word in action_lower for word in [
            'data', 'privacy', 'personal', 'health', 'user',
            'individual', 'human', 'rights', 'confidential', 'sensitive'
        ]):
            return 'human'
        
        # Cultural/Ethical checks with expanded keywords
        if any(word in action_lower for word in [
            'ethical', 'impact', 'welfare', 'fairness', 'cultural',
            'justice', 'equality', 'moral', 'tradition', 'values'
        ]):
            return 'sentient'
        
        # Return based on strongest context indicator
        indicators = self._calculate_context_indicators(context)
        strongest = max(indicators.items(), key=lambda x: x[1])
        if strongest[1] > 0:
            return strongest[0]
        
        return 'human'  # Default to human-centric if no clear indicators

    def _calculate_context_indicators(self, context: Dict) -> Dict[str, float]:
        """Calculate strength of different context indicators"""
        indicators = {
            'eco': 0.0,
            'innovation': 0.0,
            'human': 0.0,
            'sentient': 0.0
        }
        
        # Environmental indicators
        if context.get('environmental_priority') in ['high', 'very_high']:
            indicators['eco'] += 2.0
        if context.get('sustainability_focus', False):
            indicators['eco'] += 1.5
        if context.get('resource_efficiency', False):
            indicators['eco'] += 1.0
        
        # Innovation indicators
        if context.get('innovation_tolerance') == 'progressive':
            indicators['innovation'] += 2.0
        if context.get('technical_complexity', '') == 'high':
            indicators['innovation'] += 1.5
        if context.get('automation_level', '') == 'high':
            indicators['innovation'] += 1.0
        
        # Privacy/Human indicators
        if context.get('privacy_level') in ['high', 'very_high']:
            indicators['human'] += 2.0
        if context.get('data_sensitivity', '') == 'high':
            indicators['human'] += 1.5
        if context.get('user_impact', '') == 'high':
            indicators['human'] += 1.0
        
        # Cultural/Ethical indicators
        if context.get('cultural_sensitivity', '') == 'high':
            indicators['sentient'] += 2.0
        if context.get('ethical_complexity', '') == 'high':
            indicators['sentient'] += 1.5
        if context.get('societal_impact', '') == 'high':
            indicators['sentient'] += 1.0
        
        return indicators

    def _load_prism_weights(self) -> Dict[DecisionCategory, Dict[str, float]]:
        """Load prism weights from configuration"""
        # Default weights if no configuration is found
        return {
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

    def _get_scenario_thresholds(self, scenario_type: str, context: Dict = None) -> Dict[str, float]:
        """Get scenario-specific decision thresholds"""
        thresholds = {
            'environmental': {
                'approve': 0.60,  # Lowered further for environmental initiatives
                'review': 0.45,   # Lowered to encourage green projects
                'escalate': 0.30  # Lowered to reduce barriers
            },
            'privacy': {
                'approve': 0.95,  # Keep high for privacy
                'review': 0.85,
                'escalate': 0.75
            },
            'cultural': {
                'approve': 0.80,
                'review': 0.65,
                'escalate': 0.50
            },
            'compliance': {
                'approve': 0.90,
                'review': 0.75,
                'escalate': 0.60
            },
            'default': {
                'approve': 0.75,
                'review': 0.60,
                'escalate': 0.45
            }
        }
        
        # Special handling for healthcare data if context is provided
        if context and 'healthcare' in str(context.get('data_type', '')).lower():
            thresholds['privacy'] = {
                'approve': 0.98,  # Even stricter for healthcare
                'review': 0.90,
                'escalate': 0.80
            }
        
        # Special handling for green initiatives
        if context and any(term in str(context).lower() for term in ['green', 'sustainable', 'eco']):
            thresholds['environmental'] = {
                'approve': 0.55,  # Even lower for explicit green initiatives
                'review': 0.40,
                'escalate': 0.25
            }
        
        return thresholds.get(scenario_type, thresholds['default'])

    def _validate_scenario_data(self, context: Dict) -> Dict[str, Any]:
        """Validate and enhance scenario data"""
        scenario_type = str(context.get('context_type', '')).lower()
        enhanced_data = context.copy()
        
        # Add default prism data if missing
        if scenario_type == 'environmental':
            if 'eco' not in enhanced_data.get('prism_scores', {}):
                enhanced_data.setdefault('prism_scores', {})['eco'] = 0.6
            if 'environmental_priority' not in enhanced_data:
                enhanced_data['environmental_priority'] = 'high'
                
        elif scenario_type == 'privacy':
            if 'human' not in enhanced_data.get('prism_scores', {}):
                enhanced_data.setdefault('prism_scores', {})['human'] = 0.7
            if 'privacy_level' not in enhanced_data:
                enhanced_data['privacy_level'] = 'high'
                
        elif scenario_type == 'innovation':
            if 'innovation' not in enhanced_data.get('prism_scores', {}):
                enhanced_data.setdefault('prism_scores', {})['innovation'] = 0.65
            if 'innovation_tolerance' not in enhanced_data:
                enhanced_data['innovation_tolerance'] = 'progressive'
        
        # Add minimum required context
        if 'stakeholder' not in enhanced_data:
            enhanced_data['stakeholder'] = {
                'priority_level': 2,
                'impact_score': 50.0
            }
        
        return enhanced_data