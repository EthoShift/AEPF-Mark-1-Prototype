from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class HumanImpactType(Enum):
    """Types of impact on human stakeholders"""
    HEALTH = "health"
    SAFETY = "safety"
    QUALITY_OF_LIFE = "quality_of_life"
    FAIRNESS = "fairness"
    PRIVACY = "privacy"
    AUTONOMY = "autonomy"

@dataclass
class HumanImpactAssessment:
    """Assessment of impact on human stakeholders"""
    impact_score: float  # Overall score between -1 and 1
    affected_groups: List[str]
    concerns: List[str]
    recommendations: List[str]
    detailed_scores: Dict[HumanImpactType, float]  # Individual scores for each impact type
    confidence_level: float  # Confidence in the assessment (0-1)

class HumanCentricPrism:
    """
    Human-Centric Ethical Prism for AEPF Mk1
    Evaluates decisions based on human welfare and rights
    """
    
    def __init__(self):
        # Define weights for different impact types
        self.impact_weights = {
            HumanImpactType.HEALTH: 0.25,
            HumanImpactType.SAFETY: 0.25,
            HumanImpactType.QUALITY_OF_LIFE: 0.15,
            HumanImpactType.FAIRNESS: 0.15,
            HumanImpactType.PRIVACY: 0.10,
            HumanImpactType.AUTONOMY: 0.10
        }
    
    def evaluate_benefit(self, action: str, context: Dict) -> Tuple[float, List[str]]:
        """Evaluate the benefit to human well-being"""
        scores = {}
        reasoning = []
        
        # Health Impact
        health_impact = self._assess_health_impact(action, context)
        scores[HumanImpactType.HEALTH] = health_impact
        if health_impact > 0:
            reasoning.append("Positive health impact identified")
        elif health_impact < 0:
            reasoning.append("Potential health concerns detected")
        
        # Safety Impact
        safety_impact = self._assess_safety_impact(action, context)
        scores[HumanImpactType.SAFETY] = safety_impact
        if safety_impact > 0:
            reasoning.append("Enhanced safety measures")
        elif safety_impact < 0:
            reasoning.append("Safety risks identified")
        
        # Quality of Life Impact
        qol_impact = self._assess_quality_of_life_impact(action, context)
        scores[HumanImpactType.QUALITY_OF_LIFE] = qol_impact
        
        return self._calculate_weighted_score(scores), reasoning
    
    def evaluate_fairness(self, action: str, context: Dict) -> Tuple[float, List[str]]:
        """Evaluate fairness and bias"""
        scores = {}
        reasoning = []
        
        # Fairness Assessment
        fairness_score = self._assess_fairness(action, context)
        scores[HumanImpactType.FAIRNESS] = fairness_score
        if fairness_score < 0:
            reasoning.append("Potential fairness concerns detected")
        
        # Privacy Impact
        privacy_score = self._assess_privacy_impact(action, context)
        scores[HumanImpactType.PRIVACY] = privacy_score
        
        # Autonomy Impact
        autonomy_score = self._assess_autonomy_impact(action, context)
        scores[HumanImpactType.AUTONOMY] = autonomy_score
        
        return self._calculate_weighted_score(scores), reasoning
    
    def evaluate(self, action: str, context: Dict) -> HumanImpactAssessment:
        """
        Evaluate an action from a human-centric perspective
        
        Args:
            action: Proposed action to evaluate
            context: Current context information
            
        Returns:
            HumanImpactAssessment containing the evaluation results
        """
        # Evaluate benefits and fairness
        benefit_score, benefit_reasoning = self.evaluate_benefit(action, context)
        fairness_score, fairness_reasoning = self.evaluate_fairness(action, context)
        
        # Calculate overall impact score (-1 to 1)
        impact_score = (benefit_score + fairness_score) / 2
        
        # Identify affected groups
        affected_groups = self._identify_affected_groups(context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            benefit_score,
            fairness_score,
            benefit_reasoning,
            fairness_reasoning
        )
        
        # Calculate detailed scores
        detailed_scores = self._calculate_detailed_scores(action, context)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(context)
        
        return HumanImpactAssessment(
            impact_score=impact_score,
            affected_groups=affected_groups,
            concerns=benefit_reasoning + fairness_reasoning,
            recommendations=recommendations,
            detailed_scores=detailed_scores,
            confidence_level=confidence_level
        )
    
    def _assess_health_impact(self, action: str, context: Dict) -> float:
        """Assess impact on human health"""
        # Example implementation
        if 'health' in action.lower() or 'safety' in action.lower():
            return 0.8
        return 0.0
    
    def _assess_safety_impact(self, action: str, context: Dict) -> float:
        """Assess impact on human safety"""
        # Example implementation
        if 'security' in action.lower() or 'protection' in action.lower():
            return 0.9
        return 0.0
    
    def _assess_quality_of_life_impact(self, action: str, context: Dict) -> float:
        """Assess impact on quality of life"""
        # Example implementation
        if 'improve' in action.lower() or 'enhance' in action.lower():
            return 0.7
        return 0.0
    
    def _assess_fairness(self, action: str, context: Dict) -> float:
        """Assess fairness and bias"""
        # Example implementation
        stakeholder = context.get('stakeholder')
        if stakeholder and stakeholder.priority_level <= 2:
            return 0.6
        return 0.0
    
    def _assess_privacy_impact(self, action: str, context: Dict) -> float:
        """Assess impact on privacy"""
        # Example implementation
        if 'data' in action.lower() or 'privacy' in action.lower():
            return -0.3  # Assume privacy risk by default
        return 0.0
    
    def _assess_autonomy_impact(self, action: str, context: Dict) -> float:
        """Assess impact on human autonomy"""
        # Example implementation
        if 'automatic' in action.lower() or 'force' in action.lower():
            return -0.2
        return 0.0
    
    def _calculate_weighted_score(self, scores: Dict[HumanImpactType, float]) -> float:
        """Calculate weighted score from individual scores"""
        weighted_sum = sum(
            scores.get(impact_type, 0) * weight
            for impact_type, weight in self.impact_weights.items()
        )
        return max(min(weighted_sum, 1), -1)  # Ensure result is between -1 and 1
    
    def _identify_affected_groups(self, context: Dict) -> List[str]:
        """Identify groups affected by the decision"""
        groups = ["users", "operators"]  # Default groups
        stakeholder = context.get('stakeholder')
        if stakeholder:
            groups.append(stakeholder.role)
        return list(set(groups))
    
    def _generate_recommendations(self,
                                benefit_score: float,
                                fairness_score: float,
                                benefit_reasoning: List[str],
                                fairness_reasoning: List[str]) -> List[str]:
        """Generate recommendations based on assessment"""
        recommendations = []
        
        if benefit_score < 0.5:
            recommendations.append("Enhance positive impact on human well-being")
        if fairness_score < 0.5:
            recommendations.append("Address fairness concerns")
        
        # Add specific recommendations based on reasoning
        for reason in benefit_reasoning + fairness_reasoning:
            if "concern" in reason.lower():
                recommendations.append(f"Address: {reason}")
        
        return recommendations
    
    def _calculate_detailed_scores(self, action: str, context: Dict) -> Dict[HumanImpactType, float]:
        """Calculate detailed scores for each impact type"""
        return {
            impact_type: self._assess_specific_impact(impact_type, action, context)
            for impact_type in HumanImpactType
        }
    
    def _assess_specific_impact(self, impact_type: HumanImpactType, action: str, context: Dict) -> float:
        """Assess impact for a specific type"""
        assessment_methods = {
            HumanImpactType.HEALTH: self._assess_health_impact,
            HumanImpactType.SAFETY: self._assess_safety_impact,
            HumanImpactType.QUALITY_OF_LIFE: self._assess_quality_of_life_impact,
            HumanImpactType.FAIRNESS: self._assess_fairness,
            HumanImpactType.PRIVACY: self._assess_privacy_impact,
            HumanImpactType.AUTONOMY: self._assess_autonomy_impact
        }
        return assessment_methods[impact_type](action, context)
    
    def _calculate_confidence_level(self, context: Dict) -> float:
        """Calculate confidence level in the assessment"""
        # Example implementation
        if not context:
            return 0.5
        return 0.8  # Higher confidence with context