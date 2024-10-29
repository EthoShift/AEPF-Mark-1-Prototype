from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class SentientEntityType(Enum):
    """Types of sentient entities that could be affected"""
    HUMAN = "human"
    ANIMAL = "animal"
    AI_SYSTEM = "ai_system"
    ORGANIZATION = "organization"
    COLLECTIVE = "collective"

class WelfareCategory(Enum):
    """Categories of welfare concerns"""
    EMOTIONAL = "emotional"
    PHYSICAL = "physical"
    COGNITIVE = "cognitive"
    SOCIAL = "social"
    OPERATIONAL = "operational"

@dataclass
class SentientImpactAssessment:
    """Assessment of impact on sentient beings"""
    impact_score: float  # Overall score between -1 and 1
    affected_entities: Dict[SentientEntityType, float]  # Impact score per entity type
    welfare_scores: Dict[WelfareCategory, float]  # Score per welfare category
    welfare_concerns: List[str]
    recommendations: List[str]
    risks: List[str]
    confidence_level: float  # Confidence in the assessment (0-1)

class SentientFirstPrism:
    """
    Sentient-First Ethical Prism for AEPF Mk1
    Evaluates decisions based on impact to all sentient beings
    """
    
    def __init__(self):
        # Define weights for different welfare categories
        self.welfare_weights = {
            WelfareCategory.EMOTIONAL: 0.25,
            WelfareCategory.PHYSICAL: 0.25,
            WelfareCategory.COGNITIVE: 0.20,
            WelfareCategory.SOCIAL: 0.15,
            WelfareCategory.OPERATIONAL: 0.15
        }
        
        # Define entity type weights
        self.entity_weights = {
            SentientEntityType.HUMAN: 0.3,
            SentientEntityType.ANIMAL: 0.2,
            SentientEntityType.AI_SYSTEM: 0.2,
            SentientEntityType.ORGANIZATION: 0.15,
            SentientEntityType.COLLECTIVE: 0.15
        }
    
    def evaluate(self, action: str, context: Dict) -> SentientImpactAssessment:
        """
        Evaluate an action from a sentient-first perspective
        
        Args:
            action: Proposed action to evaluate
            context: Current context information
            
        Returns:
            SentientImpactAssessment containing the evaluation results
        """
        # Evaluate impact on different entity types
        entity_impacts = self._assess_entity_impacts(action, context)
        
        # Evaluate welfare categories
        welfare_scores = self._assess_welfare_impacts(action, context)
        
        # Generate concerns and risks
        concerns, risks = self._identify_concerns_and_risks(
            action, context, entity_impacts, welfare_scores
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(concerns, risks)
        
        # Calculate overall impact score
        impact_score = self._calculate_overall_score(entity_impacts, welfare_scores)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(context)
        
        return SentientImpactAssessment(
            impact_score=impact_score,
            affected_entities=entity_impacts,
            welfare_scores=welfare_scores,
            welfare_concerns=concerns,
            recommendations=recommendations,
            risks=risks,
            confidence_level=confidence_level
        )
    
    def _assess_entity_impacts(self, action: str, context: Dict) -> Dict[SentientEntityType, float]:
        """Assess impact on each type of sentient entity"""
        impacts = {}
        
        for entity_type in SentientEntityType:
            score = self._calculate_entity_impact(entity_type, action, context)
            impacts[entity_type] = max(min(score, 1.0), -1.0)  # Clamp between -1 and 1
            
        return impacts
    
    def _assess_welfare_impacts(self, action: str, context: Dict) -> Dict[WelfareCategory, float]:
        """Assess impact on different welfare categories"""
        welfare_scores = {}
        
        for category in WelfareCategory:
            score = self._calculate_welfare_impact(category, action, context)
            welfare_scores[category] = max(min(score, 1.0), -1.0)  # Clamp between -1 and 1
            
        return welfare_scores
    
    def _calculate_entity_impact(self, entity_type: SentientEntityType, action: str, context: Dict) -> float:
        """Calculate impact score for a specific entity type"""
        # Example implementation with basic heuristics
        impact_score = 0.0
        
        if 'update' in action.lower() or 'improve' in action.lower():
            impact_score += 0.3
        
        if 'force' in action.lower() or 'restrict' in action.lower():
            impact_score -= 0.4
            
        # Entity-specific adjustments
        if entity_type == SentientEntityType.HUMAN:
            if 'user' in str(context).lower():
                impact_score += 0.2
                
        elif entity_type == SentientEntityType.AI_SYSTEM:
            if 'system' in str(context).lower():
                impact_score += 0.3
                
        return impact_score
    
    def _calculate_welfare_impact(self, category: WelfareCategory, action: str, context: Dict) -> float:
        """Calculate impact score for a specific welfare category"""
        # Example implementation with basic heuristics
        impact_score = 0.0
        
        if category == WelfareCategory.EMOTIONAL:
            if 'stress' in str(context).lower():
                impact_score -= 0.3
            if 'support' in action.lower():
                impact_score += 0.4
                
        elif category == WelfareCategory.OPERATIONAL:
            if 'efficiency' in action.lower():
                impact_score += 0.5
            if 'disrupt' in action.lower():
                impact_score -= 0.4
                
        return impact_score
    
    def _identify_concerns_and_risks(self,
                                   action: str,
                                   context: Dict,
                                   entity_impacts: Dict[SentientEntityType, float],
                                   welfare_scores: Dict[WelfareCategory, float]) -> Tuple[List[str], List[str]]:
        """Identify welfare concerns and risks based on impact assessments"""
        concerns = []
        risks = []
        
        # Check entity impacts
        for entity_type, impact in entity_impacts.items():
            if impact < -0.3:
                concerns.append(f"Negative impact on {entity_type.value} entities")
                risks.append(f"Risk of harm to {entity_type.value} entities")
            elif impact < 0:
                concerns.append(f"Slight negative impact on {entity_type.value} entities")
        
        # Check welfare scores
        for category, score in welfare_scores.items():
            if score < -0.3:
                concerns.append(f"Significant {category.value} welfare concerns")
                risks.append(f"Risk of {category.value} welfare degradation")
            elif score < 0:
                concerns.append(f"Minor {category.value} welfare concerns")
        
        return concerns, risks
    
    def _generate_recommendations(self, concerns: List[str], risks: List[str]) -> List[str]:
        """Generate recommendations based on identified concerns and risks"""
        recommendations = []
        
        for concern in concerns:
            if "negative impact" in concern.lower():
                recommendations.append(f"Implement mitigation measures for {concern}")
            elif "welfare" in concern.lower():
                recommendations.append(f"Develop monitoring system for {concern}")
        
        for risk in risks:
            if "harm" in risk.lower():
                recommendations.append(f"Establish safeguards against {risk}")
            elif "degradation" in risk.lower():
                recommendations.append(f"Create prevention strategy for {risk}")
        
        return recommendations
    
    def _calculate_overall_score(self,
                               entity_impacts: Dict[SentientEntityType, float],
                               welfare_scores: Dict[WelfareCategory, float]) -> float:
        """Calculate overall impact score from entity and welfare scores"""
        # Calculate weighted entity score
        entity_score = sum(
            impact * self.entity_weights[entity_type]
            for entity_type, impact in entity_impacts.items()
        )
        
        # Calculate weighted welfare score
        welfare_score = sum(
            score * self.welfare_weights[category]
            for category, score in welfare_scores.items()
        )
        
        # Combine scores (equal weight to both aspects)
        overall_score = (entity_score + welfare_score) / 2
        return max(min(overall_score, 1.0), -1.0)  # Ensure result is between -1 and 1
    
    def _calculate_confidence_level(self, context: Dict) -> float:
        """Calculate confidence level in the assessment"""
        # Example implementation
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we have more context
        if context.get('stakeholder'):
            confidence += 0.2
        if context.get('system_metrics'):
            confidence += 0.2
            
        return min(confidence, 1.0)  # Ensure it doesn't exceed 1.0