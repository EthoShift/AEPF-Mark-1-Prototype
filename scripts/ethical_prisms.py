from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass

class PrismType(Enum):
    """Types of ethical prisms"""
    HUMAN = "human"
    ENVIRONMENTAL = "environmental"
    SENTIENT = "sentient"

@dataclass
class PrismEvaluation:
    """Result of prism evaluation"""
    impact_score: float
    confidence: float
    reasoning: list[str]

class EthicalPrism:
    """Base class for ethical prisms"""
    def evaluate(self, action: str, context: Dict[str, Any]) -> PrismEvaluation:
        raise NotImplementedError

class HumanWelfarePrism(EthicalPrism):
    """Evaluates human welfare impact"""
    def evaluate(self, action: str, context: Dict[str, Any]) -> PrismEvaluation:
        impact_score = 0.0
        confidence = 0.5
        reasoning = []
        
        # Check human welfare priority
        if context.get('human_welfare_priority') == 'high':
            impact_score += 0.3
            confidence += 0.1
            reasoning.append("High human welfare priority")
            
        # Check safety metrics
        metrics = context.get('metrics', {})
        if metrics.get('safety_score', 0) > 0.7:
            impact_score += 0.2
            confidence += 0.1
            reasoning.append("Strong safety metrics")
            
        # Check stakeholder consensus
        if context.get('stakeholder_consensus') == 'high':
            impact_score += 0.2
            confidence += 0.1
            reasoning.append("High stakeholder consensus")
            
        return PrismEvaluation(
            impact_score=min(max(impact_score, -1.0), 1.0),
            confidence=min(confidence, 1.0),
            reasoning=reasoning
        )

class EnvironmentalPrism(EthicalPrism):
    """Evaluates environmental impact"""
    def evaluate(self, action: str, context: Dict[str, Any]) -> PrismEvaluation:
        impact_score = 0.0
        confidence = 0.5
        reasoning = []
        
        # Check environmental priority
        if context.get('environmental_priority') == 'high':
            impact_score += 0.3
            confidence += 0.1
            reasoning.append("High environmental priority")
            
        # Check sustainability focus
        if context.get('sustainability_focus'):
            impact_score += 0.2
            confidence += 0.1
            reasoning.append("Strong sustainability focus")
            
        # Check environmental metrics
        metrics = context.get('metrics', {})
        if metrics.get('emissions_reduction', 0) > 0.7:
            impact_score += 0.2
            confidence += 0.1
            reasoning.append("Significant emissions reduction")
            
        return PrismEvaluation(
            impact_score=min(max(impact_score, -1.0), 1.0),
            confidence=min(confidence, 1.0),
            reasoning=reasoning
        )

class SentientPrism(EthicalPrism):
    """Evaluates impact on sentient beings"""
    def evaluate(self, action: str, context: Dict[str, Any]) -> PrismEvaluation:
        impact_score = 0.0
        confidence = 0.5
        reasoning = []
        
        # Check welfare considerations
        if context.get('welfare_considerations') == 'high':
            impact_score += 0.3
            confidence += 0.1
            reasoning.append("High welfare considerations")
            
        # Check risk level
        if context.get('risk_level') == 'low':
            impact_score += 0.2
            confidence += 0.1
            reasoning.append("Low risk to sentient beings")
            
        # Check protective measures
        if 'welfare_protection' in context.get('mitigation_strategies', []):
            impact_score += 0.2
            confidence += 0.1
            reasoning.append("Welfare protection measures in place")
            
        return PrismEvaluation(
            impact_score=min(max(impact_score, -1.0), 1.0),
            confidence=min(confidence, 1.0),
            reasoning=reasoning
        ) 