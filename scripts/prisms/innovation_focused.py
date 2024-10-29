from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class InnovationType(Enum):
    """Types of innovation to consider"""
    TECHNICAL = "technical"
    PROCESS = "process"
    SOCIAL = "social"
    ETHICAL = "ethical"
    ENVIRONMENTAL = "environmental"

class RiskLevel(Enum):
    """Risk levels for innovation assessment"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class InnovationImpactAssessment:
    """Assessment of impact on innovation and progress"""
    impact_score: float  # Overall score between -1 and 1
    innovation_scores: Dict[InnovationType, float]  # Score per innovation type
    opportunities: List[str]
    risks: List[Dict[str, str]]  # List of {description, level, mitigation}
    recommendations: List[str]
    advancement_potential: float  # Score between 0 and 1
    risk_level: RiskLevel
    confidence_level: float  # Confidence in assessment (0-1)

class InnovationFocusedPrism:
    """
    Innovation-Focused Ethical Prism for AEPF Mk1
    Evaluates decisions based on potential for progress
    """
    
    def __init__(self):
        # Define weights for different innovation types
        self.innovation_weights = {
            InnovationType.TECHNICAL: 0.25,
            InnovationType.PROCESS: 0.20,
            InnovationType.SOCIAL: 0.20,
            InnovationType.ETHICAL: 0.20,
            InnovationType.ENVIRONMENTAL: 0.15
        }
        
        # Risk thresholds for different innovation types
        self.risk_thresholds = {
            InnovationType.TECHNICAL: 0.7,
            InnovationType.PROCESS: 0.6,
            InnovationType.SOCIAL: 0.8,
            InnovationType.ETHICAL: 0.9,
            InnovationType.ENVIRONMENTAL: 0.8
        }
    
    def evaluate(self, action: str, context: Dict) -> InnovationImpactAssessment:
        """
        Evaluate an action from an innovation-focused perspective
        
        Args:
            action: Proposed action to evaluate
            context: Current context information
            
        Returns:
            InnovationImpactAssessment containing the evaluation results
        """
        # Evaluate innovation potential for each type
        innovation_scores = self._assess_innovation_potential(action, context)
        
        # Identify opportunities and risks
        opportunities = self._identify_opportunities(action, context, innovation_scores)
        risks = self._assess_risks(action, context, innovation_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            innovation_scores, opportunities, risks
        )
        
        # Calculate overall scores
        impact_score = self._calculate_overall_score(innovation_scores)
        advancement_potential = self._calculate_advancement_potential(
            innovation_scores, len(risks)
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(risks, impact_score)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(context)
        
        return InnovationImpactAssessment(
            impact_score=impact_score,
            innovation_scores=innovation_scores,
            opportunities=opportunities,
            risks=risks,
            recommendations=recommendations,
            advancement_potential=advancement_potential,
            risk_level=risk_level,
            confidence_level=confidence_level
        )
    
    def _assess_innovation_potential(self, action: str, context: Dict) -> Dict[InnovationType, float]:
        """Assess innovation potential for each type"""
        scores = {}
        
        for innovation_type in InnovationType:
            score = self._calculate_type_score(innovation_type, action, context)
            scores[innovation_type] = max(min(score, 1.0), -1.0)
            
        return scores
    
    def _calculate_type_score(self, innovation_type: InnovationType, action: str, context: Dict) -> float:
        """Calculate score for specific innovation type"""
        score = 0.0
        
        if innovation_type == InnovationType.TECHNICAL:
            if 'automate' in action.lower() or 'optimize' in action.lower():
                score += 0.6
            if 'improve' in action.lower() or 'enhance' in action.lower():
                score += 0.4
            if 'legacy' in action.lower():
                score -= 0.3
                
        elif innovation_type == InnovationType.PROCESS:
            if 'streamline' in action.lower() or 'efficiency' in action.lower():
                score += 0.5
            if 'standardize' in action.lower():
                score += 0.3
            if 'complex' in action.lower():
                score -= 0.2
                
        elif innovation_type == InnovationType.SOCIAL:
            if 'collaborate' in action.lower() or 'share' in action.lower():
                score += 0.7
            if 'restrict' in action.lower():
                score -= 0.4
                
        elif innovation_type == InnovationType.ETHICAL:
            if 'responsible' in action.lower() or 'ethical' in action.lower():
                score += 0.8
            if 'risk' in action.lower():
                score -= 0.3
                
        elif innovation_type == InnovationType.ENVIRONMENTAL:
            if 'sustainable' in action.lower() or 'green' in action.lower():
                score += 0.6
            if 'consume' in action.lower():
                score -= 0.4
                
        return score
    
    def _identify_opportunities(self,
                              action: str,
                              context: Dict,
                              innovation_scores: Dict[InnovationType, float]) -> List[str]:
        """Identify innovation opportunities"""
        opportunities = []
        
        for innovation_type, score in innovation_scores.items():
            if score > 0.6:
                opportunities.append(f"High potential for {innovation_type.value} innovation")
            elif score > 0.3:
                opportunities.append(f"Moderate potential for {innovation_type.value} advancement")
        
        # Context-specific opportunities
        if context.get('system_metrics'):
            opportunities.append("Potential for metrics-driven optimization")
        
        return opportunities
    
    def _assess_risks(self,
                     action: str,
                     context: Dict,
                     innovation_scores: Dict[InnovationType, float]) -> List[Dict[str, str]]:
        """Assess risks associated with innovation"""
        risks = []
        
        for innovation_type, score in innovation_scores.items():
            threshold = self.risk_thresholds[innovation_type]
            if score > threshold:
                risks.append({
                    'description': f"High-impact {innovation_type.value} change",
                    'level': RiskLevel.HIGH.value,
                    'mitigation': f"Implement staged rollout for {innovation_type.value} changes"
                })
        
        # Check for specific risk patterns
        if 'automate' in action.lower():
            risks.append({
                'description': "Automation complexity risk",
                'level': RiskLevel.MODERATE.value,
                'mitigation': "Develop comprehensive testing framework"
            })
        
        return risks
    
    def _generate_recommendations(self,
                                innovation_scores: Dict[InnovationType, float],
                                opportunities: List[str],
                                risks: List[Dict[str, str]]) -> List[str]:
        """Generate recommendations based on assessment"""
        recommendations = []
        
        # Add opportunity-based recommendations
        for opportunity in opportunities:
            if "high potential" in opportunity.lower():
                recommendations.append(f"Prioritize and accelerate: {opportunity}")
            else:
                recommendations.append(f"Explore and develop: {opportunity}")
        
        # Add risk-mitigation recommendations
        for risk in risks:
            recommendations.append(f"Risk mitigation: {risk['mitigation']}")
        
        # Add general recommendations
        if any(score > 0.7 for score in innovation_scores.values()):
            recommendations.append("Establish innovation metrics and monitoring")
        
        return recommendations
    
    def _calculate_overall_score(self, innovation_scores: Dict[InnovationType, float]) -> float:
        """Calculate overall innovation impact score"""
        weighted_sum = sum(
            score * self.innovation_weights[innovation_type]
            for innovation_type, score in innovation_scores.items()
        )
        return max(min(weighted_sum, 1.0), -1.0)
    
    def _calculate_advancement_potential(self,
                                      innovation_scores: Dict[InnovationType, float],
                                      risk_count: int) -> float:
        """Calculate potential for advancement considering risks"""
        base_potential = sum(innovation_scores.values()) / len(innovation_scores)
        risk_factor = max(0, 1 - (risk_count * 0.1))  # Reduce potential based on risk count
        return max(min(base_potential * risk_factor, 1.0), 0.0)
    
    def _determine_risk_level(self, risks: List[Dict[str, str]], impact_score: float) -> RiskLevel:
        """Determine overall risk level"""
        high_risks = sum(1 for risk in risks if risk['level'] == RiskLevel.HIGH.value)
        
        if high_risks >= 3 or impact_score > 0.9:
            return RiskLevel.CRITICAL
        elif high_risks >= 2 or impact_score > 0.7:
            return RiskLevel.HIGH
        elif high_risks >= 1 or impact_score > 0.5:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _calculate_confidence_level(self, context: Dict) -> float:
        """Calculate confidence level in the assessment"""
        confidence = 0.5  # Base confidence
        
        if context.get('system_metrics'):
            confidence += 0.2
        if context.get('stakeholder'):
            confidence += 0.2
        if context.get('historical_data'):
            confidence += 0.1
            
        return min(confidence, 1.0)