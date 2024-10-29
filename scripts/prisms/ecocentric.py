from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class EcosystemType(Enum):
    """Types of ecosystems that could be affected"""
    DIGITAL = "digital"
    PHYSICAL = "physical"
    ENERGY = "energy"
    DATA = "data"
    NETWORK = "network"

class EnvironmentalFactor(Enum):
    """Environmental factors to consider"""
    ENERGY_USAGE = "energy_usage"
    RESOURCE_CONSUMPTION = "resource_consumption"
    WASTE_GENERATION = "waste_generation"
    EMISSIONS = "emissions"
    BIODIVERSITY = "biodiversity"
    SUSTAINABILITY = "sustainability"

@dataclass
class EcologicalImpactAssessment:
    """Assessment of environmental impact"""
    impact_score: float  # Overall score between -1 and 1
    affected_ecosystems: Dict[EcosystemType, float]  # Impact score per ecosystem
    environmental_scores: Dict[EnvironmentalFactor, float]  # Score per factor
    environmental_concerns: List[str]
    sustainability_recommendations: List[str]
    risks: List[str]
    mitigation_strategies: List[str]
    confidence_level: float  # Confidence in the assessment (0-1)

class EcocentricPrism:
    """
    Ecocentric Ethical Prism for AEPF Mk1
    Evaluates decisions based on environmental impact
    """
    
    def __init__(self):
        # Define weights for different environmental factors
        self.factor_weights = {
            EnvironmentalFactor.ENERGY_USAGE: 0.25,
            EnvironmentalFactor.RESOURCE_CONSUMPTION: 0.20,
            EnvironmentalFactor.WASTE_GENERATION: 0.15,
            EnvironmentalFactor.EMISSIONS: 0.15,
            EnvironmentalFactor.BIODIVERSITY: 0.15,
            EnvironmentalFactor.SUSTAINABILITY: 0.10
        }
        
        # Define ecosystem weights
        self.ecosystem_weights = {
            EcosystemType.DIGITAL: 0.25,
            EcosystemType.PHYSICAL: 0.25,
            EcosystemType.ENERGY: 0.20,
            EcosystemType.DATA: 0.15,
            EcosystemType.NETWORK: 0.15
        }
    
    def evaluate(self, action: str, context: Dict) -> EcologicalImpactAssessment:
        """
        Evaluate an action from an ecocentric perspective
        
        Args:
            action: Proposed action to evaluate
            context: Current context information
            
        Returns:
            EcologicalImpactAssessment containing the evaluation results
        """
        # Evaluate impact on different ecosystems
        ecosystem_impacts = self._assess_ecosystem_impacts(action, context)
        
        # Evaluate environmental factors
        environmental_scores = self._assess_environmental_factors(action, context)
        
        # Generate concerns and risks
        concerns, risks = self._identify_concerns_and_risks(
            action, context, ecosystem_impacts, environmental_scores
        )
        
        # Generate recommendations and mitigation strategies
        recommendations, mitigations = self._generate_recommendations_and_mitigations(
            concerns, risks, environmental_scores
        )
        
        # Calculate overall impact score
        impact_score = self._calculate_overall_score(ecosystem_impacts, environmental_scores)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(context)
        
        return EcologicalImpactAssessment(
            impact_score=impact_score,
            affected_ecosystems=ecosystem_impacts,
            environmental_scores=environmental_scores,
            environmental_concerns=concerns,
            sustainability_recommendations=recommendations,
            risks=risks,
            mitigation_strategies=mitigations,
            confidence_level=confidence_level
        )
    
    def _assess_ecosystem_impacts(self, action: str, context: Dict) -> Dict[EcosystemType, float]:
        """Assess impact on each type of ecosystem"""
        impacts = {}
        
        for ecosystem_type in EcosystemType:
            score = self._calculate_ecosystem_impact(ecosystem_type, action, context)
            impacts[ecosystem_type] = max(min(score, 1.0), -1.0)
            
        return impacts
    
    def _assess_environmental_factors(self, action: str, context: Dict) -> Dict[EnvironmentalFactor, float]:
        """Assess impact on different environmental factors"""
        factor_scores = {}
        
        for factor in EnvironmentalFactor:
            score = self._calculate_factor_impact(factor, action, context)
            factor_scores[factor] = max(min(score, 1.0), -1.0)
            
        return factor_scores
    
    def _calculate_ecosystem_impact(self, ecosystem_type: EcosystemType, action: str, context: Dict) -> float:
        """Calculate impact score for a specific ecosystem type"""
        impact_score = 0.0
        
        # Digital ecosystem impacts
        if ecosystem_type == EcosystemType.DIGITAL:
            if 'optimize' in action.lower():
                impact_score += 0.5
            if 'data' in action.lower():
                impact_score -= 0.2  # Data operations have some environmental cost
                
        # Physical ecosystem impacts
        elif ecosystem_type == EcosystemType.PHYSICAL:
            if 'hardware' in action.lower():
                impact_score -= 0.4  # Hardware changes often have environmental impact
            if 'green' in action.lower():
                impact_score += 0.6
                
        # Energy ecosystem impacts
        elif ecosystem_type == EcosystemType.ENERGY:
            if 'efficiency' in action.lower():
                impact_score += 0.7
            if 'processing' in action.lower():
                impact_score -= 0.3
                
        return impact_score
    
    def _calculate_factor_impact(self, factor: EnvironmentalFactor, action: str, context: Dict) -> float:
        """Calculate impact score for a specific environmental factor"""
        impact_score = 0.0
        
        if factor == EnvironmentalFactor.ENERGY_USAGE:
            if 'efficiency' in action.lower():
                impact_score += 0.6
            if 'processing' in action.lower():
                impact_score -= 0.4
                
        elif factor == EnvironmentalFactor.RESOURCE_CONSUMPTION:
            if 'optimize' in action.lower():
                impact_score += 0.5
            if 'expand' in action.lower():
                impact_score -= 0.3
                
        elif factor == EnvironmentalFactor.SUSTAINABILITY:
            if 'sustainable' in action.lower() or 'green' in action.lower():
                impact_score += 0.8
                
        return impact_score
    
    def _identify_concerns_and_risks(self,
                                   action: str,
                                   context: Dict,
                                   ecosystem_impacts: Dict[EcosystemType, float],
                                   environmental_scores: Dict[EnvironmentalFactor, float]) -> Tuple[List[str], List[str]]:
        """Identify environmental concerns and risks"""
        concerns = []
        risks = []
        
        # Check ecosystem impacts
        for ecosystem_type, impact in ecosystem_impacts.items():
            if impact < -0.3:
                concerns.append(f"Significant negative impact on {ecosystem_type.value} ecosystem")
                risks.append(f"Risk of {ecosystem_type.value} ecosystem degradation")
            elif impact < 0:
                concerns.append(f"Minor negative impact on {ecosystem_type.value} ecosystem")
        
        # Check environmental factors
        for factor, score in environmental_scores.items():
            if score < -0.3:
                concerns.append(f"High {factor.value} impact detected")
                risks.append(f"Unsustainable {factor.value} levels")
            elif score < 0:
                concerns.append(f"Moderate {factor.value} impact detected")
        
        return concerns, risks
    
    def _generate_recommendations_and_mitigations(self,
                                                concerns: List[str],
                                                risks: List[str],
                                                environmental_scores: Dict[EnvironmentalFactor, float]) -> Tuple[List[str], List[str]]:
        """Generate recommendations and mitigation strategies"""
        recommendations = []
        mitigations = []
        
        # Generate recommendations based on environmental scores
        for factor, score in environmental_scores.items():
            if score < 0:
                recommendations.append(f"Improve {factor.value} efficiency")
                mitigations.append(f"Implement {factor.value} monitoring and optimization")
        
        # Add specific recommendations for concerns
        for concern in concerns:
            if "ecosystem" in concern:
                recommendations.append(f"Develop protection plan for affected ecosystem")
            if "impact" in concern:
                recommendations.append(f"Establish impact reduction measures")
        
        # Add mitigation strategies for risks
        for risk in risks:
            if "degradation" in risk:
                mitigations.append("Implement ecosystem health monitoring")
            if "unsustainable" in risk:
                mitigations.append("Develop sustainability metrics and thresholds")
        
        return recommendations, mitigations
    
    def _calculate_overall_score(self,
                               ecosystem_impacts: Dict[EcosystemType, float],
                               environmental_scores: Dict[EnvironmentalFactor, float]) -> float:
        """Calculate overall impact score"""
        # Calculate weighted ecosystem score
        ecosystem_score = sum(
            impact * self.ecosystem_weights[ecosystem_type]
            for ecosystem_type, impact in ecosystem_impacts.items()
        )
        
        # Calculate weighted environmental factor score
        factor_score = sum(
            score * self.factor_weights[factor]
            for factor, score in environmental_scores.items()
        )
        
        # Combine scores (equal weight to both aspects)
        overall_score = (ecosystem_score + factor_score) / 2
        return max(min(overall_score, 1.0), -1.0)
    
    def _calculate_confidence_level(self, context: Dict) -> float:
        """Calculate confidence level in the assessment"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available context
        if context.get('system_metrics'):
            confidence += 0.2
        if context.get('environmental_data'):
            confidence += 0.3
            
        return min(confidence, 1.0)