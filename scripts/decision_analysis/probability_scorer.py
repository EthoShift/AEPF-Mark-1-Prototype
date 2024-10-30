from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

class ProbabilityBand(Enum):
    """Probability bands for decision outcomes"""
    HIGH = "high"        # 0.7 - 1.0
    MEDIUM = "medium"    # 0.4 - 0.69
    LOW = "low"         # 0.0 - 0.39

@dataclass
class ProbabilityScore:
    """Detailed probability score for a decision outcome"""
    raw_score: float
    adjusted_score: float
    band: ProbabilityBand
    influencing_factors: Dict[str, float]
    cultural_adjustments: Dict[str, float]
    compliance_impacts: Dict[str, float]
    confidence_level: float

class ProbabilityScorer:
    """Calculates probability scores for decision outcomes"""
    
    def __init__(self):
        # Adjust thresholds for better differentiation
        self.band_thresholds = {
            ProbabilityBand.HIGH: 0.65,    # Decreased from 0.75
            ProbabilityBand.MEDIUM: 0.45,  # Decreased from 0.55
            ProbabilityBand.LOW: 0.0
        }
        
        # Adjust weights for better balance
        self.influence_weights = {
            'prism_scores': 0.4,          # Increased from 0.35
            'cultural_context': 0.3,      # Decreased from 0.35
            'compliance': 0.3             # Unchanged
        }
    
    def calculate_probability(self,
                            prism_scores: Dict[str, float],
                            cultural_context: Dict,
                            compliance_data: Dict,
                            decision_impact: str) -> ProbabilityScore:
        """Calculate probability score with scenario-specific handling"""
        # Get scenario type and weights
        scenario_type = self._determine_scenario_type(cultural_context, compliance_data)
        weights = self._get_scenario_weights(scenario_type)
        
        # Calculate base score with scenario-specific adjustments
        base_score = self._calculate_base_score(prism_scores, scenario_type)
        
        # Apply scenario-specific adjustments with stronger effects
        if scenario_type == 'cultural':
            cultural_alignment = cultural_context.get('cultural_alignment', 0.5)
            social_impact = cultural_context.get('social_impact_score', 0.5)
            base_score = base_score * (1 + (cultural_alignment + social_impact) / 3)  # Increased from /4
            
        elif scenario_type == 'environmental':
            if cultural_context.get('environmental_priority') == 'high':
                base_score *= 1.4  # Increased from 1.3
            if cultural_context.get('sustainability_focus', False):
                base_score *= 1.3  # Increased from 1.2
                
        elif scenario_type == 'privacy':
            if cultural_context.get('privacy_level') == 'high':
                base_score *= 0.7  # Less reduction from 0.8
            if any('gdpr' in str(req).lower() for req in compliance_data.get('compliance', [])):
                base_score *= 0.8  # Less reduction from 0.7
        
        # Calculate cultural and compliance impacts
        cultural_adjustments = self._calculate_cultural_adjustments(
            cultural_context,
            prism_scores,
            scenario_type
        )
        compliance_impacts = self._assess_compliance_impact(
            compliance_data,
            decision_impact,
            scenario_type
        )
        
        # Combine scores with scenario-specific weights
        weighted_score = (
            base_score * weights['prism_scores'] +
            sum(cultural_adjustments.values()) * weights['cultural_context'] +
            sum(compliance_impacts.values()) * weights['compliance']
        )
        
        # Ensure score stays within bounds
        weighted_score = max(min(weighted_score, 1.0), 0.0)
        
        # Determine band with scenario-specific thresholds
        band = self._determine_probability_band(weighted_score, scenario_type)
        
        # Calculate confidence with scenario context
        confidence = self._calculate_confidence_level(
            prism_scores,
            cultural_context,
            compliance_data,
            scenario_type
        )
        
        return ProbabilityScore(
            raw_score=base_score,
            adjusted_score=weighted_score,
            band=band,
            influencing_factors={'base_score': base_score},
            cultural_adjustments=cultural_adjustments,
            compliance_impacts=compliance_impacts,
            confidence_level=confidence
        )
    
    def _calculate_base_score(self, prism_scores: Dict[str, float], scenario_type: str) -> float:
        """Calculate base probability score from prism evaluations"""
        # Normalize scores to 0-1 range and apply higher weight to positive scores
        normalized_scores = {}
        for k, v in prism_scores.items():
            normalized = (v + 1) / 2  # Convert from -1:1 to 0:1
            if v > 0:
                normalized *= 2.0  # Increase boost for positive scores
            normalized_scores[k] = min(normalized, 1.0)
        
        # Calculate weighted average with higher base value
        base_score = 0.5  # Start with a higher base score
        if normalized_scores:
            base_score += sum(normalized_scores.values()) / len(normalized_scores)
        
        return min(base_score, 1.0)
    
    def _calculate_cultural_adjustments(self,
                                     cultural_context: Dict,
                                     prism_scores: Dict[str, float],
                                     scenario_type: str) -> Dict[str, float]:
        """Calculate cultural influence adjustments"""
        adjustments = {}
        
        # Increase adjustment values
        if 'individualistic' in str(cultural_context.get('primary_values', [])).lower():
            adjustments['individualism'] = 0.4  # Increased from 0.3
            if 'innovation' in prism_scores:
                adjustments['innovation_boost'] = 0.45  # Increased from 0.35
        elif 'collectivist' in str(cultural_context.get('primary_values', [])).lower():
            adjustments['collectivism'] = 0.4  # Increased from 0.3
            if 'human' in prism_scores and 'eco' in prism_scores:
                adjustments['social_eco_boost'] = 0.45  # Increased from 0.35
        
        # Adjust for privacy emphasis with higher impact
        privacy_emphasis = str(cultural_context.get('privacy_emphasis', '')).lower()
        if privacy_emphasis in ['high', 'very_high']:
            adjustments['privacy_emphasis'] = 0.5  # Increased from 0.4
        
        return adjustments
    
    def _assess_compliance_impact(self,
                                compliance_data: Dict,
                                decision_impact: str,
                                scenario_type: str) -> Dict[str, float]:
        """Assess impact of compliance factors"""
        impacts = {}
        
        # Reduce negative impacts to allow for higher overall scores
        if compliance_data.get('data_protection_level') == 'very_high':
            impacts['data_protection'] = -0.15  # Reduced from -0.2
        
        if 'AI Act' in str(compliance_data.get('ai_regulations', [])):
            impacts['ai_regulation'] = -0.1  # Reduced from -0.15
        
        if decision_impact == 'critical':
            impacts['critical_impact'] = -0.2  # Reduced from -0.25
        elif decision_impact == 'high':
            impacts['high_impact'] = -0.1  # Reduced from -0.15
        
        return impacts
    
    def _determine_probability_band(self, score: float, scenario_type: str) -> ProbabilityBand:
        """Determine probability band from score with scenario-specific thresholds"""
        thresholds = {
            'cultural': {
                'high': 0.65,   # Lowered from 0.70
                'medium': 0.40  # Lowered from 0.45
            },
            'environmental': {
                'high': 0.55,   # Lowered further for environmental
                'medium': 0.35  # Lowered for better alignment
            },
            'privacy': {
                'high': 0.85,   # Increased for stricter privacy
                'medium': 0.65  # Increased for privacy
            },
            'default': {
                'high': 0.65,
                'medium': 0.45
            }
        }
        
        scenario_threshold = thresholds.get(scenario_type, thresholds['default'])
        
        # Apply scenario-specific boost for environmental approvals
        if scenario_type == 'environmental' and score > 0.45:  # Added boost
            score *= 1.2
        
        if score >= scenario_threshold['high']:
            return ProbabilityBand.HIGH
        elif score >= scenario_threshold['medium']:
            return ProbabilityBand.MEDIUM
        else:
            return ProbabilityBand.LOW
    
    def _calculate_confidence_level(self,
                                  prism_scores: Dict[str, float],
                                  cultural_context: Dict,
                                  compliance_data: Dict,
                                  scenario_type: str) -> float:
        """Calculate confidence level in probability assessment"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available data
        if prism_scores:
            confidence += 0.2
        if cultural_context:
            confidence += 0.15
        if compliance_data:
            confidence += 0.15
            
        return min(confidence, 1.0) 

    def _calculate_scenario_boost(self,
                                cultural_context: Dict,
                                compliance_data: Dict,
                                decision_impact: str) -> float:
        """Calculate scenario-specific probability boost"""
        boost = 0.0
        
        # Check for scenario-specific indicators
        scenario_type = self._determine_scenario_type(cultural_context, compliance_data)
        
        if scenario_type == 'privacy':
            boost -= 0.3  # Reduced from -0.4
            if 'GDPR' in str(compliance_data):
                boost -= 0.1
        
        elif scenario_type == 'innovation':
            boost += 0.4  # Reduced from 0.6
            if cultural_context.get('innovation_tolerance') == 'progressive':
                boost += 0.2
        
        elif scenario_type == 'environmental':
            boost += 0.3  # Base boost for environmental
            if cultural_context.get('environmental_priority') == 'high':
                boost += 0.2
        
        # Risk level adjustments
        risk_level = str(cultural_context.get('risk_level', '')).lower()
        if risk_level == 'low':
            boost += 0.1  # Reduced from 0.2
        elif risk_level == 'high':
            boost -= 0.1  # Reduced from -0.2
        
        return boost

    def _determine_scenario_type(self, cultural_context: Dict, compliance_data: Dict) -> str:
        """Determine the primary type of scenario"""
        context_str = str(cultural_context) + str(compliance_data)
        
        if any(term in context_str.lower() for term in ['privacy', 'gdpr', 'data protection']):
            return 'privacy'
        elif any(term in context_str.lower() for term in ['innovation', 'ai', 'optimize']):
            return 'innovation'
        elif any(term in context_str.lower() for term in ['environment', 'green', 'sustainable']):
            return 'environmental'
        
        return 'general'

    def _get_scenario_weights(self, scenario_type: str) -> Dict[str, float]:
        """Get scenario-specific weights"""
        weights = {
            'compliance': {
                'prism_scores': 0.25,    # Reduced from 0.3
                'cultural_context': 0.25,
                'compliance': 0.50       # Increased from 0.4 for stricter compliance
            },
            'cultural': {
                'prism_scores': 0.30,
                'cultural_context': 0.45, # Increased for better cultural sensitivity
                'compliance': 0.25
            },
            'environmental': {
                'prism_scores': 0.45,    # Increased for resource efficiency
                'cultural_context': 0.25,
                'compliance': 0.30
            },
            'privacy': {
                'prism_scores': 0.20,    # Reduced for privacy focus
                'cultural_context': 0.30,
                'compliance': 0.50       # Increased for privacy protection
            },
            'medical': {
                'prism_scores': 0.15,    # Lowest for medical scenarios
                'cultural_context': 0.25,
                'compliance': 0.60       # Highest for medical compliance
            },
            'default': {
                'prism_scores': 0.35,
                'cultural_context': 0.35,
                'compliance': 0.30
            }
        }
        return weights.get(scenario_type, weights['default'])