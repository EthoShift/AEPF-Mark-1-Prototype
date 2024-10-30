from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)

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
    initial_recommendation: Optional[ProbabilityBand] = None

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
                            context: Dict,
                            compliance_data: Dict,
                            decision_impact: str) -> ProbabilityScore:
        """Calculate probability score with enhanced confidence handling"""
        logger.debug("Entering calculate_probability")
        logger.debug(f"Input - prism_scores: {prism_scores}")
        
        try:
            # Calculate base scores without recursion
            base_score = self._calculate_base_score(prism_scores)
            logger.debug(f"Base score calculated: {base_score}")
            
            # Calculate confidence directly
            confidence = self._calculate_direct_confidence(context)
            logger.debug(f"Confidence calculated: {confidence}")
            
            # Determine band without recursion
            band = self._determine_band_from_score(base_score)
            logger.debug(f"Initial band determined: {band}")
            
            # Create probability score
            score = ProbabilityScore(
                raw_score=base_score,
                adjusted_score=base_score,
                band=band,
                influencing_factors=prism_scores,
                cultural_adjustments={},
                compliance_impacts=compliance_data,
                confidence_level=confidence
            )
            
            logger.debug("Exiting calculate_probability successfully")
            return score
            
        except Exception as e:
            logger.error(f"Error in calculate_probability: {str(e)}")
            raise
    
    def _calculate_base_score(self, prism_scores: Dict[str, float]) -> float:
        """Calculate base score without recursion"""
        logger.debug("Calculating base score")
        
        if not prism_scores:
            return 0.0
            
        # Simple weighted average
        total_weight = sum(self.influence_weights.values())
        weighted_sum = sum(
            score * self.influence_weights.get(prism, 0.25)
            for prism, score in prism_scores.items()
        )
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_direct_confidence(self, context: Dict) -> float:
        """Calculate confidence directly without recursion"""
        logger.debug("Calculating direct confidence")
        
        base_confidence = 0.5  # Start with moderate confidence
        
        # Add confidence based on context completeness
        if context.get('stakeholder'):
            base_confidence += 0.1
        if context.get('metrics'):
            base_confidence += 0.1
        if context.get('historical_data'):
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
    
    def _determine_band_from_score(self, score: float) -> ProbabilityBand:
        """Determine band directly from score"""
        logger.debug(f"Determining band for score: {score}")
        
        if score >= 0.65:
            return ProbabilityBand.HIGH
        elif score >= 0.45:
            return ProbabilityBand.MEDIUM
        else:
            return ProbabilityBand.LOW