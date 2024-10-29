from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from scripts.decision_analysis.probability_scorer import ProbabilityBand

@dataclass
class NarrativeComponents:
    """Components for building outcome narrative"""
    main_outcome: str
    secondary_effects: List[str]
    regional_factors: List[str]
    confidence_statement: str

class OutcomeNarrative:
    """Generates probable outcome narratives for decisions"""
    
    def __init__(self):
        self.impact_phrases = {
            'high_positive': [
                "is highly likely to yield positive results",
                "shows strong potential for success",
                "demonstrates significant promise"
            ],
            'moderate_positive': [
                "may produce favorable outcomes",
                "shows moderate potential for success",
                "indicates possible benefits"
            ],
            'low_positive': [
                "might have limited positive impact",
                "shows minimal potential for improvement",
                "indicates possible minor benefits"
            ],
            'negative': [
                "may face significant challenges",
                "shows potential risks",
                "indicates possible adverse effects"
            ]
        }
        
        self.confidence_phrases = {
            'high': "High confidence in this assessment based on comprehensive data",
            'moderate': "Moderate confidence in this assessment with some uncertainty",
            'low': "Limited confidence due to uncertainty in available data"
        }
    
    def generate_narrative(self,
                         probability_score: float,
                         probability_band: ProbabilityBand,
                         prism_scores: Dict[str, float],
                         regional_context: Dict,
                         confidence_level: float) -> str:
        """Generate a narrative description of probable outcomes"""
        
        # Generate narrative components
        components = self._build_narrative_components(
            probability_score,
            probability_band,
            prism_scores,
            regional_context,
            confidence_level
        )
        
        # Compose full narrative
        narrative = [
            f"This decision {self._get_impact_phrase(probability_score)}.",
            components.main_outcome
        ]
        
        # Add secondary effects if present
        if components.secondary_effects:
            narrative.append("\nSecondary Effects:")
            narrative.extend([f"- {effect}" for effect in components.secondary_effects])
        
        # Add regional factors if present
        if components.regional_factors:
            narrative.append("\nRegional Considerations:")
            narrative.extend([f"- {factor}" for factor in components.regional_factors])
        
        # Add confidence statement
        narrative.append(f"\n{components.confidence_statement}")
        
        return "\n".join(narrative)
    
    def _build_narrative_components(self,
                                  probability_score: float,
                                  probability_band: ProbabilityBand,
                                  prism_scores: Dict[str, float],
                                  regional_context: Dict,
                                  confidence_level: float) -> NarrativeComponents:
        """Build components of the narrative"""
        
        # Generate main outcome based on probability and top prism scores
        main_outcome = self._generate_main_outcome(
            probability_score,
            probability_band,
            prism_scores
        )
        
        # Identify secondary effects from other significant prism scores
        secondary_effects = self._identify_secondary_effects(prism_scores)
        
        # Extract relevant regional factors
        regional_factors = self._extract_regional_factors(regional_context)
        
        # Generate confidence statement
        confidence_statement = self._generate_confidence_statement(confidence_level)
        
        return NarrativeComponents(
            main_outcome=main_outcome,
            secondary_effects=secondary_effects,
            regional_factors=regional_factors,
            confidence_statement=confidence_statement
        )
    
    def _get_impact_phrase(self, probability_score: float) -> str:
        """Get appropriate impact phrase based on probability score"""
        if probability_score >= 0.7:
            return self.impact_phrases['high_positive'][0]
        elif probability_score >= 0.4:
            return self.impact_phrases['moderate_positive'][0]
        elif probability_score >= 0:
            return self.impact_phrases['low_positive'][0]
        else:
            return self.impact_phrases['negative'][0]
    
    def _generate_main_outcome(self,
                             probability_score: float,
                             probability_band: ProbabilityBand,
                             prism_scores: Dict[str, float]) -> str:
        """Generate main outcome description"""
        # Find most influential prism
        top_prism = max(prism_scores.items(), key=lambda x: abs(x[1]))
        
        outcome = f"Primary impact is expected in the {top_prism[0]} domain "
        outcome += f"with a {probability_band.value} probability of success. "
        
        if probability_score >= 0.7:
            outcome += "Strong positive outcomes are anticipated."
        elif probability_score >= 0.4:
            outcome += "Moderate positive outcomes are possible."
        else:
            outcome += "Outcomes may face challenges."
            
        return outcome
    
    def _identify_secondary_effects(self, prism_scores: Dict[str, float]) -> List[str]:
        """Identify secondary effects from other significant prism scores"""
        effects = []
        
        for prism, score in prism_scores.items():
            if abs(score) >= 0.3:  # Threshold for significant secondary effects
                if score > 0:
                    effects.append(f"Positive {prism} impact expected (score: {score:.2f})")
                else:
                    effects.append(f"Potential {prism} concerns noted (score: {score:.2f})")
                    
        return effects
    
    def _extract_regional_factors(self, regional_context: Dict) -> List[str]:
        """Extract relevant regional factors"""
        factors = []
        
        # Handle privacy emphasis
        if regional_context.get('privacy_emphasis') in ['high', 'very_high']:
            factors.append("Strong privacy considerations in this region")
            
        # Handle innovation tolerance
        if regional_context.get('innovation_tolerance') == 'progressive':
            factors.append("Region favors innovative approaches")
            
        # Handle environmental priority - convert to float if string
        env_priority = regional_context.get('environmental_priority', '0')
        try:
            env_priority = float(env_priority)
            if env_priority >= 0.8:
                factors.append("High environmental standards in effect")
        except (ValueError, TypeError):
            # If it's a string value, check for high/very high
            if str(env_priority).lower() in ['high', 'very_high']:
                factors.append("High environmental standards in effect")
            
        return factors
    
    def _generate_confidence_statement(self, confidence_level: float) -> str:
        """Generate appropriate confidence statement"""
        if confidence_level >= 0.8:
            return self.confidence_phrases['high']
        elif confidence_level >= 0.5:
            return self.confidence_phrases['moderate']
        else:
            return self.confidence_phrases['low'] 