from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import json
from pathlib import Path

class PrivacyEmphasis(Enum):
    """Privacy emphasis levels in different regions"""
    VERY_HIGH = "very_high"  # e.g., EU GDPR regions
    HIGH = "high"            # e.g., California
    MODERATE = "moderate"    # e.g., Canada
    BASIC = "basic"         # e.g., regions with minimal privacy laws
    
class InnovationTolerance(Enum):
    """Innovation tolerance levels"""
    PROGRESSIVE = "progressive"  # Actively promotes innovation
    SUPPORTIVE = "supportive"   # Generally supports innovation
    NEUTRAL = "neutral"         # Neither promotes nor restricts
    CONSERVATIVE = "conservative"  # Prefers established methods

class CulturalValues(Enum):
    """Cultural value orientations"""
    INDIVIDUALISTIC = "individualistic"
    COLLECTIVIST = "collectivist"
    HIERARCHICAL = "hierarchical"
    EGALITARIAN = "egalitarian"
    PROGRESSIVE = "progressive"  # Added this value
    TRADITIONAL = "traditional"  # Added for completeness
    INNOVATIVE = "innovative"    # Added for completeness
    CONSERVATIVE = "conservative"  # Added for completeness

@dataclass
class SocietalNorms:
    """Scores for different societal priorities"""
    privacy_importance: float  # 0-1 scale
    innovation_focus: float
    environmental_priority: float
    community_focus: float
    individual_rights: float
    authority_respect: float

@dataclass
class LegalContext:
    """Legal framework information"""
    privacy_laws: List[str]
    data_protection_level: str
    environmental_regulations: List[str]
    ai_regulations: List[str]
    special_requirements: Dict[str, str]

@dataclass
class CulturalContext:
    """Cultural context information"""
    primary_values: List[CulturalValues]
    innovation_tolerance: InnovationTolerance
    privacy_emphasis: PrivacyEmphasis
    decision_making_style: str
    risk_tolerance: float  # 0-1 scale

@dataclass
class RegionalContext:
    """Complete regional context information"""
    region_id: str  # e.g., "US-CA" for California, USA
    country: str
    state_province: Optional[str]
    city: Optional[str]
    cultural_context: CulturalContext
    legal_context: LegalContext
    societal_norms: SocietalNorms
    context_confidence: float  # Confidence in context data (0-1)
    last_updated: str  # ISO format date

class LocationContextManager:
    """Manages location-based context data"""
    
    def __init__(self, context_file: str = "config/location_contexts.json"):
        self.context_file = Path(context_file)
        self.contexts: Dict[str, RegionalContext] = {}
        self.load_contexts()
        
    def load_contexts(self) -> None:
        """Load context data from JSON file"""
        if self.context_file.exists():
            with open(self.context_file, 'r') as f:
                data = json.load(f)
                for region_data in data['regions']:
                    context = self._parse_region_context(region_data)
                    self.contexts[context.region_id] = context
    
    def _parse_region_context(self, data: Dict) -> RegionalContext:
        """Parse JSON data into RegionalContext object"""
        return RegionalContext(
            region_id=data['region_id'],
            country=data['country'],
            state_province=data.get('state_province'),
            city=data.get('city'),
            cultural_context=CulturalContext(
                primary_values=[CulturalValues(v) for v in data['cultural_context']['primary_values']],
                innovation_tolerance=InnovationTolerance(data['cultural_context']['innovation_tolerance']),
                privacy_emphasis=PrivacyEmphasis(data['cultural_context']['privacy_emphasis']),
                decision_making_style=data['cultural_context']['decision_making_style'],
                risk_tolerance=data['cultural_context']['risk_tolerance']
            ),
            legal_context=LegalContext(
                privacy_laws=data['legal_context']['privacy_laws'],
                data_protection_level=data['legal_context']['data_protection_level'],
                environmental_regulations=data['legal_context']['environmental_regulations'],
                ai_regulations=data['legal_context']['ai_regulations'],
                special_requirements=data['legal_context']['special_requirements']
            ),
            societal_norms=SocietalNorms(
                privacy_importance=data['societal_norms']['privacy_importance'],
                innovation_focus=data['societal_norms']['innovation_focus'],
                environmental_priority=data['societal_norms']['environmental_priority'],
                community_focus=data['societal_norms']['community_focus'],
                individual_rights=data['societal_norms']['individual_rights'],
                authority_respect=data['societal_norms']['authority_respect']
            ),
            context_confidence=data['context_confidence'],
            last_updated=data['last_updated']
        )
    
    def get_context(self, region_id: str) -> Optional[RegionalContext]:
        """Retrieve context for a specific region"""
        return self.contexts.get(region_id)
    
    def get_nearest_context(self, region_id: str) -> Optional[RegionalContext]:
        """Get nearest available context if exact match not found"""
        if region_id in self.contexts:
            return self.contexts[region_id]
        
        # Try to find parent region (e.g., country level for state)
        country_code = region_id.split('-')[0]
        for context in self.contexts.values():
            if context.country == country_code:
                return context
        
        return None
    
    def adjust_weights(self, weights: Dict[str, float], context: RegionalContext) -> Dict[str, float]:
        """Adjust decision weights based on regional context"""
        adjusted = weights.copy()
        
        # Adjust based on cultural values
        if CulturalValues.INDIVIDUALISTIC in context.cultural_context.primary_values:
            adjusted['human'] *= 1.2
            adjusted['innovation'] *= 1.1
        elif CulturalValues.COLLECTIVIST in context.cultural_context.primary_values:
            adjusted['sentient'] *= 1.2
            adjusted['eco'] *= 1.1
        
        # Adjust based on privacy emphasis
        if context.cultural_context.privacy_emphasis in [PrivacyEmphasis.VERY_HIGH, PrivacyEmphasis.HIGH]:
            adjusted['human'] *= 1.3
        
        # Normalize weights
        total = sum(adjusted.values())
        return {k: v/total for k, v in adjusted.items()} 