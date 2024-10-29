from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

class TimeHorizon(Enum):
    """Time horizons for impact assessment"""
    IMMEDIATE = "immediate"      # 0-1 year
    SHORT_TERM = "short_term"    # 1-3 years
    MEDIUM_TERM = "medium_term"  # 3-7 years
    LONG_TERM = "long_term"      # 7-15 years
    FUTURE_GEN = "future_gen"    # 15+ years

class SustainabilityDomain(Enum):
    """Domains of sustainability impact"""
    RESOURCE_USE = "resource_use"
    ADAPTABILITY = "adaptability"
    SCALABILITY = "scalability"
    MAINTENANCE = "maintenance"
    EVOLUTION = "evolution"
    KNOWLEDGE = "knowledge"

@dataclass
class SustainabilityAssessment:
    """Assessment of long-term sustainability impact"""
    impact_score: float  # Overall score between -1 and 1
    temporal_scores: Dict[TimeHorizon, float]  # Impact scores across time horizons
    domain_scores: Dict[SustainabilityDomain, float]  # Scores per sustainability domain
    sustainability_concerns: List[str]
    future_risks: List[Dict[str, Any]]  # Potential future risks
    opportunities: List[str]
    adaptation_strategies: List[str]
    confidence_level: float  # Confidence in assessment (0-1)

class SustainabilityPrism:
    """
    Sustainability-Focused Ethical Prism for AEPF Mk1
    Evaluates decisions based on long-term sustainability and future impact
    """
    
    def __init__(self):
        # Define weights for different time horizons
        self.time_weights = {
            TimeHorizon.IMMEDIATE: 0.15,
            TimeHorizon.SHORT_TERM: 0.20,
            TimeHorizon.MEDIUM_TERM: 0.25,
            TimeHorizon.LONG_TERM: 0.25,
            TimeHorizon.FUTURE_GEN: 0.15
        }
        
        # Define weights for sustainability domains
        self.domain_weights = {
            SustainabilityDomain.RESOURCE_USE: 0.20,
            SustainabilityDomain.ADAPTABILITY: 0.20,
            SustainabilityDomain.SCALABILITY: 0.15,
            SustainabilityDomain.MAINTENANCE: 0.15,
            SustainabilityDomain.EVOLUTION: 0.15,
            SustainabilityDomain.KNOWLEDGE: 0.15
        }
    
    def evaluate(self, action: str, context: Dict) -> SustainabilityAssessment:
        """
        Evaluate an action from a sustainability perspective
        
        Args:
            action: Proposed action to evaluate
            context: Current context information
            
        Returns:
            SustainabilityAssessment containing the evaluation results
        """
        # Evaluate temporal impacts
        temporal_scores = self._assess_temporal_impacts(action, context)
        
        # Evaluate domain impacts
        domain_scores = self._assess_domain_impacts(action, context)
        
        # Identify concerns and risks
        concerns, risks = self._identify_concerns_and_risks(
            action, context, temporal_scores, domain_scores
        )
        
        # Identify opportunities
        opportunities = self._identify_opportunities(action, context)
        
        # Generate adaptation strategies
        adaptation_strategies = self._generate_adaptation_strategies(
            concerns, risks, opportunities
        )
        
        # Calculate overall impact score
        impact_score = self._calculate_overall_score(temporal_scores, domain_scores)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(context)
        
        return SustainabilityAssessment(
            impact_score=impact_score,
            temporal_scores=temporal_scores,
            domain_scores=domain_scores,
            sustainability_concerns=concerns,
            future_risks=risks,
            opportunities=opportunities,
            adaptation_strategies=adaptation_strategies,
            confidence_level=confidence_level
        )
    
    def _assess_temporal_impacts(self, action: str, context: Dict) -> Dict[TimeHorizon, float]:
        """Assess impact across different time horizons"""
        scores = {}
        
        for horizon in TimeHorizon:
            score = self._calculate_temporal_impact(horizon, action, context)
            scores[horizon] = max(min(score, 1.0), -1.0)
            
        return scores
    
    def _assess_domain_impacts(self, action: str, context: Dict) -> Dict[SustainabilityDomain, float]:
        """Assess impact across sustainability domains"""
        scores = {}
        
        for domain in SustainabilityDomain:
            score = self._calculate_domain_impact(domain, action, context)
            scores[domain] = max(min(score, 1.0), -1.0)
            
        return scores
    
    def _calculate_temporal_impact(self, horizon: TimeHorizon, action: str, context: Dict) -> float:
        """Calculate impact score for a specific time horizon"""
        score = 0.0
        
        # Immediate impact assessment
        if horizon == TimeHorizon.IMMEDIATE:
            if 'quick' in action.lower() or 'immediate' in action.lower():
                score += 0.5
            if 'temporary' in action.lower():
                score -= 0.3
                
        # Short-term impact assessment
        elif horizon == TimeHorizon.SHORT_TERM:
            if 'optimize' in action.lower():
                score += 0.4
            if 'workaround' in action.lower():
                score -= 0.2
                
        # Medium-term impact assessment
        elif horizon == TimeHorizon.MEDIUM_TERM:
            if 'sustainable' in action.lower():
                score += 0.6
            if 'legacy' in action.lower():
                score -= 0.4
                
        # Long-term impact assessment
        elif horizon == TimeHorizon.LONG_TERM:
            if 'future-proof' in action.lower():
                score += 0.8
            if 'obsolete' in action.lower():
                score -= 0.6
                
        # Future generations impact assessment
        elif horizon == TimeHorizon.FUTURE_GEN:
            if 'foundation' in action.lower():
                score += 0.7
            if 'lock-in' in action.lower():
                score -= 0.5
                
        return score
    
    def _calculate_domain_impact(self, domain: SustainabilityDomain, action: str, context: Dict) -> float:
        """Calculate impact score for a specific sustainability domain"""
        score = 0.0
        
        if domain == SustainabilityDomain.RESOURCE_USE:
            if 'efficient' in action.lower():
                score += 0.6
            if 'consume' in action.lower():
                score -= 0.4
                
        elif domain == SustainabilityDomain.ADAPTABILITY:
            if 'flexible' in action.lower():
                score += 0.7
            if 'rigid' in action.lower():
                score -= 0.5
                
        elif domain == SustainabilityDomain.SCALABILITY:
            if 'scale' in action.lower():
                score += 0.5
            if 'limit' in action.lower():
                score -= 0.3
                
        return score
    
    def _identify_concerns_and_risks(self,
                                   action: str,
                                   context: Dict,
                                   temporal_scores: Dict[TimeHorizon, float],
                                   domain_scores: Dict[SustainabilityDomain, float]) -> Tuple[List[str], List[Dict]]:
        """Identify sustainability concerns and future risks"""
        concerns = []
        risks = []
        
        # Check temporal impacts
        for horizon, score in temporal_scores.items():
            if score < -0.3:
                concerns.append(f"Significant negative impact in {horizon.value} timeframe")
                risks.append({
                    'timeframe': horizon.value,
                    'risk_type': 'temporal',
                    'severity': 'high',
                    'description': f"Potential sustainability issues in {horizon.value} timeframe",
                    'mitigation_required': True
                })
        
        # Check domain impacts
        for domain, score in domain_scores.items():
            if score < -0.3:
                concerns.append(f"Sustainability concerns in {domain.value} domain")
                risks.append({
                    'domain': domain.value,
                    'risk_type': 'domain',
                    'severity': 'high',
                    'description': f"Sustainability risk in {domain.value}",
                    'mitigation_required': True
                })
        
        return concerns, risks
    
    def _identify_opportunities(self, action: str, context: Dict) -> List[str]:
        """Identify sustainability opportunities"""
        opportunities = []
        
        if 'improve' in action.lower():
            opportunities.append("Potential for sustainability improvement")
        if 'innovate' in action.lower():
            opportunities.append("Opportunity for sustainable innovation")
        if 'optimize' in action.lower():
            opportunities.append("Resource optimization potential")
            
        return opportunities
    
    def _generate_adaptation_strategies(self,
                                     concerns: List[str],
                                     risks: List[Dict],
                                     opportunities: List[str]) -> List[str]:
        """Generate adaptation strategies"""
        strategies = []
        
        # Address concerns
        for concern in concerns:
            if "temporal" in concern:
                strategies.append("Develop temporal adaptation framework")
            if "domain" in concern:
                strategies.append("Implement domain-specific sustainability measures")
        
        # Address risks
        for risk in risks:
            if risk['mitigation_required']:
                strategies.append(f"Develop mitigation strategy for {risk['description']}")
        
        # Leverage opportunities
        for opportunity in opportunities:
            strategies.append(f"Develop plan to leverage: {opportunity}")
        
        return strategies
    
    def _calculate_overall_score(self,
                               temporal_scores: Dict[TimeHorizon, float],
                               domain_scores: Dict[SustainabilityDomain, float]) -> float:
        """Calculate overall sustainability impact score"""
        # Calculate weighted temporal score
        temporal_score = sum(
            score * self.time_weights[horizon]
            for horizon, score in temporal_scores.items()
        )
        
        # Calculate weighted domain score
        domain_score = sum(
            score * self.domain_weights[domain]
            for domain, score in domain_scores.items()
        )
        
        # Combine scores (equal weight to both aspects)
        overall_score = (temporal_score + domain_score) / 2
        return max(min(overall_score, 1.0), -1.0)
    
    def _calculate_confidence_level(self, context: Dict) -> float:
        """Calculate confidence level in the assessment"""
        confidence = 0.5  # Base confidence
        
        if context.get('historical_data'):
            confidence += 0.2
        if context.get('future_projections'):
            confidence += 0.2
        if context.get('sustainability_metrics'):
            confidence += 0.1
            
        return min(confidence, 1.0) 