import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import time

from scripts.context_engine import ContextEngine
from scripts.ethical_governor import EthicalGovernor
from scripts.context_models import StakeholderData, RealTimeMetrics, ContextEntry
from scripts.report_templates import TestReport, ReportManager, LocationTestReport

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_location_decisions() -> LocationTestReport:
    """Test ethical decision-making across different regions"""
    
    # Initialize report
    report = LocationTestReport.create(
        test_name="regional_ethical_analysis",
        components=["ContextEngine", "EthicalGovernor", "LocationContext"]
    )
    
    start_time = time.time()
    location_results = {}
    regional_comparisons = {
        'privacy_emphasis': {},
        'innovation_tolerance': {},
        'environmental_priority': {},
        'cultural_alignment': {}
    }
    
    # Define ethically challenging scenarios
    test_scenarios = [
        {
            "region": "US-CA",
            "action": "implement_automated_decision_system",
            "description": "Deploy AI system for automated customer service decisions",
            "ethical_considerations": [
                "Privacy of customer data",
                "Algorithmic bias",
                "Job displacement"
            ],
            "stakeholder": StakeholderData(
                id=1,
                name="California Service Center",
                role="user",
                region="US-CA",
                priority_level=2,
                impact_score=75.0
            ),
            "expected_emphasis": ["privacy", "innovation"]
        },
        {
            "region": "DE-BY",
            "action": "collect_user_behavioral_data",
            "description": "Implement comprehensive user tracking for service improvement",
            "ethical_considerations": [
                "GDPR compliance",
                "Data minimization",
                "User consent"
            ],
            "stakeholder": StakeholderData(
                id=2,
                name="Bavarian Data Center",
                role="user",
                region="DE-BY",
                priority_level=2,
                impact_score=80.0
            ),
            "expected_emphasis": ["privacy", "data_protection"]
        },
        {
            "region": "JP-13",
            "action": "deploy_ai_assistant",
            "description": "Deploy AI assistant for personalized customer service",
            "ethical_considerations": [
                "Privacy of customer data",
                "Algorithmic bias",
                "Job displacement"
            ],
            "stakeholder": StakeholderData(
                id=3,
                name="Tokyo AI Center",
                role="user",
                region="JP-13",
                priority_level=2,
                impact_score=70.0
            ),
            "expected_emphasis": ["privacy", "innovation"]
        },
        {
            "region": "SG-01",
            "action": "implement_facial_recognition",
            "description": "Implement facial recognition system for security and convenience",
            "ethical_considerations": [
                "Privacy of personal data",
                "Algorithmic bias",
                "Job displacement"
            ],
            "stakeholder": StakeholderData(
                id=4,
                name="Singapore AI Center",
                role="user",
                region="SG-01",
                priority_level=2,
                impact_score=85.0
            ),
            "expected_emphasis": ["privacy", "efficiency"]
        },
        {
            "region": "IN-KA",
            "action": "outsource_data_processing",
            "description": "Outsource data processing to a third-party service provider",
            "ethical_considerations": [
                "Privacy of data",
                "Data security",
                "Job displacement"
            ],
            "stakeholder": StakeholderData(
                id=5,
                name="Bangalore Data Center",
                role="user",
                region="IN-KA",
                priority_level=2,
                impact_score=65.0
            ),
            "expected_emphasis": ["privacy", "scalability"]
        }
    ]
    
    try:
        # Initialize components
        context_engine = ContextEngine()
        ethical_governor = EthicalGovernor()
        location_manager = context_engine.location_manager
        
        results = []
        for scenario in test_scenarios:
            logger.info(f"\nAnalyzing ethical decision for region: {scenario['region']}")
            logger.info(f"Scenario: {scenario['description']}")
            logger.info(f"Ethical considerations: {scenario['ethical_considerations']}")
            
            # Set location context
            context_engine.set_location_context(scenario['region'])
            
            # Create test context with ethical considerations
            test_context = {
                "stakeholder": scenario['stakeholder'],
                "urgency_level": "medium",
                "location": scenario['region'],
                "ethical_considerations": scenario['ethical_considerations'],
                "regional_context": location_manager.get_context(scenario['region'])
            }
            
            # Evaluate through ethical governor
            decision = ethical_governor.evaluate_action(scenario['action'], test_context)
            
            # Analyze regional influence
            regional_analysis = analyze_regional_influence(
                scenario, 
                decision, 
                location_manager.get_context(scenario['region'])
            )
            
            # Record detailed results
            results.append({
                "region": scenario['region'],
                "scenario": scenario['description'],
                "ethical_considerations": scenario['ethical_considerations'],
                "decision": decision,
                "regional_analysis": regional_analysis,
                "expected_emphasis": scenario['expected_emphasis']
            })
            
            # Log detailed analysis
            logger.info(f"\nEthical Decision Analysis for {scenario['region']}:")
            logger.info(f"Decision: {decision.recommendation}")
            logger.info(f"Confidence Score: {decision.confidence_score}")
            logger.info(f"Prism Scores: {decision.prism_scores}")
            logger.info(f"Regional Influence: {regional_analysis}")
            
        # Update report with comprehensive analysis
        report.location_specific_data = format_location_results(results)
        report.regional_comparisons = analyze_regional_patterns(results)
        report.cultural_impact_analysis = analyze_cultural_impact(results, location_manager)
        report.status = "completed"
        
        # Add performance metrics
        end_time = time.time()
        report.performance_metrics = {
            'total_execution_time': end_time - start_time,
            'regions_tested': len(results),
            'average_confidence': sum(r['decision'].confidence_score for r in results) / len(results)
        }
        
    except Exception as e:
        report.status = "failed"
        report.errors.append(str(e))
        logger.error(f"Error during ethical analysis: {str(e)}", exc_info=True)
    
    return report

def analyze_regional_influence(scenario: Dict, decision: Any, regional_context: Any) -> Dict:
    """Analyze how regional context influenced the ethical decision"""
    return {
        "cultural_factors": {
            "privacy_emphasis": regional_context.cultural_context.privacy_emphasis.value,
            "innovation_tolerance": regional_context.cultural_context.innovation_tolerance.value,
            "decision_making_style": regional_context.cultural_context.decision_making_style
        },
        "legal_impact": {
            "privacy_laws": regional_context.legal_context.privacy_laws,
            "ai_regulations": regional_context.legal_context.ai_regulations
        },
        "societal_influence": {
            "privacy_importance": regional_context.societal_norms.privacy_importance,
            "innovation_focus": regional_context.societal_norms.innovation_focus
        },
        "decision_alignment": {
            "matches_cultural_values": check_cultural_alignment(decision, regional_context),
            "complies_with_regulations": check_regulatory_compliance(decision, regional_context)
        }
    }

def format_location_results(results: List[Dict]) -> Dict:
    """Format location-specific results for reporting"""
    formatted_results = {}
    for result in results:
        formatted_results[result['region']] = {
            'scenario': result['scenario'],
            'ethical_considerations': result['ethical_considerations'],
            'decision': str(result['decision'].recommendation),
            'confidence_score': result['decision'].confidence_score,
            'prism_scores': result['decision'].prism_scores,
            'regional_influence': result['regional_analysis']
        }
    return formatted_results

def analyze_cultural_impact(results: List[Dict], location_manager) -> Dict[str, Any]:
    """Analyze cultural impact patterns across regions"""
    analysis = {
        'cultural_patterns': {},
        'decision_variations': {},
        'compliance_factors': {}
    }
    
    # Analyze patterns in decisions across cultural contexts
    for result in results:
        region = result['region']
        context = location_manager.get_context(region)
        if context:
            # Record decision patterns for similar cultural values
            for value in context.cultural_context.primary_values:
                # Convert Enum to string value for serialization
                value_str = value.value
                if value_str not in analysis['cultural_patterns']:
                    analysis['cultural_patterns'][value_str] = []
                analysis['cultural_patterns'][value_str].append({
                    'region': region,
                    'decision': str(result['decision'].recommendation)
                })
    
    # Add decision variations analysis
    analysis['decision_variations'] = {
        'by_region': {
            result['region']: str(result['decision'].recommendation)
            for result in results
        }
    }
    
    # Add compliance analysis
    analysis['compliance_factors'] = {
        region: {
            'privacy_level': context.legal_context.data_protection_level,
            'ai_regulations': context.legal_context.ai_regulations
        }
        for region, context in location_manager.contexts.items()
    }
    
    return analysis

def analyze_regional_patterns(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Analyze patterns in decision-making across regions"""
    patterns = {
        'privacy_emphasis': {},
        'innovation_tolerance': {},
        'environmental_priority': {},
        'cultural_alignment': {},
        'decision_confidence': {}
    }
    
    for result in results:
        region = result['region']
        decision = result['decision']
        
        # Record decision confidence
        patterns['decision_confidence'][region] = decision.confidence_score
        
        # Record prism scores
        for prism_name, score in decision.prism_scores.items():
            if prism_name not in patterns:
                patterns[prism_name] = {}
            patterns[prism_name][region] = score
    
    return patterns

def check_cultural_alignment(decision: Any, regional_context: Any) -> bool:
    """Check if decision aligns with cultural values"""
    # Get decision characteristics
    is_privacy_focused = decision.prism_scores.get('human', 0) > 0.7
    is_innovation_focused = decision.prism_scores.get('innovation', 0) > 0.7
    is_community_focused = decision.prism_scores.get('sentient', 0) > 0.7
    
    # Check alignment with cultural context
    privacy_aligned = (
        is_privacy_focused == 
        (regional_context.cultural_context.privacy_emphasis in ['high', 'very_high'])
    )
    
    innovation_aligned = (
        is_innovation_focused == 
        (regional_context.cultural_context.innovation_tolerance == 'progressive')
    )
    
    community_aligned = (
        is_community_focused == 
        any(val in ['collectivist', 'hierarchical'] 
            for val in regional_context.cultural_context.primary_values)
    )
    
    # Return True if majority of aspects align
    return sum([privacy_aligned, innovation_aligned, community_aligned]) >= 2

def check_regulatory_compliance(decision: Any, regional_context: Any) -> bool:
    """Check if decision complies with regional regulations"""
    # Get decision characteristics
    involves_data = any(
        'data' in r['description'].lower() 
        if isinstance(r, dict) and 'description' in r 
        else 'data' in str(r).lower() 
        for r in decision.risk_factors
    )
    
    involves_ai = any(
        'ai' in r['description'].lower()
        if isinstance(r, dict) and 'description' in r
        else 'ai' in str(r).lower()
        for r in decision.risk_factors
    )
    
    involves_privacy = any(
        'privacy' in r['description'].lower()
        if isinstance(r, dict) and 'description' in r
        else 'privacy' in str(r).lower()
        for r in decision.risk_factors
    )
    
    # Check compliance requirements
    if involves_data and regional_context.legal_context.data_protection_level == 'very_high':
        if decision.confidence_score < 0.8:
            return False
            
    if involves_ai and 'AI Act' in regional_context.legal_context.ai_regulations:
        if decision.confidence_score < 0.75:
            return False
            
    if involves_privacy and regional_context.legal_context.special_requirements.get('data_residency') == 'mandatory':
        if decision.confidence_score < 0.85:
            return False
    
    return True

if __name__ == "__main__":
    # Initialize report manager
    report_manager = ReportManager()
    
    # Run tests and get report
    test_report = test_location_decisions()
    test_report.test_id = report_manager.get_next_id()
    
    # Save reports
    json_path, text_path = report_manager.save_location_test_report(test_report)
    print(f"\nLocation test reports saved to:")
    print(f"JSON: {json_path}")
    print(f"Text: {text_path}")
    
    # Display report content
    with open(text_path, 'r') as f:
        print(f.read()) 