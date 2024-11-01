from typing import Dict, Any
from datetime import datetime

# Test scenarios from prototype testing
TEST_SCENARIOS = {
    "Medical Diagnosis System": {
        "description": "AI-powered medical diagnosis assistant",
        "type": "medical",
        "metrics": {
            "safety": 0.85,
            "reliability": 0.78,
            "ethical": 0.82
        },
        "risk_level": "HIGH",
        "findings": {
            "safety": [
                "Data protection measures in place",
                "Security protocols validated",
                "Emergency override systems operational"
            ],
            "reliability": [
                "System stability confirmed",
                "Error rates within acceptable range",
                "Backup systems verified"
            ],
            "ethical": [
                "Privacy protection: Strong",
                "Bias mitigation: Implemented",
                "Transparency: High"
            ]
        }
    },
    "Environmental Monitoring": {
        "description": "Climate change prediction and monitoring system",
        "type": "environmental",
        "metrics": {
            "safety": 0.92,
            "reliability": 0.88,
            "ethical": 0.90
        },
        "risk_level": "MEDIUM",
        "findings": {
            "safety": [
                "Data validation protocols active",
                "Sensor network secured",
                "Redundant systems in place"
            ],
            "reliability": [
                "Prediction accuracy verified",
                "Real-time monitoring active",
                "Data quality checks passing"
            ],
            "ethical": [
                "Environmental impact: Minimal",
                "Resource usage: Optimized",
                "Community impact: Positive"
            ]
        }
    },
    "Financial Trading AI": {
        "description": "Automated trading and risk assessment system",
        "type": "financial",
        "metrics": {
            "safety": 0.75,
            "reliability": 0.82,
            "ethical": 0.71
        },
        "risk_level": "HIGH",
        "findings": {
            "safety": [
                "Transaction limits implemented",
                "Fraud detection active",
                "Risk controls in place"
            ],
            "reliability": [
                "Market analysis verified",
                "Algorithm stability tested",
                "Performance metrics tracked"
            ],
            "ethical": [
                "Fair trading practices",
                "Market manipulation prevention",
                "Transparency requirements met"
            ]
        }
    },
    "Content Moderation": {
        "description": "AI-powered content moderation system",
        "type": "social",
        "metrics": {
            "safety": 0.88,
            "reliability": 0.85,
            "ethical": 0.87
        },
        "risk_level": "MEDIUM",
        "findings": {
            "safety": [
                "Content filtering active",
                "User protection measures",
                "Report handling system"
            ],
            "reliability": [
                "Classification accuracy high",
                "Response time optimized",
                "False positive rate low"
            ],
            "ethical": [
                "Bias monitoring active",
                "Cultural sensitivity implemented",
                "Appeals process available"
            ]
        }
    }
}

class ContextEngine:
    def __init__(self):
        self.scenarios = TEST_SCENARIOS
    
    def get_scenarios(self) -> Dict[str, Dict]:
        """Return available test scenarios"""
        return self.scenarios
    
    def get_scenario_details(self, scenario_name: str) -> Dict[str, Any]:
        """Get details for a specific scenario"""
        return self.scenarios.get(scenario_name, {})
    
    def evaluate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate context using scenario data"""
        scenario_name = context.get('scenario')
        scenario_data = self.scenarios.get(scenario_name, {})
        
        return {
            'metrics': scenario_data.get('metrics', {}),
            'findings': scenario_data.get('findings', {}),
            'risk_level': scenario_data.get('risk_level', 'MEDIUM'),
            'type': scenario_data.get('type', 'unknown'),
            'description': scenario_data.get('description', '')
        }