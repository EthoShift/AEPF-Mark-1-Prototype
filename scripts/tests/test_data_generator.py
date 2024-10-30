from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
from scripts.context_models import StakeholderData, StakeholderRole

class DatasetType(Enum):
    PRIVACY = "privacy"
    ENVIRONMENTAL = "environmental"
    HIGH_RISK = "high_risk"
    CULTURAL = "cultural"
    COMPLIANCE = "compliance"

@dataclass
class TestScenario:
    """Test scenario for AEPF evaluation"""
    name: str
    action: str
    description: str
    context: Dict[str, Any]
    expected_outcome: str
    ethical_considerations: List[str]
    risk_level: str
    compliance_requirements: List[str]
    stakeholders: List[StakeholderData]
    regional_context: Dict[str, Any]
    probability_indicators: Dict[str, float]

class TestDataGenerator:
    """Generates test datasets for AEPF Mk1 evaluation"""
    
    def __init__(self):
        self.privacy_scenarios = self._generate_privacy_scenarios()
        self.environmental_scenarios = self._generate_environmental_scenarios()
        self.high_risk_scenarios = self._generate_high_risk_scenarios()
        self.cultural_scenarios = self._generate_cultural_scenarios()
        self.compliance_scenarios = self._generate_compliance_scenarios()
    
    def _generate_privacy_scenarios(self) -> List[TestScenario]:
        """Generate privacy-focused test scenarios"""
        scenarios = [
            TestScenario(
                name="EU Healthcare Data Processing",
                action="process_patient_data",
                description="Process sensitive healthcare data for research purposes",
                context={
                    "data_type": "medical_records",
                    "privacy_level": "very_high",
                    "region": "DE-BY",
                    "purpose": "medical_research",
                    "data_volume": "large",
                    "processing_location": "EU"
                },
                expected_outcome="REJECT",
                ethical_considerations=[
                    "Patient privacy rights",
                    "Medical research benefits",
                    "Data minimization requirements",
                    "Cross-border data transfer"
                ],
                risk_level="high",
                compliance_requirements=["GDPR", "HIPAA", "Medical Data Protection"],
                stakeholders=[
                    StakeholderData(
                        id=1,
                        name="European Medical Center",
                        role=StakeholderRole.MANAGER,
                        region="DE-BY",
                        priority_level=1,
                        impact_score=90.0
                    )
                ],
                regional_context={
                    "privacy_emphasis": "very_high",
                    "innovation_tolerance": "conservative",
                    "regulatory_framework": "strict",
                    "cultural_values": ["privacy", "security"]
                },
                probability_indicators={
                    "compliance_risk": 0.8,
                    "privacy_breach_risk": 0.7,
                    "benefit_probability": 0.6,
                    "success_rate": 0.5
                }
            ),
            # Add more privacy scenarios...
        ]
        return scenarios
    
    def _generate_environmental_scenarios(self) -> List[TestScenario]:
        """Generate environmental impact test scenarios"""
        scenarios = [
            TestScenario(
                name="Green Data Center Migration",
                action="migrate_to_renewable_energy",
                description="Migrate data center operations to renewable energy sources",
                context={
                    "context_type": "environmental",
                    "current_energy_source": "fossil_fuel",
                    "target_energy_source": "solar_wind_hybrid",
                    "facility_size": "large",
                    "region": "SE-AB",
                    "transition_timeline": "12_months",
                    "carbon_footprint": "high",
                    "environmental_priority": "high"
                },
                expected_outcome="APPROVE",
                ethical_considerations=[
                    "Environmental impact",
                    "Service continuity",
                    "Resource efficiency",
                    "Local community impact"
                ],
                risk_level="moderate",
                compliance_requirements=[
                    "Environmental Protection Act",
                    "Carbon Emission Standards",
                    "Energy Efficiency Regulations"
                ],
                stakeholders=[
                    StakeholderData(
                        id=2,
                        name="Nordic Operations",
                        role=StakeholderRole.MANAGER,
                        region="SE-AB",
                        priority_level=2,
                        impact_score=85.0
                    )
                ],
                regional_context={
                    "environmental_priority": "very_high",
                    "innovation_tolerance": "progressive",
                    "regulatory_framework": "supportive",
                    "cultural_values": ["sustainability", "innovation"]
                },
                probability_indicators={
                    "success_probability": 0.8,
                    "environmental_benefit": 0.9,
                    "operational_risk": 0.4,
                    "cost_effectiveness": 0.7
                }
            ),
            # Add more environmental scenarios...
        ]
        return scenarios
    
    def _generate_high_risk_scenarios(self) -> List[TestScenario]:
        """Generate high-risk, high-reward test scenarios"""
        scenarios = [
            TestScenario(
                name="AI Medical Diagnosis System",
                action="deploy_ai_diagnosis",
                description="Deploy AI system for critical medical diagnosis",
                context={
                    "application_area": "critical_care",
                    "ai_model_accuracy": 0.95,
                    "human_oversight": "required",
                    "deployment_scope": "hospital_wide",
                    "testing_coverage": "extensive",
                    "fallback_systems": "available"
                },
                expected_outcome="REVIEW",
                ethical_considerations=[
                    "Patient safety",
                    "AI decision reliability",
                    "Medical staff autonomy",
                    "Error accountability"
                ],
                risk_level="critical",
                compliance_requirements=[
                    "Medical Device Regulations",
                    "AI Ethics Guidelines",
                    "Healthcare Standards"
                ],
                stakeholders=[
                    StakeholderData(
                        id=3,
                        name="Medical AI Research",
                        role=StakeholderRole.DEVELOPER,
                        region="US-CA",
                        priority_level=1,
                        impact_score=95.0
                    )
                ],
                regional_context={
                    "innovation_tolerance": "progressive",
                    "regulatory_framework": "evolving",
                    "healthcare_standards": "high",
                    "cultural_values": ["innovation", "safety"]
                },
                probability_indicators={
                    "success_rate": 0.85,
                    "risk_factor": 0.7,
                    "benefit_probability": 0.9,
                    "adoption_rate": 0.6
                }
            ),
            # Add more high-risk scenarios...
        ]
        return scenarios
    
    def _generate_cultural_scenarios(self) -> List[TestScenario]:
        """Generate cultural and societal conflict test scenarios"""
        scenarios = [
            TestScenario(
                name="Traditional vs Modern Values",
                action="implement_facial_recognition",
                description="Deploy facial recognition in culturally conservative region",
                context={
                    "region": "JP-13",  # Tokyo, Japan
                    "cultural_context": {
                        "privacy_emphasis": "very_high",
                        "social_harmony": "critical",
                        "innovation_tolerance": "progressive",
                        "primary_values": ["collective_harmony", "privacy"]
                    },
                    "stakeholder": StakeholderData(
                        id=4,
                        name="Tokyo Operations",
                        role=StakeholderRole.MANAGER,
                        region="JP-13",
                        priority_level=1,
                        impact_score=95.0
                    ),
                    "compliance_requirements": ["APPI", "Cultural Heritage Protection"],
                    "risk_level": "high"
                },
                expected_outcome="REVIEW",
                ethical_considerations=[
                    "Cultural privacy expectations",
                    "Social harmony impact",
                    "Traditional vs modern values",
                    "Community consent"
                ],
                risk_level="high",
                compliance_requirements=[
                    "APPI",
                    "Cultural Heritage Protection",
                    "Local Privacy Standards"
                ],
                stakeholders=[
                    StakeholderData(
                        id=4,
                        name="Tokyo Operations",
                        role=StakeholderRole.MANAGER,
                        region="JP-13",
                        priority_level=1,
                        impact_score=95.0
                    )
                ],
                regional_context={
                    "cultural_emphasis": "very_high",
                    "innovation_tolerance": "progressive",
                    "regulatory_framework": "strict",
                    "cultural_values": ["harmony", "privacy", "tradition"]
                },
                probability_indicators={
                    "cultural_acceptance": 0.4,
                    "social_harmony_risk": 0.7,
                    "benefit_probability": 0.6,
                    "community_support": 0.3
                }
            )
        ]
        return scenarios

    def _generate_compliance_scenarios(self) -> List[TestScenario]:
        """Generate compliance-focused test scenarios"""
        scenarios = [
            TestScenario(
                name="Cross-Border Data Transfer",
                action="transfer_sensitive_data",
                description="Transfer sensitive data between EU and non-EU regions",
                context={
                    "data_type": "personal_health",
                    "source_region": "DE-BY",
                    "destination_region": "US-CA",
                    "transfer_volume": "large",
                    "data_sensitivity": "very_high",
                    "stakeholder": StakeholderData(
                        id=5,
                        name="International Health Research",
                        role=StakeholderRole.MANAGER,
                        region="DE-BY",
                        priority_level=1,
                        impact_score=98.0
                    ),
                    "compliance_requirements": [
                        "GDPR",
                        "HIPAA",
                        "EU-US Data Privacy Framework"
                    ],
                    "risk_level": "critical"
                },
                expected_outcome="ESCALATE",
                ethical_considerations=[
                    "Cross-border data protection",
                    "International compliance",
                    "Data sovereignty",
                    "Research benefits"
                ],
                risk_level="critical",
                compliance_requirements=[
                    "GDPR",
                    "HIPAA",
                    "EU-US Data Privacy Framework",
                    "German Federal Data Protection Act"
                ],
                stakeholders=[
                    StakeholderData(
                        id=5,
                        name="International Health Research",
                        role=StakeholderRole.MANAGER,
                        region="DE-BY",
                        priority_level=1,
                        impact_score=98.0
                    )
                ],
                regional_context={
                    "privacy_emphasis": "very_high",
                    "compliance_framework": "strict",
                    "regulatory_requirements": "complex",
                    "cultural_values": ["privacy", "data_protection"]
                },
                probability_indicators={
                    "compliance_risk": 0.9,
                    "data_breach_risk": 0.7,
                    "benefit_probability": 0.8,
                    "successful_transfer": 0.5
                }
            )
        ]
        return scenarios
    
    def save_datasets(self, output_dir: str = "test_data") -> None:
        """Save generated test datasets to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        datasets = {
            "privacy": self.privacy_scenarios,
            "environmental": self.environmental_scenarios,
            "high_risk": self.high_risk_scenarios,
            "cultural": self.cultural_scenarios,
            "compliance": self.compliance_scenarios
        }
        
        for dataset_name, scenarios in datasets.items():
            file_path = output_path / f"{dataset_name}_scenarios.json"
            with open(file_path, 'w') as f:
                json.dump(
                    [self._scenario_to_dict(scenario) for scenario in scenarios],
                    f,
                    indent=2
                )
    
    def _scenario_to_dict(self, scenario: TestScenario) -> Dict:
        """Convert TestScenario to dictionary format with proper serialization"""
        return {
            "name": scenario.name,
            "action": scenario.action,
            "description": scenario.description,
            "context": self._serialize_context(scenario.context),
            "expected_outcome": scenario.expected_outcome,
            "ethical_considerations": scenario.ethical_considerations,
            "risk_level": scenario.risk_level,
            "compliance_requirements": scenario.compliance_requirements,
            "stakeholders": [s.to_dict() for s in scenario.stakeholders],
            "regional_context": scenario.regional_context,
            "probability_indicators": scenario.probability_indicators
        }
    
    def _serialize_context(self, context: Dict) -> Dict:
        """Serialize context data for JSON storage"""
        serialized = {}
        for key, value in context.items():
            if isinstance(value, StakeholderData):
                serialized[key] = value.to_dict()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_context(value)
            else:
                serialized[key] = value
        return serialized

if __name__ == "__main__":
    # Generate and save test datasets
    generator = TestDataGenerator()
    generator.save_datasets() 