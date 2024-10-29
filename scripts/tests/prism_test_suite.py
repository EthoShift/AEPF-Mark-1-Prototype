from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np

from scripts.prisms.human_centric import HumanCentricPrism
from scripts.prisms.sentient_first import SentientFirstPrism
from scripts.prisms.ecocentric import EcocentricPrism
from scripts.prisms.innovation_focused import InnovationFocusedPrism
from scripts.prisms.sustainability_prism import SustainabilityPrism

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestScenario:
    """Represents a test scenario for prism evaluation"""
    scenario_id: str
    description: str
    action: str
    context: Dict[str, Any]
    expected_outcomes: Dict[str, float]  # Expected scores per prism
    tags: List[str]  # Categorization tags

@dataclass
class PrismTestResult:
    """Results from testing a scenario across all prisms"""
    scenario_id: str
    timestamp: datetime
    scores: Dict[str, float]
    analysis: Dict[str, str]
    conflicts: List[str]
    alignments: List[str]
    average_score: float

class PrismTestSuite:
    """Test suite for evaluating ethical prisms"""
    
    def __init__(self, output_dir: str = "../reports/prism_tests"):
        self.prisms = {
            'human': HumanCentricPrism(),
            'sentient': SentientFirstPrism(),
            'eco': EcocentricPrism(),
            'innovation': InnovationFocusedPrism(),
            'sustainability': SustainabilityPrism()
        }
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scenarios = self._generate_test_scenarios()
        
    def _generate_test_scenarios(self) -> List[TestScenario]:
        """Generate a set of test scenarios"""
        scenarios = [
            TestScenario(
                scenario_id="SC001",
                description="AI System Automation Upgrade",
                action="implement_automated_decision_system",
                context={
                    "urgency_level": "high",
                    "system_metrics": {"reliability": 0.95},
                    "stakeholder_impact": {"users": 1000, "operators": 50},
                    "environmental_data": {"energy_usage": "moderate"}
                },
                expected_outcomes={
                    'human': 0.7,
                    'sentient': 0.6,
                    'eco': -0.3,
                    'innovation': 0.9,
                    'sustainability': 0.4
                },
                tags=['automation', 'high_impact', 'technical']
            ),
            TestScenario(
                scenario_id="SC002",
                description="Green Energy Transition",
                action="implement_renewable_energy_system",
                context={
                    "urgency_level": "medium",
                    "environmental_data": {"emissions_reduction": 0.8},
                    "stakeholder_impact": {"community": 5000},
                    "resource_usage": {"energy": "high", "materials": "medium"}
                },
                expected_outcomes={
                    'human': 0.5,
                    'sentient': 0.8,
                    'eco': 0.9,
                    'innovation': 0.7,
                    'sustainability': 0.9
                },
                tags=['environmental', 'community_impact', 'sustainable']
            ),
            # Add more scenarios as needed
        ]
        return scenarios
    
    def run_tests(self) -> List[PrismTestResult]:
        """Run all test scenarios through each prism"""
        results = []
        
        for scenario in self.scenarios:
            logger.info(f"Testing scenario: {scenario.scenario_id} - {scenario.description}")
            
            # Evaluate scenario across all prisms
            scores = {}
            for prism_name, prism in self.prisms.items():
                try:
                    evaluation = prism.evaluate(scenario.action, scenario.context)
                    scores[prism_name] = evaluation.impact_score
                except Exception as e:
                    logger.error(f"Error evaluating {prism_name} prism: {str(e)}")
                    scores[prism_name] = 0.0
            
            # Analyze results
            analysis = self._analyze_scores(scores, scenario.expected_outcomes)
            conflicts, alignments = self._identify_conflicts_and_alignments(scores)
            average_score = sum(scores.values()) / len(scores)
            
            # Create test result
            result = PrismTestResult(
                scenario_id=scenario.scenario_id,
                timestamp=datetime.now(),
                scores=scores,
                analysis=analysis,
                conflicts=conflicts,
                alignments=alignments,
                average_score=average_score
            )
            
            results.append(result)
            
        return results
    
    def _analyze_scores(self, 
                       actual_scores: Dict[str, float], 
                       expected_scores: Dict[str, float]) -> Dict[str, str]:
        """Analyze actual scores against expected outcomes"""
        analysis = {}
        
        for prism_name, actual_score in actual_scores.items():
            expected_score = expected_scores.get(prism_name, 0.0)
            difference = actual_score - expected_score
            
            if abs(difference) < 0.1:
                analysis[prism_name] = "Matches expected outcome"
            elif difference > 0:
                analysis[prism_name] = f"Higher than expected (+{difference:.2f})"
            else:
                analysis[prism_name] = f"Lower than expected ({difference:.2f})"
                
        return analysis
    
    def _identify_conflicts_and_alignments(self, 
                                         scores: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """Identify conflicts and alignments between prisms"""
        conflicts = []
        alignments = []
        
        # Compare each pair of prisms
        prisms = list(scores.keys())
        for i in range(len(prisms)):
            for j in range(i + 1, len(prisms)):
                prism1, prism2 = prisms[i], prisms[j]
                score1, score2 = scores[prism1], scores[prism2]
                
                # Check for significant differences
                if abs(score1 - score2) > 0.5:
                    conflicts.append(f"Conflict between {prism1} ({score1:.2f}) and {prism2} ({score2:.2f})")
                elif abs(score1 - score2) < 0.2:
                    alignments.append(f"Alignment between {prism1} ({score1:.2f}) and {prism2} ({score2:.2f})")
                    
        return conflicts, alignments
    
    def generate_report(self, results: List[PrismTestResult]) -> None:
        """Generate comprehensive test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"prism_test_report_{timestamp}"
        
        # Create report directory
        report_path.mkdir(exist_ok=True)
        
        # Generate JSON report
        self._generate_json_report(results, report_path)
        
        # Generate CSV report
        self._generate_csv_report(results, report_path)
        
        # Generate text summary
        self._generate_text_summary(results, report_path)
        
        logger.info(f"Reports generated in: {report_path}")
    
    def _generate_json_report(self, results: List[PrismTestResult], report_path: Path) -> None:
        """Generate detailed JSON report"""
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'total_scenarios': len(results),
            'results': [
                {
                    'scenario_id': r.scenario_id,
                    'timestamp': r.timestamp.isoformat(),
                    'scores': r.scores,
                    'analysis': r.analysis,
                    'conflicts': r.conflicts,
                    'alignments': r.alignments,
                    'average_score': r.average_score
                }
                for r in results
            ]
        }
        
        with open(report_path / 'detailed_report.json', 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _generate_csv_report(self, results: List[PrismTestResult], report_path: Path) -> None:
        """Generate CSV report for easy data analysis"""
        data = []
        for result in results:
            row = {
                'scenario_id': result.scenario_id,
                'timestamp': result.timestamp,
                'average_score': result.average_score
            }
            row.update(result.scores)  # Add individual prism scores
            data.append(row)
            
        df = pd.DataFrame(data)
        df.to_csv(report_path / 'scores.csv', index=False)
    
    def _generate_text_summary(self, results: List[PrismTestResult], report_path: Path) -> None:
        """Generate human-readable text summary"""
        lines = [
            "AEPF Mk1 - Prism Test Results",
            "=" * 80,
            f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Scenarios: {len(results)}",
            "",
            "Summary of Results:",
            "-" * 40
        ]
        
        for result in results:
            lines.extend([
                f"\nScenario: {result.scenario_id}",
                "Scores:",
                *[f"  {prism}: {score:.2f}" for prism, score in result.scores.items()],
                f"Average Score: {result.average_score:.2f}",
                "",
                "Conflicts:",
                *[f"  - {conflict}" for conflict in result.conflicts],
                "",
                "Alignments:",
                *[f"  - {alignment}" for alignment in result.alignments],
                "-" * 40
            ])
        
        with open(report_path / 'summary.txt', 'w') as f:
            f.write('\n'.join(lines))

if __name__ == "__main__":
    # Run test suite
    test_suite = PrismTestSuite()
    results = test_suite.run_tests()
    test_suite.generate_report(results) 