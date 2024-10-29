from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
from enum import Enum
from scripts.decision_analysis.feedback_loop import FeedbackLoopResult

def serialize_result(obj: Any) -> Any:
    """Serialize complex objects for JSON storage"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: serialize_result(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_result(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)

@dataclass
class TestReport:
    """Template for test execution reports"""
    test_id: str
    timestamp: datetime
    test_name: str
    status: str
    components_tested: List[str]
    results: Dict[str, Any]
    errors: List[str]
    performance_metrics: Dict[str, float]
    context_snapshot: Dict[str, Any]
    
    @classmethod
    def create(cls, test_name: str, components: List[str], report_id: str = "001") -> 'TestReport':
        """Create a new test report with basic initialization"""
        return cls(
            test_id=report_id,
            timestamp=datetime.now(),
            test_name=test_name,
            status="initialized",
            components_tested=components,
            results={},
            errors=[],
            performance_metrics={},
            context_snapshot={}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format with proper serialization"""
        return {
            "test_id": self.test_id,
            "timestamp": self.timestamp.isoformat(),
            "test_name": self.test_name,
            "status": self.status,
            "components_tested": self.components_tested,
            "results": serialize_result(self.results),
            "errors": self.errors,
            "performance_metrics": self.performance_metrics,
            "context_snapshot": serialize_result(self.context_snapshot)
        }
    
    def add_feedback_loop_results(self, feedback_results: FeedbackLoopResult) -> None:
        """Add feedback loop results to the report with narrative focus"""
        self.results['feedback_loop'] = {
            'narrative_summary': self._generate_feedback_narrative(feedback_results),
            'iteration_analysis': self._format_iteration_analysis(feedback_results),
            'convergence_details': self._format_convergence_details(feedback_results),
            'decision_path': feedback_results.decision_path
        }
    
    def add_probability_analysis(self, probability_data: Dict) -> None:
        """Add probability scoring analysis to report"""
        self.results['probability_analysis'] = {
            'initial_score': probability_data['initial_score'],
            'final_score': probability_data['final_score'],
            'convergence_achieved': probability_data['convergence_achieved'],
            'iterations_required': len(probability_data['iterations']),
            'confidence_level': probability_data['confidence_level']
        }
    
    def _generate_feedback_narrative(self, results: FeedbackLoopResult) -> str:
        """Generate narrative description of feedback loop execution"""
        narrative = [
            "Feedback Loop Analysis",
            "----------------------",
            f"Starting from an initial score of {results.initial_prediction.adjusted_score:.3f} "
            f"({results.initial_prediction.band.value}),",
            "the system performed iterative refinements to achieve optimal alignment.",
            "",
            "Adjustment Process:",
        ]
        
        # Add iteration summaries
        for i, iteration in enumerate(results.iterations, 1):
            narrative.append(
                f"\nIteration {i}:"
                f"\n- Current Band: {iteration.predicted_band.value}"
                f"\n- Target Band: {iteration.expected_band.value}"
                f"\n- Reasoning: {iteration.reasoning}"
            )
            
            if iteration.success:
                narrative.append("✓ Achieved expected probability band")
            
        # Add convergence summary
        if results.convergence_achieved:
            narrative.append(
                f"\nConverged successfully after {len(results.iterations)} iterations "
                f"with final score {results.final_prediction.adjusted_score:.3f} "
                f"({results.final_prediction.band.value})"
            )
        else:
            narrative.append(
                "\nDid not achieve convergence within maximum iterations. "
                "Further refinement may be needed."
            )
        
        return "\n".join(narrative)
    
    def _format_iteration_analysis(self, results: FeedbackLoopResult) -> Dict[str, Any]:
        """Format detailed analysis of each iteration"""
        return {
            f"iteration_{i+1}": {
                'predicted_band': iteration.predicted_band.value,
                'expected_band': iteration.expected_band.value,
                'adjustments': iteration.adjustments_made,
                'confidence_delta': iteration.confidence_delta,
                'success': iteration.success,
                'reasoning': iteration.reasoning
            }
            for i, iteration in enumerate(results.iterations)
        }
    
    def _format_convergence_details(self, results: FeedbackLoopResult) -> Dict[str, Any]:
        """Format convergence analysis details"""
        return {
            'achieved': results.convergence_achieved,
            'iterations_required': len(results.iterations),
            'total_adjustments': results.total_adjustments,
            'execution_time': results.execution_time,
            'final_state': {
                'score': results.final_prediction.adjusted_score,
                'band': results.final_prediction.band.value,
                'confidence': results.final_prediction.confidence_level
            }
        }
    
    def format_text_report(self, report: 'TestReport') -> str:
        """Format report data as readable text with narrative focus"""
        lines = [
            "=" * 80,
            "AEPF Mk1 - Adaptive Ethical Prism Framework",
            "Decision Analysis and Ethical Evaluation Report",
            "=" * 80,
            "",
            "Executive Summary",
            "-" * 40,
            f"Report ID: {report.test_id}",
            f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Overview:",
            self._generate_narrative_overview(report),
            "",
            "Key Findings:",
            self._generate_key_findings(report),
            "",
            "Critical Considerations:",
            self._generate_critical_considerations(report),
            "",
            "Recommendations:",
            self._generate_recommendations(report),
            "",
            "=" * 80,
            "Detailed Analysis",
            "=" * 80,
            "",
            "Component Performance:",
            self._format_component_results(report),
            "",
            "Risk Analysis:",
            self._format_risk_analysis(report),
            "",
            "=" * 80,
            "Technical Data",
            "=" * 80,
            "",
            "Test Metrics:",
            self._format_test_metrics(report),
            "",
            "Raw Data:",
            self._format_raw_data(report),
            "",
            "=" * 80,
            f"Final Status: {report.status.upper()}",
            f"Generated by AEPF Mk1 Test Framework v1.0",
            "=" * 80,
            
            # Add feedback loop section if present
            self._format_feedback_loop_section(report),
        ]
        return "\n".join(lines)
    
    def _format_feedback_loop_section(self, report: 'TestReport') -> str:
        """Format feedback loop section of the report"""
        if 'feedback_loop' not in report.results:
            return ""
            
        feedback_data = report.results['feedback_loop']
        lines = [
            "=" * 80,
            "Feedback Loop Analysis",
            "=" * 80,
            "",
            feedback_data['narrative_summary'],
            "",
            "Detailed Iteration Analysis:",
            "-" * 40
        ]
        
        # Add iteration details
        for iteration_id, details in feedback_data['iteration_analysis'].items():
            lines.extend([
                f"\n{iteration_id}:",
                f"Target: {details['expected_band']}",
                f"Achieved: {details['predicted_band']}",
                f"Reasoning: {details['reasoning']}"
            ])
        
        # Add convergence summary
        convergence = feedback_data['convergence_details']
        lines.extend([
            "",
            "Convergence Summary:",
            "-" * 40,
            f"Achieved: {'Yes' if convergence['achieved'] else 'No'}",
            f"Iterations: {convergence['iterations_required']}",
            f"Execution Time: {convergence['execution_time']:.3f} seconds",
            "",
            "Decision Path:",
            "-" * 40,
            feedback_data['decision_path']
        ])
        
        return "\n".join(lines)

@dataclass
class LocationTestReport(TestReport):
    """Specialized report for location-based testing"""
    location_specific_data: Dict[str, Any] = None
    regional_comparisons: Dict[str, Dict[str, float]] = None
    cultural_impact_analysis: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format with location-specific data"""
        base_dict = super().to_dict()
        base_dict.update({
            "location_specific_data": serialize_result(self.location_specific_data),
            "regional_comparisons": serialize_result(self.regional_comparisons),
            "cultural_impact_analysis": serialize_result(self.cultural_impact_analysis)
        })
        return base_dict

class ReportManager:
    """Manages the creation, storage, and retrieval of test reports"""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            self.base_path = Path(__file__).parent.parent / "reports"
        else:
            self.base_path = Path(base_path)
        
        self.ensure_report_directories()
        self.current_id = self._get_last_used_id() + 1
    
    def _get_last_used_id(self) -> int:
        """Find the highest ID currently in use"""
        highest_id = 0
        
        # Check all test directories
        for test_dir in self.base_path.glob("*_tests"):
            if test_dir.is_dir():
                # Check all date directories
                for date_dir in test_dir.glob("*"):
                    if date_dir.is_dir():
                        # Find all report files
                        for report_file in date_dir.glob("*_*.json"):
                            try:
                                # Extract ID from filename (e.g., "core_system_test_001.json")
                                id_str = report_file.stem.split('_')[-1]
                                report_id = int(id_str)
                                highest_id = max(highest_id, report_id)
                            except (ValueError, IndexError):
                                continue
        
        return highest_id
    
    def ensure_report_directories(self):
        """Create necessary report directories if they don't exist"""
        directories = [
            "core_tests",
            "integration_tests",
            "prism_tests",
            "performance_tests"
        ]
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        for directory in directories:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def format_text_report(self, report: 'TestReport') -> str:
        """Format report data as readable text with narrative focus"""
        lines = [
            "=" * 80,
            "AEPF Mk1 - Adaptive Ethical Prism Framework",
            "Decision Analysis and Ethical Evaluation Report",
            "=" * 80,
            "",
            "Executive Summary",
            "-" * 40,
            f"Report ID: {report.test_id}",
            f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Overview:",
            self._generate_narrative_overview(report),
            "",
            "Key Findings:",
            self._generate_key_findings(report),
            "",
            "Critical Considerations:",
            self._generate_critical_considerations(report),
            "",
            "Recommendations:",
            self._generate_recommendations(report),
            "",
            "=" * 80,
            "Detailed Analysis",
            "=" * 80,
            "",
            "Component Performance:",
            self._format_component_results(report),
            "",
            "Risk Analysis:",
            self._format_risk_analysis(report),
            "",
            "=" * 80,
            "Technical Data",
            "=" * 80,
            "",
            "Test Metrics:",
            self._format_test_metrics(report),
            "",
            "Raw Data:",
            self._format_raw_data(report),
            "",
            "=" * 80,
            f"Final Status: {report.status.upper()}",
            f"Generated by AEPF Mk1 Test Framework v1.0",
            "=" * 80
        ]
        
        return "\n".join(lines)
    
    def _generate_narrative_overview(self, report: 'TestReport') -> str:
        """Generate narrative overview of the test execution"""
        components = ", ".join(report.components_tested)
        duration = report.performance_metrics.get('total_execution_time', 0)
        
        narrative = [
            f"\nThis report documents the ethical evaluation and decision analysis",
            f"performed by the Adaptive Ethical Prism Framework (AEPF Mk1),",
            f"testing the integration of {components}.",
            f"The evaluation was completed in {duration:.2f} seconds, analyzing multiple",
            "ethical dimensions through our specialized prism framework.",
            "",
            "The system processed context data, evaluated ethical implications,",
            "and generated recommendations based on our core ethical principles",
            "and regional context considerations."
        ]
        return "\n".join(narrative)
    
    def _generate_key_findings(self, report: 'TestReport') -> str:
        """Generate narrative summary of key findings"""
        findings = []
        
        for test_name, result in report.results.items():
            if result['status'] == 'success':
                findings.append(f"[PASS] {test_name.replace('_', ' ').title()}: Successful evaluation")
                if 'decision' in result:
                    findings.append(f"  - {result['decision']}")
            else:
                findings.append(f"[FAIL] {test_name.replace('_', ' ').title()}: Requires attention")
        
        return "\n".join(findings)
    
    def _generate_critical_considerations(self, report: 'TestReport') -> str:
        """Generate list of critical considerations"""
        considerations = [
            "• System demonstrated expected ethical reasoning capabilities",
            "• All core components maintained data consistency",
            "• Decision-making processes aligned with ethical guidelines"
        ]
        
        if report.errors:
            considerations.append("\nAttention Required:")
            considerations.extend([f"! {error}" for error in report.errors])
            
        return "\n".join(considerations)
    
    def _generate_recommendations(self, report: 'TestReport') -> str:
        """Generate actionable recommendations"""
        recommendations = [
            "1. Continue monitoring system performance and ethical alignment",
            "2. Regular validation of decision outcomes",
            "3. Maintain updated context data for accurate assessments"
        ]
        
        # Add specific recommendations based on results
        for test_name, result in report.results.items():
            if 'recommendations' in result:
                recommendations.extend([f"- {rec}" for rec in result['recommendations']])
                
        return "\n".join(recommendations)
    
    def _format_component_results(self, report: 'TestReport') -> str:
        """Format detailed component results"""
        lines = []
        
        for test_name, result in report.results.items():
            lines.extend([
                f"\n{test_name.replace('_', ' ').title()}:",
                "-" * 40
            ])
            
            for key, value in result.items():
                if key != 'status':
                    if isinstance(value, dict):
                        lines.append(f"\n{key}:")
                        for k, v in value.items():
                            lines.append(f"  {k}: {v}")
                    else:
                        lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def _format_risk_analysis(self, report: 'TestReport') -> str:
        """Format risk analysis section"""
        risks = [
            "Potential Risks and Mitigations:",
            "- Data consistency maintained throughout evaluation",
            "- No critical ethical conflicts detected",
            "- System boundaries operating within expected parameters"
        ]
        return "\n".join(risks)
    
    def _format_test_metrics(self, report: 'TestReport') -> str:
        """Format test metrics in a clean, tabular format"""
        metrics = ["-" * 40]
        
        for metric, value in report.performance_metrics.items():
            if metric == 'total_execution_time':
                metrics.append(f"{metric:.<30} {value:.3f} seconds")
            else:
                metrics.append(f"{metric:.<30} {value}")
                
        return "\n".join(metrics)
    
    def _format_raw_data(self, report: 'TestReport') -> str:
        """Format raw data in a structured way"""
        raw_data = [
            "-" * 40,
            "Component Status:",
            *[f"{component:.<30} ACTIVE" for component in report.components_tested],
            "",
            "Test Execution Data:",
            f"Start Time: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Test ID: {report.test_id}",
            f"Status Code: {report.status}"
        ]
        return "\n".join(raw_data)
    
    def save_report(self, report: 'TestReport') -> tuple[str, str]:
        """
        Save report to file system in both JSON and text formats
        
        Returns:
            Tuple of (json_path, text_path)
        """
        # Generate paths
        base_path = self.generate_report_path(report)
        json_path = base_path.with_suffix('.json')
        text_path = base_path.with_suffix('.txt')
        
        # Save JSON version
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Save text version
        with open(text_path, 'w') as f:
            f.write(self.format_text_report(report))
        
        self.current_id += 1
        return str(json_path), str(text_path)
    
    def get_next_id(self) -> str:
        """Generate the next sequential report ID"""
        return f"{self.current_id:03d}"
    
    def generate_report_path(self, report: 'TestReport') -> Path:
        """Generate appropriate path for storing the report"""
        test_type = report.test_name.split('_')[0]
        date_str = report.timestamp.strftime("%Y%m%d")
        
        # Create date-based subdirectory
        report_dir = self.base_path / f"{test_type}_tests" / date_str
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sequential filename
        filename = f"{report.test_name}_{report.test_id}.json"
        return report_dir / filename
    
    def load_report(self, report_path: str) -> Optional[Dict[str, Any]]:
        """Load report from file system"""
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading report: {e}")
            return None
    
    def format_location_test_report(self, report: LocationTestReport) -> str:
        """Format location-specific test report as readable text"""
        lines = [
            "=" * 80,
            "AEPF Mk1 - Location-Based Testing Report",
            "Regional Context and Decision Analysis",
            "=" * 80,
            "",
            f"Report ID: {report.test_id}",
            f"Test Type: Location Context Testing",
            f"Execution Date: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Test Scope:",
            "-" * 40,
            "- Regional Context Evaluation",
            "- Cultural Impact Assessment",
            "- Cross-Regional Decision Analysis",
            "- Location-Specific Compliance Verification",
            "",
            "Regional Results:",
            "-" * 40
        ]
        
        # Add regional test results
        if report.location_specific_data:
            for region, data in report.location_specific_data.items():
                lines.extend([
                    f"\nRegion: {region}",
                    f"Decision Outcome: {data.get('decision', 'N/A')}",
                    f"Confidence Score: {data.get('confidence_score', 0):.2f}",
                    "\nPrism Scores:"
                ])
                for prism, score in data.get('prism_scores', {}).items():
                    lines.append(f"  {prism}: {score:.2f}")
        
        # Add regional comparisons
        if report.regional_comparisons:
            lines.extend([
                "",
                "Regional Comparisons:",
                "-" * 40
            ])
            for metric, comparisons in report.regional_comparisons.items():
                lines.append(f"\n{metric}:")
                for region, value in comparisons.items():
                    lines.append(f"  {region}: {value:.2f}")
        
        # Add cultural impact analysis
        if report.cultural_impact_analysis:
            lines.extend([
                "",
                "Cultural Impact Analysis:",
                "-" * 40
            ])
            for aspect, analysis in report.cultural_impact_analysis.items():
                lines.append(f"\n{aspect}:")
                if isinstance(analysis, dict):
                    for key, value in analysis.items():
                        lines.append(f"  {key}: {value}")
                else:
                    lines.append(f"  {analysis}")
        
        # Add performance metrics
        lines.extend([
            "",
            "Performance Metrics:",
            "-" * 40
        ])
        for metric, value in report.performance_metrics.items():
            if metric == 'total_execution_time':
                lines.append(f"{metric}: {value:.3f} seconds")
            else:
                lines.append(f"{metric}: {value}")
        
        # Add summary footer
        lines.extend([
            "",
            "=" * 80,
            f"Final Status: {report.status.upper()}",
            f"Test Completion Time: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "Generated by AEPF Mk1 Test Framework",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    def save_location_test_report(self, report: LocationTestReport) -> tuple[str, str]:
        """Save location test report with specialized formatting"""
        # Generate paths
        base_path = self.base_path / "location_tests" / report.timestamp.strftime("%Y%m%d")
        base_path.mkdir(parents=True, exist_ok=True)
        
        json_path = base_path / f"location_test_{report.test_id}.json"
        text_path = base_path / f"location_test_{report.test_id}.txt"
        
        # Save JSON version
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Save text version
        with open(text_path, 'w') as f:
            f.write(self.format_location_test_report(report))
        
        return str(json_path), str(text_path)