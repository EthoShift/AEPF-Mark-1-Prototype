from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path
from enum import Enum

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
    
    def format_text_report(self, report: TestReport) -> str:
        """Format report data as readable text"""
        lines = [
            "=" * 80,
            "AEPF Mk1 - Artificial Ethical Processing Framework",
            "Core System Integration Test Report",
            "=" * 80,
            "",
            f"Report ID: {report.test_id}",
            f"Test Type: Core System Integration and Communication Test",
            f"Execution Date: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Test Scope:",
            "-" * 40,
            "- Component Integration Testing",
            "- Ethical Decision Making Verification",
            "- Context Management System",
            "- Multi-Prism Ethical Analysis",
            "",
            f"Components Tested: {', '.join(report.components_tested)}",
            "",
            "Test Results:",
            "-" * 40
        ]
        
        # Add detailed results with better formatting
        for test_name, result in report.results.items():
            lines.extend([
                f"\n{test_name.upper()}:",
                f"Status: {result['status']}"
            ])
            
            # Format additional result details
            for key, value in result.items():
                if key != 'status':
                    if isinstance(value, dict):
                        lines.append(f"\n{key}:")
                        for k, v in value.items():
                            lines.append(f"  {k}: {v}")
                    elif isinstance(value, str) and len(value) > 100:
                        # Break long strings into multiple lines
                        lines.append(f"\n{key}:")
                        wrapped_lines = [value[i:i+80] for i in range(0, len(value), 80)]
                        for line in wrapped_lines:
                            lines.append(f"  {line}")
                    else:
                        lines.append(f"{key}: {value}")
        
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
        
        # Add errors if any
        if report.errors:
            lines.extend([
                "",
                "Errors and Warnings:",
                "-" * 40
            ])
            for error in report.errors:
                lines.append(f"- {error}")
        
        # Add summary footer with more details
        lines.extend([
            "",
            "=" * 80,
            f"Final Status: {report.status.upper()}",
            f"Test Completion Time: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "Generated by AEPF Mk1 Test Framework",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    def save_report(self, report: TestReport) -> tuple[str, str]:
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
    
    def generate_report_path(self, report: TestReport) -> Path:
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