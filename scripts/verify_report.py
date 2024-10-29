import json
from pathlib import Path
from datetime import datetime
import os

def verify_report():
    """Verify and display the latest test report"""
    # Get current date for folder name
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Try multiple possible paths
    possible_paths = [
        Path("reports/core_tests") / date_str,  # From project root
        Path("../reports/core_tests") / date_str,  # From scripts directory
        Path(__file__).parent.parent / "reports/core_tests" / date_str  # Absolute path from script location
    ]
    
    report_found = False
    
    for base_path in possible_paths:
        if base_path.exists():
            # Look for the latest report file
            report_files = list(base_path.glob("core_system_test_*.json"))
            if report_files:
                report_path = sorted(report_files)[-1]  # Get the latest report
                report_found = True
                break
    
    if not report_found:
        print("No report files found. Searched in:")
        for path in possible_paths:
            print(f"- {path.absolute()}")
        return
            
    try:
        with open(report_path, 'r') as f:
            report_data = json.load(f)
            
        print("\nReport Location:")
        print(f"Full path: {report_path.absolute()}")
        print("\nReport Content Summary:")
        print(f"Test ID: {report_data['test_id']}")
        print(f"Status: {report_data['status']}")
        print(f"Components Tested: {', '.join(report_data['components_tested'])}")
        print("\nTest Results:")
        for test_name, result in report_data['results'].items():
            print(f"\n{test_name}:")
            print(f"Status: {result['status']}")
            
        print("\nPerformance Metrics:")
        for metric_name, value in report_data['performance_metrics'].items():
            print(f"{metric_name}: {value}")
            
    except Exception as e:
        print(f"Error reading report: {str(e)}")

if __name__ == "__main__":
    verify_report() 