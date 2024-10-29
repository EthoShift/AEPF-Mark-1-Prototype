import sys
import os
from pathlib import Path

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.test_core_system import test_core_system
from scripts.report_templates import ReportManager

if __name__ == "__main__":
    # Initialize report manager with explicit path
    project_root = Path(__file__).parent
    reports_dir = project_root / "reports"
    report_manager = ReportManager(reports_dir)
    
    # Run test and get report with sequential ID
    test_report = test_core_system()
    test_report.test_id = report_manager.get_next_id()
    
    # Save reports
    json_path, text_path = report_manager.save_report(test_report)
    print(f"\nReports saved to:")
    print(f"JSON: {json_path}")
    print(f"Text: {text_path}")
    
    # Display text report content
    print("\nTest Report Content:")
    print("-" * 80)
    with open(text_path, 'r') as f:
        print(f.read())
    
    # Load and display report summary from JSON
    report_data = report_manager.load_report(json_path)
    if report_data:
        print("\nTest Report Summary:")
        print(f"Test ID: {report_data['test_id']}")
        print(f"Status: {report_data['status']}")
        print(f"Timestamp: {report_data['timestamp']}")
        print(f"Components Tested: {', '.join(report_data['components_tested'])}")
        if report_data['errors']:
            print(f"Errors: {report_data['errors']}") 