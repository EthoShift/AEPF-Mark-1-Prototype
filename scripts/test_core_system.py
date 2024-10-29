import logging
from datetime import datetime
from typing import Dict, Any
import time
from pathlib import Path

from scripts.context_engine import ContextEngine
from scripts.context_models import StakeholderData, RealTimeMetrics, ContextEntry
from scripts.ethical_governor import EthicalGovernor
from scripts.prisms.human_centric import HumanCentricPrism
from scripts.prisms.sentient_first import SentientFirstPrism
from scripts.prisms.ecocentric import EcocentricPrism
from scripts.prisms.innovation_focused import InnovationFocusedPrism
from scripts.prisms.accountability_prism import AccountabilityPrism
from scripts.report_templates import TestReport, ReportManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_core_system() -> TestReport:
    """Test basic communication between core system components"""
    
    # Initialize report
    report = TestReport.create(
        test_name="core_system_test",
        components=["ContextEngine", "EthicalGovernor", "AccountabilityPrism"]
    )
    
    start_time = time.time()
    
    try:
        logger.info("Initializing core system components...")
        
        # Initialize core components
        context_engine = ContextEngine()
        ethical_governor = EthicalGovernor()
        accountability_prism = AccountabilityPrism()
        
        # Test 1: Context Management
        logger.info("\nTest 1: Testing Context Management...")
        
        # Create test stakeholder
        stakeholder = StakeholderData(
            id=1,
            name="Test User",
            role="user",
            region="North America",
            priority_level=2,
            impact_score=75.0
        )
        
        # Create test metric
        metric = RealTimeMetrics(
            metric_name="system_reliability",
            value=95.5,
            timestamp=datetime.now(),
            source="sensor"
        )
        
        # Create context entries
        stakeholder_entry = ContextEntry(
            entry_type="stakeholder",
            data=stakeholder
        )
        
        metric_entry = ContextEntry(
            entry_type="metric",
            data=metric
        )
        
        # Add to context engine
        context_engine.add_context_entry(stakeholder_entry)
        context_engine.add_context_entry(metric_entry)
        
        # Verify context storage
        context_analysis = context_engine.analyze_context()
        logger.info(f"Context Analysis: {context_analysis}")
        
        # Test 2: Ethical Evaluation
        logger.info("\nTest 2: Testing Ethical Evaluation...")
        
        # Create test action
        test_action = "deploy_system_update"
        test_context = {
            "stakeholder": stakeholder,
            "system_metrics": metric,
            "urgency_level": "high"
        }
        
        # Evaluate through ethical governor
        decision = ethical_governor.evaluate_action(test_action, test_context)
        logger.info(f"Ethical Decision: {decision}")
        
        # Test 3: Accountability Tracking
        logger.info("\nTest 3: Testing Accountability Tracking...")
        
        # Log the decision
        accountability_prism.log_decision(
            decision_id=1,
            decision_data={
                "context": test_context,
                "action": test_action,
                "rationale": "System update required for security",
                "impact_level": "medium",
                "responsible_entity": "system"
            }
        )
        
        # Assign responsibility
        accountability_prism.assign_responsibility(1, "supervisor")
        
        # Review decision
        review_data = accountability_prism.review_decision(1)
        logger.info(f"Decision Review Data: {review_data}")
        
        # Test 4: Cross-Component Integration
        logger.info("\nTest 4: Testing Cross-Component Integration...")
        
        # Get accountability assessment
        assessment = accountability_prism.evaluate(test_action, test_context)
        logger.info(f"Accountability Assessment: {assessment}")
        
        # Update context with assessment
        assessment_entry = ContextEntry(
            entry_type="assessment",
            data={
                "type": "accountability",
                "score": assessment.transparency_score,
                "timestamp": datetime.now()
            }
        )
        context_engine.add_context_entry(assessment_entry)
        
        # Record results for each test phase
        report.results["context_management"] = {
            "status": "success",
            "context_analysis": context_analysis
        }
        
        # Record ethical evaluation results
        report.results["ethical_evaluation"] = {
            "status": "success",
            "decision": str(decision)
        }
        
        # Record accountability results
        report.results["accountability_tracking"] = {
            "status": "success",
            "review_data": review_data
        }
        
        # Record integration results
        report.results["cross_component_integration"] = {
            "status": "success",
            "assessment": str(assessment)
        }
        
        # Calculate performance metrics
        end_time = time.time()
        report.performance_metrics = {
            "total_execution_time": end_time - start_time,
            "context_store_size": len(context_engine.context_store),
            "decisions_recorded": len(accountability_prism.decision_log)
        }
        
        report.status = "completed"
        logger.info("\nCore system testing completed successfully.")
        
    except Exception as e:
        report.status = "failed"
        report.errors.append(str(e))
        logger.error(f"Error during core system test: {str(e)}", exc_info=True)
    
    return report

if __name__ == "__main__":
    # Initialize report manager
    report_manager = ReportManager()
    
    # Run test and get report with sequential ID
    test_report = test_core_system()
    test_report.test_id = report_manager.get_next_id()  # Set sequential ID
    
    # Save report
    report_path = report_manager.save_report(test_report)
    logger.info(f"Test report saved to: {report_path}")
    
    # Load and display report summary
    report_data = report_manager.load_report(report_path)
    if report_data:
        logger.info("\nTest Report Summary:")
        logger.info(f"Test ID: {report_data['test_id']}")
        logger.info(f"Status: {report_data['status']}")
        logger.info(f"Timestamp: {report_data['timestamp']}")
        logger.info(f"Components Tested: {', '.join(report_data['components_tested'])}")
        if report_data['errors']:
            logger.error(f"Errors: {report_data['errors']}") 