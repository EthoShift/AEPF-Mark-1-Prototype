from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import pandas as pd

@dataclass
class DecisionPattern:
    """Pattern data for decision analysis"""
    scenario_type: str
    initial_confidence: float
    final_confidence: float
    initial_band: str
    final_band: str
    iterations_count: int
    adjustments_made: Dict[str, float]
    success: bool
    timestamp: datetime

class PatternLogger:
    """Logs and analyzes decision-making patterns"""
    
    def __init__(self, log_dir: str = "logs/patterns"):
        self.logger = logging.getLogger(__name__)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.patterns: List[DecisionPattern] = []
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging handlers"""
        file_handler = logging.FileHandler(
            self.log_dir / f"pattern_analysis_{self.current_session}.log"
        )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_decision_pattern(self,
                           scenario_type: str,
                           initial_confidence: float,
                           final_confidence: float,
                           initial_band: str,
                           final_band: str,
                           iterations: List[Any],
                           adjustments: Dict[str, float],
                           success: bool):
        """Log a decision pattern"""
        pattern = DecisionPattern(
            scenario_type=scenario_type,
            initial_confidence=initial_confidence,
            final_confidence=final_confidence,
            initial_band=initial_band,
            final_band=final_band,
            iterations_count=len(iterations),
            adjustments_made=adjustments,
            success=success,
            timestamp=datetime.now()
        )
        
        self.patterns.append(pattern)
        self._log_pattern(pattern)
        self._analyze_trends()
    
    def _log_pattern(self, pattern: DecisionPattern):
        """Log pattern details"""
        self.logger.info(
            f"Decision Pattern - Type: {pattern.scenario_type}\n"
            f"Confidence: {pattern.initial_confidence:.2f} -> {pattern.final_confidence:.2f}\n"
            f"Band: {pattern.initial_band} -> {pattern.final_band}\n"
            f"Iterations: {pattern.iterations_count}\n"
            f"Adjustments: {json.dumps(pattern.adjustments_made, indent=2)}\n"
            f"Success: {pattern.success}"
        )
    
    def _analyze_trends(self):
        """Analyze patterns for trends and misalignments"""
        if len(self.patterns) < 5:  # Need minimum patterns for analysis
            return
        
        # Convert patterns to DataFrame for analysis
        df = pd.DataFrame([
            {
                'scenario_type': p.scenario_type,
                'confidence_delta': p.final_confidence - p.initial_confidence,
                'iterations': p.iterations_count,
                'success': p.success,
                'timestamp': p.timestamp
            }
            for p in self.patterns
        ])
        
        # Analyze by scenario type
        type_analysis = df.groupby('scenario_type').agg({
            'confidence_delta': 'mean',
            'iterations': 'mean',
            'success': 'mean'
        })
        
        # Identify potential issues
        for scenario_type, stats in type_analysis.iterrows():
            if stats['success'] < 0.7:  # Less than 70% success rate
                self.logger.warning(
                    f"Low success rate for {scenario_type}: {stats['success']:.2f}\n"
                    f"Avg confidence adjustment: {stats['confidence_delta']:.2f}\n"
                    f"Avg iterations: {stats['iterations']:.1f}"
                )
            
            if stats['iterations'] > 2:  # High iteration count
                self.logger.warning(
                    f"High iteration count for {scenario_type}: {stats['iterations']:.1f}"
                )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate analysis report"""
        if not self.patterns:
            return {"status": "No patterns logged"}
        
        # Calculate success rates by scenario type
        success_rates = {}
        iteration_rates = {}
        confidence_changes = {}
        
        for pattern in self.patterns:
            if pattern.scenario_type not in success_rates:
                success_rates[pattern.scenario_type] = []
                iteration_rates[pattern.scenario_type] = []
                confidence_changes[pattern.scenario_type] = []
            
            success_rates[pattern.scenario_type].append(pattern.success)
            iteration_rates[pattern.scenario_type].append(pattern.iterations_count)
            confidence_changes[pattern.scenario_type].append(
                pattern.final_confidence - pattern.initial_confidence
            )
        
        report = {
            "session_id": self.current_session,
            "total_decisions": len(self.patterns),
            "scenario_types": {
                scenario_type: {
                    "success_rate": sum(rates) / len(rates),
                    "avg_iterations": sum(iteration_rates[scenario_type]) / len(rates),
                    "avg_confidence_change": sum(confidence_changes[scenario_type]) / len(rates)
                }
                for scenario_type, rates in success_rates.items()
            },
            "recommendations": self._generate_recommendations(success_rates, iteration_rates)
        }
        
        # Save report
        report_path = self.log_dir / f"pattern_report_{self.current_session}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_recommendations(self,
                                success_rates: Dict[str, List[bool]],
                                iteration_rates: Dict[str, List[int]]) -> List[str]:
        """Generate recommendations based on pattern analysis"""
        recommendations = []
        
        for scenario_type, rates in success_rates.items():
            success_rate = sum(rates) / len(rates)
            avg_iterations = sum(iteration_rates[scenario_type]) / len(rates)
            
            if success_rate < 0.7:
                recommendations.append(
                    f"Review decision criteria for {scenario_type} scenarios "
                    f"(success rate: {success_rate:.2f})"
                )
            
            if avg_iterations > 2:
                recommendations.append(
                    f"Optimize initial confidence calculation for {scenario_type} scenarios "
                    f"(avg iterations: {avg_iterations:.1f})"
                )
        
        return recommendations 