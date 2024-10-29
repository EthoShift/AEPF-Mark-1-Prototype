from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
from scripts.decision_analysis.probability_scorer import ProbabilityBand, ProbabilityScore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AdjustmentExplanation:
    """Detailed explanation of an adjustment made during feedback"""
    parameter: str
    original_value: float
    new_value: float
    reason: str
    impact: str

@dataclass
class IterationResult:
    """Results from a single feedback iteration with detailed explanations"""
    iteration_number: int
    predicted_band: ProbabilityBand
    expected_band: ProbabilityBand
    adjustments_made: Dict[str, float]
    adjustment_explanations: List[AdjustmentExplanation]
    confidence_delta: float
    success: bool
    reasoning: str  # Explanation of why adjustments were made

@dataclass
class FeedbackLoopResult:
    """Complete results from feedback loop execution with transparency"""
    initial_prediction: ProbabilityScore
    final_prediction: ProbabilityScore
    iterations: List[IterationResult]
    convergence_achieved: bool
    total_adjustments: Dict[str, float]
    adjustment_history: List[AdjustmentExplanation]
    execution_time: float
    decision_path: str  # Narrative explanation of decision process

class FeedbackLoop:
    """Implements iterative feedback loop for probability score refinement with transparency"""
    
    def __init__(self, max_iterations: int = 5, convergence_threshold: float = 0.1):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.adjustment_rates = {
            'prism_weights': 0.1,
            'cultural_context': 0.1,
            'compliance': 0.1
        }
        self.adjustment_history: List[AdjustmentExplanation] = []
        
    def refine_probability(self,
                         initial_score: ProbabilityScore,
                         expected_band: ProbabilityBand,
                         context: Dict) -> FeedbackLoopResult:
        """Refine probability score through iterative feedback with transparency"""
        start_time = datetime.now()
        iterations: List[IterationResult] = []
        current_score = initial_score
        total_adjustments = {
            'prism_weights': 0.0,
            'cultural_context': 0.0,
            'compliance': 0.0
        }
        
        decision_path = [
            f"Starting refinement process with initial score: {initial_score.adjusted_score:.3f}",
            f"Target probability band: {expected_band.value}"
        ]
        
        for iteration in range(self.max_iterations):
            logger.info(f"\nIteration {iteration + 1}:")
            logger.info(f"Current score: {current_score.adjusted_score:.3f}")
            logger.info(f"Current band: {current_score.band.value}")
            
            # Check if we've reached expected band
            if current_score.band == expected_band:
                decision_path.append(
                    f"Achieved expected band {expected_band.value} "
                    f"after {iteration + 1} iterations"
                )
                return self._create_result(
                    initial_score,
                    current_score,
                    iterations,
                    True,
                    total_adjustments,
                    decision_path,
                    start_time
                )
            
            # Calculate needed adjustments with explanations
            adjustments, explanations, reasoning = self._calculate_adjustments(
                current_score,
                expected_band,
                context,
                iteration
            )
            
            # Apply adjustments
            new_score = self._apply_adjustments(current_score, adjustments)
            
            # Update total adjustments and history
            for key, value in adjustments.items():
                total_adjustments[key] += value
            self.adjustment_history.extend(explanations)
            
            # Record iteration details
            iterations.append(IterationResult(
                iteration_number=iteration + 1,
                predicted_band=current_score.band,
                expected_band=expected_band,
                adjustments_made=adjustments,
                adjustment_explanations=explanations,
                confidence_delta=new_score.confidence_level - current_score.confidence_level,
                success=new_score.band == expected_band,
                reasoning=reasoning
            ))
            
            decision_path.append(
                f"Iteration {iteration + 1}: {reasoning} "
                f"(Score: {new_score.adjusted_score:.3f})"
            )
            
            # Check for convergence
            if self._check_convergence(current_score, new_score):
                decision_path.append(
                    f"Converged after {iteration + 1} iterations "
                    f"(final score: {new_score.adjusted_score:.3f})"
                )
                return self._create_result(
                    initial_score,
                    new_score,
                    iterations,
                    True,
                    total_adjustments,
                    decision_path,
                    start_time
                )
            
            current_score = new_score
            
        # Max iterations reached
        decision_path.append(
            f"Maximum iterations ({self.max_iterations}) reached without convergence. "
            f"Final score: {current_score.adjusted_score:.3f}"
        )
        
        return self._create_result(
            initial_score,
            current_score,
            iterations,
            False,
            total_adjustments,
            decision_path,
            start_time
        )
    
    def _calculate_adjustments(self,
                             current_score: ProbabilityScore,
                             expected_band: ProbabilityBand,
                             context: Dict,
                             iteration: int) -> Tuple[Dict[str, float], List[AdjustmentExplanation], str]:
        """Calculate needed adjustments with detailed explanations"""
        adjustments = {}
        explanations = []
        
        # Calculate direction and magnitude
        if self._band_value(current_score.band) < self._band_value(expected_band):
            direction = 1.0  # Need to increase
            adjustment_reason = "Increasing score to reach higher probability band"
        else:
            direction = -1.0  # Need to decrease
            adjustment_reason = "Decreasing score to reach lower probability band"
        
        magnitude = self._calculate_adjustment_magnitude(
            current_score,
            expected_band,
            iteration
        )
        
        # Apply context-specific adjustments
        if 'innovation_impact' in str(context).lower():
            value = direction * magnitude * self.adjustment_rates['prism_weights']
            adjustments['prism_weights'] = value
            explanations.append(AdjustmentExplanation(
                parameter='prism_weights',
                original_value=current_score.adjusted_score,
                new_value=current_score.adjusted_score + value,
                reason="Innovation context requires prism weight adjustment",
                impact=f"Score change: {value:.3f}"
            ))
        
        # Add other adjustments with explanations...
        
        return adjustments, explanations, adjustment_reason
    
    def _apply_adjustments(self,
                          score: ProbabilityScore,
                          adjustments: Dict[str, float]) -> ProbabilityScore:
        """Apply adjustments to create new probability score"""
        new_adjusted_score = score.adjusted_score
        
        # Apply each adjustment
        for key, adjustment in adjustments.items():
            if key == 'prism_weights':
                new_adjusted_score *= (1.0 + adjustment)
            else:
                new_adjusted_score += adjustment
        
        # Ensure score stays within bounds
        new_adjusted_score = max(min(new_adjusted_score, 1.0), 0.0)
        
        # Determine new probability band
        new_band = self._determine_band(new_adjusted_score)
        
        return ProbabilityScore(
            raw_score=score.raw_score,
            adjusted_score=new_adjusted_score,
            band=new_band,
            influencing_factors=score.influencing_factors,
            cultural_adjustments=score.cultural_adjustments,
            compliance_impacts=score.compliance_impacts,
            confidence_level=score.confidence_level
        )
    
    def _check_convergence(self,
                          previous_score: ProbabilityScore,
                          current_score: ProbabilityScore) -> bool:
        """Check if scores have converged"""
        return abs(current_score.adjusted_score - previous_score.adjusted_score) < self.convergence_threshold
    
    def _band_value(self, band: ProbabilityBand) -> float:
        """Convert probability band to numeric value"""
        band_values = {
            ProbabilityBand.LOW: 0.0,
            ProbabilityBand.MEDIUM: 0.5,
            ProbabilityBand.HIGH: 1.0
        }
        return band_values[band]
    
    def _determine_band(self, score: float) -> ProbabilityBand:
        """Determine probability band from score"""
        if score >= 0.7:
            return ProbabilityBand.HIGH
        elif score >= 0.4:
            return ProbabilityBand.MEDIUM
        else:
            return ProbabilityBand.LOW
    
    def _calculate_adjustment_magnitude(self,
                                     current_score: ProbabilityScore,
                                     expected_band: ProbabilityBand,
                                     iteration: int) -> float:
        """Calculate magnitude of adjustment needed"""
        # Base magnitude on distance between current and expected bands
        distance = abs(self._band_value(current_score.band) - self._band_value(expected_band))
        
        # Reduce magnitude in later iterations for stability
        iteration_factor = 1.0 - (iteration / self.max_iterations)
        
        return distance * iteration_factor
    
    def _create_result(self,
                      initial_score: ProbabilityScore,
                      final_score: ProbabilityScore,
                      iterations: List[IterationResult],
                      converged: bool,
                      total_adjustments: Dict[str, float],
                      decision_path: List[str],
                      start_time: datetime) -> FeedbackLoopResult:
        """Create feedback loop result with complete transparency"""
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return FeedbackLoopResult(
            initial_prediction=initial_score,
            final_prediction=final_score,
            iterations=iterations,
            convergence_achieved=converged,
            total_adjustments=total_adjustments,
            adjustment_history=self.adjustment_history,
            execution_time=execution_time,
            decision_path="\n".join(decision_path)
        ) 