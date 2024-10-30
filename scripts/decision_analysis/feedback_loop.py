from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
from scripts.decision_analysis.probability_scorer import ProbabilityBand, ProbabilityScore
from scripts.decision_analysis.pattern_logger import PatternLogger

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
    """Refines probability scores through iterative feedback"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_iterations = 3
        self.convergence_threshold = 0.01
        self.adjustment_history: List[AdjustmentExplanation] = []
        self.pattern_logger = PatternLogger()  # Add pattern logger
        self.adjustment_rates = {
            'prism_weights': 0.1,
            'cultural_context': 0.1,
            'compliance': 0.1
        }
    
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
            'compliance': 0.0,
            'scenario_specific': 0.0
        }
        
        decision_path = [
            f"Starting refinement process with initial score: {initial_score.adjusted_score:.3f}",
            f"Target probability band: {expected_band.value}"
        ]
        
        # First layer: Standard probability refinement
        current_score, layer1_iterations = self._standard_refinement(
            current_score, expected_band, context, decision_path, total_adjustments
        )
        iterations.extend(layer1_iterations)
        
        # Second layer: Scenario-specific adjustments
        if current_score.band != expected_band:
            current_score, layer2_iterations = self._scenario_refinement(
                current_score, expected_band, context, decision_path, total_adjustments
            )
            iterations.extend(layer2_iterations)
        
        # Final layer: Band-specific fine-tuning
        if current_score.band != expected_band:
            current_score, layer3_iterations = self._band_refinement(
                current_score, expected_band, context, decision_path, total_adjustments
            )
            iterations.extend(layer3_iterations)
        
        # Log pattern data
        self.pattern_logger.log_decision_pattern(
            scenario_type=str(context.get('context_type', 'unknown')),
            initial_confidence=initial_score.confidence_level,
            final_confidence=current_score.confidence_level,
            initial_band=initial_score.band.value,
            final_band=current_score.band.value,
            iterations=iterations,
            adjustments=total_adjustments,
            success=current_score.band == expected_band
        )
        
        return self._create_result(
            initial_score,
            current_score,
            iterations,
            current_score.band == expected_band,
            total_adjustments,
            decision_path,
            start_time
        )
    
    def _standard_refinement(self,
                             current_score: ProbabilityScore,
                             expected_band: ProbabilityBand,
                             context: Dict,
                             decision_path: List[str],
                             total_adjustments: Dict[str, float]) -> Tuple[ProbabilityScore, List[IterationResult]]:
        """Standard probability refinement process"""
        iterations = []
        iteration_count = 0
        
        while (current_score.band != expected_band and 
               iteration_count < self.max_iterations):
            
            # Calculate adjustment magnitude based on scenario type
            scenario_type = str(context.get('context_type', '')).lower()
            base_adjustment = self._calculate_base_adjustment(
                current_score, expected_band, scenario_type
            )
            
            # Apply scenario-specific adjustments
            adjustments = {
                'prism_weights': base_adjustment * self._get_scenario_weight(scenario_type),
                'cultural_context': base_adjustment * 0.8 if scenario_type == 'cultural' else base_adjustment * 0.5,
                'compliance': base_adjustment * 0.9 if scenario_type in ['privacy', 'compliance'] else base_adjustment * 0.4
            }
            
            # Apply adjustments
            new_score = current_score
            for param, adjustment in adjustments.items():
                new_score = self._apply_adjustment(new_score, adjustment)
                total_adjustments[param] += adjustment
            
            # Create iteration result
            iteration = self._create_iteration_result(
                iteration_count + 1,
                current_score,
                new_score,
                expected_band,
                adjustments,
                f"Iteration {iteration_count + 1}: Adjusting for {scenario_type} scenario"
            )
            iterations.append(iteration)
            
            # Update current score
            current_score = new_score
            iteration_count += 1
            
            # Early exit if we're getting close
            if abs(current_score.adjusted_score - self._get_band_target(expected_band)) < 0.1:
                break
        
        return current_score, iterations
    
    def _calculate_base_adjustment(self,
                                 current_score: ProbabilityScore,
                                 expected_band: ProbabilityBand,
                                 scenario_type: str) -> float:
        """Calculate base adjustment magnitude"""
        # Get target score for expected band
        target = self._get_band_target(expected_band)
        current = current_score.adjusted_score
        
        # Calculate raw difference
        difference = target - current
        
        # Apply scenario-specific scaling
        scaling_factors = {
            'cultural': 0.8,    # Reduced from 1.0 for more gradual changes
            'environmental': 1.2,  # Increased for faster convergence
            'privacy': 0.7,     # Reduced for more conservative changes
            'compliance': 0.7,  # Reduced for more conservative changes
            'default': 1.0
        }
        
        scaling = scaling_factors.get(scenario_type, scaling_factors['default'])
        
        return difference * scaling * 0.5  # Reduced from 0.7 for more stable adjustments
    
    def _get_band_target(self, band: ProbabilityBand) -> float:
        """Get target score for a probability band"""
        targets = {
            ProbabilityBand.HIGH: 0.75,
            ProbabilityBand.MEDIUM: 0.55,
            ProbabilityBand.LOW: 0.35
        }
        return targets[band]
    
    def _get_scenario_weight(self, scenario_type: str) -> float:
        """Get scenario-specific weight for adjustments"""
        weights = {
            'cultural': 0.8,    # Reduced from 1.0
            'environmental': 1.2,  # Increased from 1.0
            'privacy': 0.7,     # Reduced from 0.8
            'compliance': 0.7,  # Reduced from 0.8
            'default': 1.0
        }
        return weights.get(scenario_type, weights['default'])
    
    def _scenario_refinement(self,
                            score: ProbabilityScore,
                            expected_band: ProbabilityBand,
                            context: Dict,
                            decision_path: List[str],
                            total_adjustments: Dict[str, float]) -> Tuple[ProbabilityScore, List[IterationResult]]:
        """Second layer: Apply scenario-specific refinements"""
        iterations = []
        current_score = score
        
        # Get scenario type
        scenario_type = str(context.get('context_type', '')).lower()
        
        # Apply scenario-specific adjustments
        if scenario_type == 'environmental':
            adjustment = 0.1 if context.get('environmental_priority') == 'high' else 0.05
        elif scenario_type == 'privacy':
            adjustment = -0.1 if context.get('privacy_level') == 'high' else -0.05
        elif scenario_type == 'innovation':
            adjustment = 0.15 if context.get('innovation_tolerance') == 'progressive' else 0.1
        else:
            adjustment = 0
        
        if adjustment != 0:
            new_score = self._apply_adjustment(current_score, adjustment)
            iterations.append(self._create_iteration_result(
                len(iterations) + 1,
                current_score,
                new_score,
                expected_band,
                {'scenario_adjustment': adjustment},
                f"Applied scenario-specific adjustment for {scenario_type}"
            ))
            current_score = new_score
        
        return current_score, iterations
    
    def _band_refinement(self,
                        score: ProbabilityScore,
                        expected_band: ProbabilityBand,
                        context: Dict,
                        decision_path: List[str],
                        total_adjustments: Dict[str, float]) -> Tuple[ProbabilityScore, List[IterationResult]]:
        """Third layer: Fine-tune based on probability band"""
        iterations = []
        current_score = score
        
        # Calculate band difference
        band_diff = self._band_value(expected_band) - self._band_value(current_score.band)
        
        if abs(band_diff) > 0:
            # Apply fine-tuning adjustment
            adjustment = band_diff * 0.1  # Small adjustment based on band difference
            new_score = self._apply_adjustment(current_score, adjustment)
            
            iterations.append(self._create_iteration_result(
                len(iterations) + 1,
                current_score,
                new_score,
                expected_band,
                {'band_adjustment': adjustment},
                f"Applied band-specific fine-tuning"
            ))
            current_score = new_score
        
        return current_score, iterations
    
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
        if score >= 0.65:
            return ProbabilityBand.HIGH
        elif score >= 0.45:
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
        
        # Adjust confidence based on convergence and iterations
        confidence_adjustment = 0.8 if converged else 0.6
        iteration_penalty = len(iterations) * 0.05  # Reduce confidence with more iterations
        
        final_confidence = min(
            final_score.confidence_level * confidence_adjustment - iteration_penalty,
            0.95  # Cap maximum confidence
        )
        
        # Update final score with adjusted confidence
        final_score.confidence_level = final_confidence
        
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
    
    def _apply_adjustment(self, score: ProbabilityScore, adjustment: float) -> ProbabilityScore:
        """Apply a single adjustment to a probability score"""
        try:
            # Calculate new adjusted score
            new_adjusted_score = score.adjusted_score + adjustment
            
            # Ensure score stays within bounds
            new_adjusted_score = max(min(new_adjusted_score, 1.0), 0.0)
            
            # Create new probability score with updated values
            return ProbabilityScore(
                raw_score=score.raw_score,
                adjusted_score=new_adjusted_score,
                band=self._determine_band(new_adjusted_score),
                influencing_factors=score.influencing_factors.copy(),
                cultural_adjustments=score.cultural_adjustments.copy(),
                compliance_impacts=score.compliance_impacts.copy(),
                confidence_level=score.confidence_level
            )
        except Exception as e:
            self.logger.error(f"Error applying adjustment: {str(e)}")
            # Return original score if adjustment fails
            return score
    
    def _create_iteration_result(self,
                               iteration_number: int,
                               current_score: ProbabilityScore,
                               new_score: ProbabilityScore,
                               expected_band: ProbabilityBand,
                               adjustments: Dict[str, float],
                               reasoning: str) -> IterationResult:
        """Create result for a single iteration"""
        return IterationResult(
            iteration_number=iteration_number,
            predicted_band=current_score.band,
            expected_band=expected_band,
            adjustments_made=adjustments,
            adjustment_explanations=[
                AdjustmentExplanation(
                    parameter=param,
                    original_value=current_score.adjusted_score,
                    new_value=new_score.adjusted_score,
                    reason=reasoning,
                    impact=f"Score change: {value:.3f}"
                )
                for param, value in adjustments.items()
            ],
            confidence_delta=new_score.confidence_level - current_score.confidence_level,
            success=new_score.band == expected_band,
            reasoning=reasoning
        )