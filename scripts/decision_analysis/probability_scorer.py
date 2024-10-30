from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class ProbabilityBand(Enum):
    """Probability bands for decision outcomes"""
    HIGH = "high"        # 0.7 - 1.0
    MEDIUM = "medium"    # 0.4 - 0.69
    LOW = "low"         # 0.0 - 0.39

@dataclass
class ProbabilityScore:
    """Detailed probability score for a decision outcome"""
    raw_score: float
    adjusted_score: float
    band: ProbabilityBand
    influencing_factors: Dict[str, float]
    cultural_adjustments: Dict[str, float]
    compliance_impacts: Dict[str, float]
    confidence_level: float
    initial_recommendation: Optional[ProbabilityBand] = None

@dataclass
class ContextualProbability:
    """Contextual probability calculation result"""
    base_score: float
    context_multiplier: float
    secondary_effects_weight: float
    confidence_adjustment: float
    final_score: float
    reasoning: List[str]

class BayesianBandPredictor:
    """Bayesian predictor for probability bands"""
    
    def __init__(self):
        # Initialize prior probabilities for each band
        self.priors = {
            ProbabilityBand.LOW: 0.3,
            ProbabilityBand.MEDIUM: 0.4,
            ProbabilityBand.HIGH: 0.3
        }
        
        # Initialize likelihood parameters for each band
        self.likelihoods = {
            ProbabilityBand.LOW: {'mean': 0.3, 'std': 0.1},
            ProbabilityBand.MEDIUM: {'mean': 0.6, 'std': 0.1},
            ProbabilityBand.HIGH: {'mean': 0.8, 'std': 0.1}
        }
        
        # Store historical observations
        self.observations: List[Dict] = []
    
    def update_priors(self, evidence: Dict[str, float]) -> None:
        """Update prior probabilities using new evidence"""
        # Calculate likelihoods for each band
        likelihoods = self._calculate_likelihoods(evidence)
        
        # Calculate posterior probabilities
        total_probability = sum(
            self.priors[band] * likelihood 
            for band, likelihood in likelihoods.items()
        )
        
        # Update priors with posterior probabilities
        for band in ProbabilityBand:
            self.priors[band] = (
                self.priors[band] * likelihoods[band] / total_probability
            )
    
    def predict_band(self, evidence: Dict[str, float]) -> Tuple[ProbabilityBand, float]:
        """Predict probability band using current priors"""
        # Calculate likelihoods
        likelihoods = self._calculate_likelihoods(evidence)
        
        # Calculate posterior probabilities
        posteriors = {}
        total_probability = sum(
            self.priors[band] * likelihood 
            for band, likelihood in likelihoods.items()
        )
        
        for band in ProbabilityBand:
            posteriors[band] = (
                self.priors[band] * likelihoods[band] / total_probability
            )
        
        # Select band with highest posterior probability
        predicted_band = max(posteriors.items(), key=lambda x: x[1])[0]
        confidence = posteriors[predicted_band]
        
        return predicted_band, confidence
    
    def _calculate_likelihoods(self, evidence: Dict[str, float]) -> Dict[ProbabilityBand, float]:
        """Calculate likelihood of evidence for each band"""
        likelihoods = {}
        
        for band in ProbabilityBand:
            # Calculate likelihood using Gaussian distribution
            mean = self.likelihoods[band]['mean']
            std = self.likelihoods[band]['std']
            
            # Combine evidence using weighted sum
            evidence_score = sum(
                value * self._get_evidence_weight(key)
                for key, value in evidence.items()
            )
            
            likelihood = np.exp(-0.5 * ((evidence_score - mean) / std) ** 2)
            likelihoods[band] = likelihood
        
        return likelihoods
    
    def _get_evidence_weight(self, evidence_type: str) -> float:
        """Get weight for different types of evidence"""
        weights = {
            'confidence': 0.4,
            'impact': 0.3,
            'risk': 0.3
        }
        return weights.get(evidence_type, 0.1)
    
    def update_from_observation(self, 
                              evidence: Dict[str, float],
                              actual_band: ProbabilityBand) -> None:
        """Update model from observed outcome"""
        self.observations.append({
            'evidence': evidence,
            'actual_band': actual_band,
            'timestamp': datetime.now()
        })
        
        # Update likelihood parameters
        self._update_likelihood_parameters()
        
        # Update priors
        self.update_priors(evidence)
    
    def _update_likelihood_parameters(self) -> None:
        """Update likelihood parameters based on observations"""
        if len(self.observations) < 10:  # Need minimum observations
            return
            
        for band in ProbabilityBand:
            # Get observations for this band
            band_observations = [
                obs['evidence'] for obs in self.observations
                if obs['actual_band'] == band
            ]
            
            if band_observations:
                # Calculate new parameters
                evidence_scores = [
                    sum(evidence.values()) for evidence in band_observations
                ]
                self.likelihoods[band]['mean'] = np.mean(evidence_scores)
                self.likelihoods[band]['std'] = np.std(evidence_scores)

class FuzzySet:
    """Represents a fuzzy set with trapezoidal membership function"""
    def __init__(self, a: float, b: float, c: float, d: float):
        self.a = a  # Left foot
        self.b = b  # Left shoulder
        self.c = c  # Right shoulder
        self.d = d  # Right foot
    
    def membership(self, x: float) -> float:
        """Calculate membership degree for value x"""
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.b <= x <= self.c:
            return 1.0
        elif x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.d - x) / (self.d - self.c)

class FuzzyProbabilityEvaluator:
    """Evaluates probability using fuzzy logic"""
    
    def __init__(self):
        # Define fuzzy sets for confidence scores
        self.confidence_sets = {
            'low': FuzzySet(0.0, 0.0, 0.3, 0.5),
            'medium': FuzzySet(0.3, 0.5, 0.7, 0.8),
            'high': FuzzySet(0.7, 0.8, 1.0, 1.0)
        }
        
        # Define fuzzy sets for impact scores
        self.impact_sets = {
            'low': FuzzySet(-1.0, -1.0, -0.3, 0.0),
            'neutral': FuzzySet(-0.3, -0.1, 0.1, 0.3),
            'high': FuzzySet(0.0, 0.3, 1.0, 1.0)
        }
    
    def evaluate(self, confidence: float, impact: float) -> Dict[ProbabilityBand, float]:
        """Evaluate fuzzy rules to determine probability band memberships"""
        # Calculate memberships
        conf_memberships = {
            level: fuzzy_set.membership(confidence)
            for level, fuzzy_set in self.confidence_sets.items()
        }
        
        impact_memberships = {
            level: fuzzy_set.membership(impact)
            for level, fuzzy_set in self.impact_sets.items()
        }
        
        # Apply fuzzy rules
        band_memberships = {
            ProbabilityBand.LOW: 0.0,
            ProbabilityBand.MEDIUM: 0.0,
            ProbabilityBand.HIGH: 0.0
        }
        
        # Rule 1: If confidence is low OR impact is low -> LOW band
        band_memberships[ProbabilityBand.LOW] = max(
            conf_memberships['low'],
            impact_memberships['low']
        )
        
        # Rule 2: If confidence is medium AND impact is neutral -> MEDIUM band
        band_memberships[ProbabilityBand.MEDIUM] = min(
            conf_memberships['medium'],
            impact_memberships['neutral']
        )
        
        # Rule 3: If confidence is high AND impact is high -> HIGH band
        band_memberships[ProbabilityBand.HIGH] = min(
            conf_memberships['high'],
            impact_memberships['high']
        )
        
        return band_memberships

class ProbabilityScorer:
    """Calculates probability scores with integrated ML, Bayesian, and fuzzy logic"""
    
    def __init__(self):
        # Initialize context-specific thresholds
        self.context_thresholds = {
            'medical': {'high': 0.75, 'medium': 0.60},
            'environmental': {'high': 0.65, 'medium': 0.45},
            'cultural': {'high': 0.70, 'medium': 0.50},
            'privacy': {'high': 0.80, 'medium': 0.65},
            'default': {'high': 0.70, 'medium': 0.55}
        }
        
        # Initialize context weights
        self.context_weights = {
            'medical': 1.2,      # Higher weight for medical decisions
            'environmental': 0.9, # More lenient for environmental
            'cultural': 1.1,     # Slightly higher for cultural
            'privacy': 1.3,      # Strict for privacy
            'default': 1.0
        }
        
        # Initialize secondary effects weights
        self.secondary_weights = {
            'human': 0.3,
            'eco': 0.2,
            'innovation': 0.2,
            'sentient': 0.3
        }
        
        # Initialize feedback history
        self.feedback_history: List[Dict] = []
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.classifier = DecisionTreeClassifier(max_depth=5)
        self.model_path = Path("models/probability_classifier.joblib")
        
        # Load or train model
        if self.model_path.exists():
            self._load_model()
        else:
            self._train_model()
        
        # Initialize Bayesian predictor
        self.bayesian_predictor = BayesianBandPredictor()
        
        # Initialize fuzzy logic evaluator
        self.fuzzy_evaluator = FuzzyProbabilityEvaluator()
    
    def _init_weights_and_thresholds(self):
        """Initialize standard weights and thresholds"""
        self.secondary_weights = {
            'human': 0.3,
            'eco': 0.2,
            'innovation': 0.2,
            'sentient': 0.3
        }
        
        self.context_weights = {
            'medical': 1.2,
            'environmental': 0.9,
            'cultural': 1.1,
            'privacy': 1.3,
            'default': 1.0
        }
        
        self.context_thresholds = {
            'environmental': {'high': 0.65, 'medium': 0.40},
            'medical': {'high': 0.80, 'medium': 0.60},
            'cultural': {'high': 0.70, 'medium': 0.45},
            'default': {'high': 0.70, 'medium': 0.45}
        }
    
    def _prepare_training_data(self) -> tuple:
        """Prepare historical data for training"""
        # Example training data structure with correct number of features
        historical_data = [
            {
                'confidence_score': 0.8,
                'context_type': 'environmental',
                'secondary_effects_score': 0.7,
                'confidence_modifier': 1.2,
                'risk_score': 0.3
            },
            {
                'confidence_score': 0.6,
                'context_type': 'medical',
                'secondary_effects_score': 0.5,
                'confidence_modifier': 0.8,
                'risk_score': 0.7
            },
            {
                'confidence_score': 0.7,
                'context_type': 'cultural',
                'secondary_effects_score': 0.6,
                'confidence_modifier': 1.0,
                'risk_score': 0.5
            }
        ]
        
        X = []  # Features
        y = []  # Labels
        
        for entry in historical_data:
            features = [
                entry['confidence_score'],
                self._encode_context_type(entry['context_type']),
                entry['secondary_effects_score'],
                entry['confidence_modifier'],
                entry['risk_score']
            ]
            X.append(features)
            y.append(1)  # Example label, should be actual band encoding
        
        return np.array(X), np.array(y)
    
    def _train_model(self):
        """Train the probability band classifier"""
        logger.info("Training probability band classifier...")
        
        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train classifier
            self.classifier.fit(X_scaled, y)
            
            # Save model and scaler
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump((self.classifier, self.scaler), self.model_path)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def _load_model(self):
        """Load trained model and scaler"""
        try:
            self.classifier, self.scaler = joblib.load(self.model_path)
            logger.info("Loaded probability band classifier")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._train_model()
    
    def _predict_probability_band(self,
                                confidence_score: float,
                                secondary_effects: Dict[str, float],
                                context_type: str) -> ProbabilityBand:
        """Predict probability band using ML model"""
        try:
            # Calculate secondary effects score
            secondary_effects_score = sum(secondary_effects.values()) / len(secondary_effects)
            
            # Get confidence modifier
            confidence_modifier = self._get_confidence_modifier(context_type)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score({'context_type': context_type})
            
            # Prepare features
            features = [
                confidence_score,
                self._encode_context_type(context_type),
                secondary_effects_score,
                confidence_modifier,
                risk_score
            ]
            
            # Scale features
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            # Predict band
            prediction = self.classifier.predict(X_scaled)[0]
            
            # Decode prediction
            return self._decode_probability_band(prediction)
            
        except Exception as e:
            logger.error(f"Error predicting probability band: {str(e)}")
            # Fallback to traditional method
            return self._determine_band_from_score(confidence_score, context_type)
    
    def _encode_context_type(self, context_type: str) -> float:
        """Encode context type as numeric value"""
        context_values = {
            'medical': 1.0,
            'environmental': 2.0,
            'cultural': 3.0,
            'privacy': 4.0
        }
        return context_values.get(context_type, 0.0)
    
    def _encode_probability_band(self, band: ProbabilityBand) -> int:
        """Encode probability band as integer"""
        return {
            ProbabilityBand.LOW: 0,
            ProbabilityBand.MEDIUM: 1,
            ProbabilityBand.HIGH: 2
        }[band]
    
    def _decode_probability_band(self, encoded_value: int) -> ProbabilityBand:
        """Decode integer to probability band"""
        return {
            0: ProbabilityBand.LOW,
            1: ProbabilityBand.MEDIUM,
            2: ProbabilityBand.HIGH
        }[encoded_value]
    
    def update_model(self, new_data: Dict):
        """Update model with new training data"""
        try:
            # Prepare new training data
            X_new = np.array([[
                new_data['confidence_score'],
                *[new_data['secondary_effects'].get(effect, 0.0) 
                  for effect in ['human', 'eco', 'innovation', 'sentient']],
                self._encode_context_type(new_data['context_type'])
            ]])
            y_new = np.array([self._encode_probability_band(new_data['probability_band'])])
            
            # Update scaler and transform new data
            X_new_scaled = self.scaler.transform(X_new)
            
            # Partial fit of classifier
            self.classifier.fit(X_new_scaled, y_new)
            
            # Save updated model
            joblib.dump((self.classifier, self.scaler), self.model_path)
            
            logger.info("Model updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
    
    def calculate_probability(self,
                            prism_scores: Dict[str, float],
                            context: Dict,
                            compliance_data: Dict,
                            decision_impact: str) -> ProbabilityScore:
        """Calculate probability score with integrated ML, Bayesian, and fuzzy logic"""
        logger.debug("Starting probability calculation with integrated components")
        
        try:
            # Get context type and base confidence
            context_type = str(context.get('context_type', '')).lower()
            base_confidence = self._calculate_direct_confidence(context)
            logger.debug(f"Initial confidence: {base_confidence} for {context_type}")
            
            # Step 1: ML Classifier Prediction
            classifier_prediction = self._predict_probability_band(
                base_confidence,
                prism_scores,
                context_type
            )
            logger.debug(f"Classifier prediction: {classifier_prediction}")
            
            # Step 2: Bayesian Update
            evidence = {
                'confidence': base_confidence,
                'impact': sum(prism_scores.values()),
                'risk': self._calculate_risk_score(context)
            }
            bayesian_band, bayesian_confidence = self.bayesian_predictor.predict_band(evidence)
            logger.debug(f"Bayesian prediction: {bayesian_band} with confidence {bayesian_confidence}")
            
            # Step 3: Fuzzy Logic Evaluation
            impact_score = self._calculate_impact_score(context_type)
            fuzzy_memberships = self.fuzzy_evaluator.evaluate(base_confidence, impact_score)
            logger.debug(f"Fuzzy memberships: {fuzzy_memberships}")
            
            # Step 4: Combine Predictions
            final_band = self._combine_predictions(
                classifier_prediction,
                bayesian_band,
                fuzzy_memberships,
                context_type
            )
            logger.debug(f"Combined final band: {final_band}")
            
            # Calculate final confidence and score
            final_confidence = self._calculate_final_confidence(
                base_confidence,
                bayesian_confidence,
                fuzzy_memberships,
                context_type
            )
            
            final_score = self._calculate_final_score(
                prism_scores,
                final_band,
                final_confidence,
                context_type
            )
            
            # Update models with new observation
            self._update_models(
                context_type=context_type,
                initial_score=base_confidence,
                final_score=final_score,
                initial_band=classifier_prediction,
                final_band=final_band,
                evidence=evidence
            )
            
            return ProbabilityScore(
                raw_score=base_confidence,
                adjusted_score=final_score,
                band=final_band,
                influencing_factors=prism_scores,
                cultural_adjustments=self._get_cultural_adjustments(context),
                compliance_impacts=compliance_data,
                confidence_level=final_confidence,
                initial_recommendation=classifier_prediction
            )
            
        except Exception as e:
            logger.error(f"Error in probability calculation: {str(e)}")
            raise

    def _combine_predictions(self,
                            classifier_band: ProbabilityBand,
                            bayesian_band: ProbabilityBand,
                            fuzzy_memberships: Dict[ProbabilityBand, float],
                            context_type: str) -> ProbabilityBand:
        """Combine predictions from different models"""
        # Get highest fuzzy membership band
        fuzzy_band = max(fuzzy_memberships.items(), key=lambda x: x[1])[0]
        
        # Weight the predictions based on context type
        weights = {
            'environmental': {'classifier': 0.3, 'bayesian': 0.3, 'fuzzy': 0.4},
            'medical': {'classifier': 0.4, 'bayesian': 0.4, 'fuzzy': 0.2},
            'cultural': {'classifier': 0.3, 'bayesian': 0.4, 'fuzzy': 0.3},
            'default': {'classifier': 0.33, 'bayesian': 0.33, 'fuzzy': 0.34}
        }
        
        model_weights = weights.get(context_type, weights['default'])
        
        # Convert bands to numeric values
        band_values = {
            ProbabilityBand.LOW: 0,
            ProbabilityBand.MEDIUM: 1,
            ProbabilityBand.HIGH: 2
        }
        
        # Calculate weighted average
        weighted_value = (
            band_values[classifier_band] * model_weights['classifier'] +
            band_values[bayesian_band] * model_weights['bayesian'] +
            band_values[fuzzy_band] * model_weights['fuzzy']
        )
        
        # Convert back to band
        if weighted_value >= 1.5:
            return ProbabilityBand.HIGH
        elif weighted_value >= 0.5:
            return ProbabilityBand.MEDIUM
        return ProbabilityBand.LOW

    def _calculate_final_confidence(self,
                                  base_confidence: float,
                                  bayesian_confidence: float,
                                  fuzzy_memberships: Dict[ProbabilityBand, float],
                                  context_type: str) -> float:
        """Calculate final confidence score"""
        # Get maximum fuzzy membership value
        max_fuzzy_confidence = max(fuzzy_memberships.values())
        
        # Weight confidences based on context
        weights = {
            'environmental': {'base': 0.3, 'bayesian': 0.3, 'fuzzy': 0.4},
            'medical': {'base': 0.4, 'bayesian': 0.4, 'fuzzy': 0.2},
            'cultural': {'base': 0.3, 'bayesian': 0.4, 'fuzzy': 0.3},
            'default': {'base': 0.33, 'bayesian': 0.33, 'fuzzy': 0.34}
        }
        
        confidence_weights = weights.get(context_type, weights['default'])
        
        # Calculate weighted confidence
        final_confidence = (
            base_confidence * confidence_weights['base'] +
            bayesian_confidence * confidence_weights['bayesian'] +
            max_fuzzy_confidence * confidence_weights['fuzzy']
        )
        
        return max(min(final_confidence, 1.0), 0.0)

    def _update_models(self,
                      context_type: str,
                      initial_score: float,
                      final_score: float,
                      initial_band: ProbabilityBand,
                      final_band: ProbabilityBand,
                      evidence: Dict[str, float]) -> None:
        """Update all models with new observation"""
        # Update classifier
        self.update_classifier({
            'score': final_score,
            'context_type': context_type,
            'actual_band': final_band
        })
        
        # Update Bayesian model
        self.bayesian_predictor.update_from_observation(evidence, final_band)
        
        # Log pattern for analysis
        self._log_decision_pattern(
            context_type=context_type,
            initial_score=initial_score,
            adjusted_score=final_score,
            initial_band=initial_band,
            final_band=final_band,
            confidence=evidence['confidence']
        )

    def _calculate_weighted_base_score(self, 
                                     prism_scores: Dict[str, float],
                                     context_type: str) -> float:
        """Calculate base score with context-specific weights"""
        # Define context-specific prism weights
        context_weights = {
            'environmental': {
                'eco': 0.4,
                'innovation': 0.3,
                'human': 0.2,
                'sentient': 0.1
            },
            'medical': {
                'human': 0.5,
                'sentient': 0.3,
                'eco': 0.1,
                'innovation': 0.1
            },
            'cultural': {
                'human': 0.4,
                'sentient': 0.3,
                'eco': 0.2,
                'innovation': 0.1
            },
            'default': {
                'human': 0.3,
                'sentient': 0.3,
                'eco': 0.2,
                'innovation': 0.2
            }
        }
        
        weights = context_weights.get(context_type, context_weights['default'])
        
        weighted_sum = sum(
            score * weights.get(prism, 0.25)
            for prism, score in prism_scores.items()
        )
        
        return max(min(weighted_sum, 1.0), 0.0)

    def _apply_contextual_adjustments(self,
                                    score: float,
                                    context_type: str,
                                    prism_scores: Dict[str, float],
                                    context: Dict) -> float:
        """Apply fine-grained contextual adjustments"""
        adjusted_score = score
        
        # Get context-specific thresholds
        thresholds = self.context_thresholds.get(
            context_type,
            self.context_thresholds['default']
        )
        
        # Calculate cumulative impact
        positive_impacts = sum(max(score, 0) for score in prism_scores.values())
        negative_impacts = sum(min(score, 0) for score in prism_scores.values())
        
        # Apply scaled adjustments based on impact balance
        if positive_impacts > abs(negative_impacts):
            adjustment = min(positive_impacts * 0.1, 0.15)
            adjusted_score *= (1 + adjustment)
        else:
            adjustment = min(abs(negative_impacts) * 0.1, 0.15)
            adjusted_score *= (1 - adjustment)
        
        # Apply threshold proximity adjustments
        if abs(adjusted_score - thresholds['medium']) < 0.05:
            # Score is very close to MEDIUM threshold
            if context.get('stakeholder_consensus') == 'high':
                adjusted_score += 0.03
        
        return max(min(adjusted_score, 1.0), 0.0)

    def _apply_feedback_refinements(self,
                                  score: float,
                                  initial_band: ProbabilityBand,
                                  context_type: str,
                                  confidence: float) -> Tuple[float, ProbabilityBand]:
        """Apply feedback-based refinements with pattern learning"""
        # Get historical patterns
        patterns = self._get_historical_patterns(context_type)
        
        if patterns:
            # Calculate average historical variance
            avg_variance = sum(p['variance'] for p in patterns) / len(patterns)
            
            # Apply pattern-based adjustment
            if abs(avg_variance) > 0.1:  # Significant historical variance
                adjustment = -avg_variance * 0.5  # Compensate for historical bias
                score = max(min(score + adjustment, 1.0), 0.0)
                
                logger.debug(
                    f"Applied pattern-based adjustment: {adjustment} "
                    f"based on historical variance: {avg_variance}"
                )
        
        # Determine final band with refined thresholds
        final_band = self._determine_refined_band(
            score,
            initial_band,
            context_type,
            confidence
        )
        
        return score, final_band

    def _determine_refined_band(self,
                                score: float,
                                initial_band: ProbabilityBand,
                                context_type: str,
                                confidence: float) -> ProbabilityBand:
        """Determine probability band with refined thresholds"""
        # Get base thresholds
        thresholds = self.context_thresholds.get(
            context_type,
            self.context_thresholds['default']
        )
        
        # Adjust thresholds based on confidence
        if confidence > 0.8:
            # High confidence allows more lenient thresholds
            thresholds = {
                k: v * 0.95 for k, v in thresholds.items()
            }
        elif confidence < 0.4:
            # Low confidence requires stricter thresholds
            thresholds = {
                k: v * 1.05 for k, v in thresholds.items()
            }
        
        # Check for threshold proximity
        if abs(score - thresholds['medium']) < 0.05:
            # Very close to MEDIUM threshold
            return ProbabilityBand.MEDIUM
        elif abs(score - thresholds['high']) < 0.05:
            # Very close to HIGH threshold
            return ProbabilityBand.HIGH
        
        # Use standard thresholds
        if score >= thresholds['high']:
            return ProbabilityBand.HIGH
        elif score >= thresholds['medium']:
            return ProbabilityBand.MEDIUM
        return ProbabilityBand.LOW

    def _log_decision_pattern(self,
                             context_type: str,
                             initial_score: float,
                             adjusted_score: float,
                             initial_band: ProbabilityBand,
                             final_band: ProbabilityBand,
                             confidence: float) -> None:
        """Log decision pattern for future refinement"""
        pattern = {
            'timestamp': datetime.now().isoformat(),
            'context_type': context_type,
            'initial_score': initial_score,
            'adjusted_score': adjusted_score,
            'variance': adjusted_score - initial_score,
            'initial_band': initial_band.value,
            'final_band': final_band.value,
            'confidence': confidence
        }
        
        # Store pattern in feedback history
        if not hasattr(self, 'pattern_history'):
            self.pattern_history = []
        self.pattern_history.append(pattern)
        
        # Limit history size
        if len(self.pattern_history) > 1000:
            self.pattern_history = self.pattern_history[-1000:]
        
        logger.debug(f"Logged decision pattern: {pattern}")

    def _get_historical_patterns(self, context_type: str) -> List[Dict]:
        """Get historical patterns for context type"""
        if not hasattr(self, 'pattern_history'):
            return []
        
        return [
            pattern for pattern in self.pattern_history
            if pattern['context_type'] == context_type
        ]

    def add_feedback(self,
                    context_type: str,
                    prism_scores: Dict[str, float],
                    predicted_score: float,
                    actual_score: float) -> None:
        """Add feedback to history for future adjustments"""
        self.feedback_history.append({
            'timestamp': datetime.now(),
            'context_type': context_type,
            'prism_scores': prism_scores.copy(),
            'predicted_score': predicted_score,
            'actual_score': actual_score
        })
        
        # Limit history size
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-1000:]
            
    def _get_cultural_adjustments(self, context: Dict) -> Dict[str, float]:
        """Calculate cultural adjustments based on context"""
        adjustments = {}
        
        if 'cultural_context' in context:
            cultural_context = context['cultural_context']
            
            if 'privacy_emphasis' in cultural_context:
                adjustments['privacy'] = {
                    'high': 0.2,
                    'very_high': 0.3
                }.get(cultural_context['privacy_emphasis'], 0.0)
                
            if 'innovation_tolerance' in cultural_context:
                adjustments['innovation'] = {
                    'progressive': 0.2,
                    'conservative': -0.1
                }.get(cultural_context['innovation_tolerance'], 0.0)
                
        return adjustments

    def _calculate_direct_confidence(self, context: Dict) -> float:
        """Calculate confidence with refined adjustments"""
        logger.debug("Calculating direct confidence")
        
        # Get scenario type
        scenario_type = str(context.get('context_type', '')).lower()
        
        # Refined base confidence levels
        base_confidence = {
            'medical': 0.45,      # Conservative base for medical
            'environmental': 0.70, # Higher base for environmental
            'cultural': 0.40,     # Lower base for cultural
            'default': 0.50
        }.get(scenario_type, 0.50)
        
        logger.debug(f"Base confidence for {scenario_type}: {base_confidence}")
        
        # Context-based adjustments
        adjustments = 0.0
        
        # Stakeholder presence increases confidence
        if context.get('stakeholder'):
            adjustments += 0.15  # Increased from 0.10
            logger.debug("Added 0.15 for stakeholder presence")
        
        # Metrics presence increases confidence
        if context.get('metrics'):
            adjustments += 0.10
            logger.debug("Added 0.10 for metrics presence")
        
        # Mitigation strategies increase confidence
        if context.get('mitigation_strategies'):
            adjustments += 0.10
            logger.debug("Added 0.10 for mitigation strategies")
        
        # Uncertainty level affects confidence
        uncertainty_level = context.get('uncertainty_level', 'medium').lower()
        uncertainty_adjustments = {
            'low': 0.15,    # Increased positive adjustment
            'medium': 0.0,
            'high': -0.15   # Increased negative adjustment
        }
        
        uncertainty_adj = uncertainty_adjustments.get(uncertainty_level, 0.0)
        adjustments += uncertainty_adj
        logger.debug(f"Added {uncertainty_adj} for uncertainty level: {uncertainty_level}")
        
        # Calculate final confidence
        final_confidence = base_confidence + adjustments
        
        # Ensure within valid range with scenario-specific caps
        confidence_caps = {
            'medical': 0.85,      # Lower cap for medical
            'environmental': 0.95, # Higher cap for environmental
            'cultural': 0.90,     # Standard cap for cultural
            'default': 0.90
        }
        
        max_confidence = confidence_caps.get(scenario_type, confidence_caps['default'])
        final_confidence = max(min(final_confidence, max_confidence), 0.30)
        
        logger.debug(f"Final confidence calculated: {final_confidence}")
        return final_confidence

    def _calculate_base_score(self, prism_scores: Dict[str, float]) -> float:
        """Calculate base score from prism scores"""
        logger.debug("Calculating base score")
        
        if not prism_scores:
            return 0.0
            
        # Calculate weighted average
        total_weight = sum(self.secondary_weights.values())
        weighted_sum = sum(
            score * self.secondary_weights.get(prism, 0.25)
            for prism, score in prism_scores.items()
        )
        
        base_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        logger.debug(f"Base score calculated: {base_score}")
        
        return base_score

    def _apply_scenario_adjustments(self,
                                  base_score: float,
                                  scenario_type: str,
                                  prism_scores: Dict[str, float]) -> float:
        """Apply scenario-specific score adjustments"""
        adjusted_score = base_score
        
        # Environmental scenario adjustments - significantly boosted
        if scenario_type == 'environmental':
            eco_impact = prism_scores.get('eco', 0)
            if eco_impact > 0.5:
                adjusted_score *= 1.8  # Significantly increased boost
            if prism_scores.get('innovation', 0) > 0.3:
                adjusted_score *= 1.4  # Additional boost for innovative eco solutions
                
        # Cultural scenario adjustments
        elif scenario_type == 'cultural':
            human_impact = prism_scores.get('human', 0)
            if human_impact > 0.6:
                adjusted_score *= 1.5  # Increased boost for positive human impact
        
        return max(min(adjusted_score, 1.0), 0.0)

    def _requires_secondary_review(self, prism_scores: Dict[str, float], context: Dict) -> bool:
        """
        Determine if secondary review is required using two-tier condition check
        """
        logger.debug("Evaluating secondary review requirement")
        
        # Calculate confidence
        confidence = self._calculate_direct_confidence(context)
        logger.debug(f"Confidence level: {confidence}")
        
        # Get scenario type
        scenario_type = str(context.get('context_type', '')).lower()
        logger.debug(f"Scenario type: {scenario_type}")
        
        # Calculate impact scores
        human_impact = prism_scores.get('human', 0)
        eco_impact = prism_scores.get('eco', 0)
        innovation_impact = prism_scores.get('innovation', 0)
        sentient_impact = prism_scores.get('sentient', 0)
        
        # Calculate total positive and negative impacts
        positive_impacts = sum(max(score, 0) for score in prism_scores.values())
        negative_impacts = abs(sum(min(score, 0) for score in prism_scores.values()))
        
        logger.debug(f"Positive impacts: {positive_impacts}")
        logger.debug(f"Negative impacts: {negative_impacts}")
        
        # Special handling for environmental scenarios with clear positive impact
        if scenario_type == 'environmental':
            if eco_impact > 0.5 and innovation_impact > 0.3:
                if confidence >= 0.6:
                    logger.debug("Skipping secondary review for positive environmental impact")
                    return False
        
        # Primary Trigger: Confidence Check
        confidence_trigger = False
        if scenario_type == 'medical':
            # Stricter confidence range for medical
            confidence_trigger = 0.6 <= confidence <= 0.85
        else:
            # Standard confidence range for other scenarios
            confidence_trigger = 0.7 <= confidence <= 0.9
        
        logger.debug(f"Confidence trigger: {confidence_trigger}")
        
        # Secondary Check: Impact Analysis
        if confidence_trigger:
            # Check for ethical ambiguity through impact analysis
            impact_triggers = [
                # Significant mixed impacts
                (positive_impacts > 1.5 and negative_impacts > 0.5),
                
                # High human/sentient impact
                (abs(human_impact) > 0.7 or abs(sentient_impact) > 0.7),
                
                # Conflicting impacts between domains
                (abs(eco_impact - human_impact) > 0.5),
                
                # High innovation impact with risks
                (innovation_impact > 0.7 and negative_impacts > 0.3)
            ]
            
            # Count triggered conditions
            trigger_count = sum(1 for trigger in impact_triggers if trigger)
            logger.debug(f"Impact trigger count: {trigger_count}")
            
            # Require at least two impact conditions for secondary review
            requires_review = trigger_count >= 2
            
            # Special case overrides
            if scenario_type == 'medical' and human_impact > 0.8:
                requires_review = True
            elif scenario_type == 'environmental' and eco_impact < -0.5:
                requires_review = True
                
            logger.debug(f"Secondary review decision: {requires_review}")
            return requires_review
        
        # If confidence trigger not met, no secondary review needed
        return False

    def _determine_band_from_score(self, score: float, context_type: str) -> ProbabilityBand:
        """Determine probability band using fuzzy logic"""
        logger.debug(f"Determining band for score {score} in {context_type} context")
        
        # Calculate impact score from context
        impact_score = self._calculate_impact_score(context_type)
        
        # Get fuzzy memberships
        memberships = self.fuzzy_evaluator.evaluate(score, impact_score)
        logger.debug(f"Fuzzy memberships: {memberships}")
        
        # Apply context-specific adjustments
        adjusted_memberships = self._adjust_memberships(memberships, context_type)
        logger.debug(f"Adjusted memberships: {adjusted_memberships}")
        
        # Select band with highest membership
        final_band = max(adjusted_memberships.items(), key=lambda x: x[1])[0]
        logger.debug(f"Selected band: {final_band}")
        
        return final_band
    
    def _adjust_memberships(self,
                          memberships: Dict[ProbabilityBand, float],
                          context_type: str) -> Dict[ProbabilityBand, float]:
        """Adjust fuzzy memberships based on context"""
        adjusted = memberships.copy()
        
        # Context-specific adjustments
        if context_type == 'environmental':
            # Boost MEDIUM and HIGH for environmental contexts
            adjusted[ProbabilityBand.MEDIUM] *= 1.2
            adjusted[ProbabilityBand.HIGH] *= 1.1
        elif context_type == 'medical':
            # More conservative for medical contexts
            adjusted[ProbabilityBand.LOW] *= 1.2
            adjusted[ProbabilityBand.HIGH] *= 0.9
        elif context_type == 'cultural':
            # Balanced approach for cultural contexts
            adjusted[ProbabilityBand.MEDIUM] *= 1.1
        
        # Normalize memberships
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}
        
        return adjusted
    
    def _calculate_impact_score(self, context_type: str) -> float:
        """Calculate impact score for fuzzy evaluation"""
        base_scores = {
            'environmental': 0.7,  # Higher base for environmental
            'medical': 0.5,       # Moderate base for medical
            'cultural': 0.4,      # Lower base for cultural
            'default': 0.5
        }
        
        return base_scores.get(context_type, base_scores['default'])

    def _get_confidence_modifier(self, context_type: str) -> float:
        """Get confidence modifier based on context type"""
        modifiers = {
            'environmental': 1.2,  # Boost environmental confidence
            'medical': 0.8,       # Reduce medical confidence
            'cultural': 0.9,      # Slightly reduce cultural confidence
            'default': 1.0
        }
        return modifiers.get(context_type, modifiers['default'])

    def _apply_confidence_adjustments(self,
                                    predicted_band: ProbabilityBand,
                                    score: float,
                                    context_type: str) -> ProbabilityBand:
        """Apply confidence-based adjustments to predicted band"""
        # Get confidence thresholds
        confidence_thresholds = {
            'environmental': 0.6,
            'medical': 0.7,
            'cultural': 0.65,
            'default': 0.65
        }
        
        threshold = confidence_thresholds.get(context_type, confidence_thresholds['default'])
        
        # Check for boundary conditions
        if predicted_band == ProbabilityBand.LOW and score > threshold:
            logger.debug("Upgrading LOW to MEDIUM due to high confidence")
            return ProbabilityBand.MEDIUM
        elif predicted_band == ProbabilityBand.HIGH and score < threshold:
            logger.debug("Downgrading HIGH to MEDIUM due to low confidence")
            return ProbabilityBand.MEDIUM
        
        return predicted_band

    def _fallback_band_determination(self, score: float, context_type: str) -> ProbabilityBand:
        """Fallback method for band determination if classifier fails"""
        logger.warning("Using fallback band determination")
        
        # Conservative thresholds
        thresholds = {
            'environmental': {'high': 0.7, 'medium': 0.5},
            'medical': {'high': 0.8, 'medium': 0.6},
            'cultural': {'high': 0.75, 'medium': 0.55},
            'default': {'high': 0.75, 'medium': 0.55}
        }
        
        scenario_thresholds = thresholds.get(context_type, thresholds['default'])
        
        if score >= scenario_thresholds['high']:
            return ProbabilityBand.HIGH
        elif score >= scenario_thresholds['medium']:
            return ProbabilityBand.MEDIUM
        return ProbabilityBand.LOW

    def update_classifier(self, new_data: Dict[str, Any]) -> None:
        """Update classifier with new training data"""
        try:
            # Prepare new training data
            features = [
                new_data['score'],
                self._encode_context_type(new_data['context_type']),
                new_data.get('secondary_effects_score', 0.5),
                new_data.get('confidence_modifier', 1.0)
            ]
            
            X_new = np.array([features])
            y_new = np.array([self._encode_probability_band(new_data['actual_band'])])
            
            # Update scaler and transform new data
            X_new_scaled = self.scaler.transform(X_new)
            
            # Partial fit of classifier
            self.classifier.partial_fit(X_new_scaled, y_new)
            
            # Save updated model
            joblib.dump((self.classifier, self.scaler), self.model_path)
            
            logger.info("Classifier updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating classifier: {str(e)}")

    def _test_band_determination(self) -> None:
        """Test band determination with sample scores"""
        test_cases = [
            ('environmental', 0.45, ProbabilityBand.MEDIUM),  # Should be MEDIUM for environmental
            ('medical', 0.65, ProbabilityBand.MEDIUM),       # Should be MEDIUM for medical
            ('cultural', 0.75, ProbabilityBand.HIGH),        # Should be HIGH for cultural
            ('privacy', 0.70, ProbabilityBand.MEDIUM),       # Should be MEDIUM for privacy
            ('default', 0.80, ProbabilityBand.HIGH)          # Should be HIGH for default
        ]
        
        for context_type, score, expected_band in test_cases:
            actual_band = self._determine_band_from_score(score, context_type)
            assert actual_band == expected_band, (
                f"Band determination failed for {context_type} context with score {score}. "
                f"Expected {expected_band}, got {actual_band}"
            )

    def _calculate_risk_score(self, context: Dict) -> float:
        """Calculate risk score based on context"""
        logger.debug("Calculating risk score")
        
        # Base risk score
        base_risk = 0.5
        
        # Get context type
        context_type = str(context.get('context_type', '')).lower()
        
        # Risk modifiers by context type
        risk_modifiers = {
            'medical': 0.3,      # Higher base risk for medical
            'environmental': 0.1, # Lower base risk for environmental
            'cultural': 0.2,     # Moderate base risk for cultural
            'default': 0.2
        }
        
        base_risk += risk_modifiers.get(context_type, risk_modifiers['default'])
        
        # Adjust for risk factors in context
        if context.get('risk_level') == 'high':
            base_risk += 0.2
        elif context.get('risk_level') == 'low':
            base_risk -= 0.1
            
        # Adjust for mitigation strategies
        if context.get('mitigation_strategies'):
            base_risk -= 0.15
            
        # Ensure risk score stays within bounds
        return max(min(base_risk, 1.0), 0.0)