from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

class ResponsibilityLevel(Enum):
    """Enumeration of responsibility levels"""
    SYSTEM = "system"
    USER = "user"
    TEAM = "team"
    SUPERVISOR = "supervisor"
    EXTERNAL = "external"

@dataclass
class DecisionRecord:
    """Detailed record of a decision and its accountability trail"""
    decision_id: int
    timestamp: datetime
    context: Dict[str, Any]
    rationale: str
    responsible_entity: str
    impact_level: str
    review_status: str
    explanation: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AccountabilityAssessment:
    """Assessment of decision accountability and transparency"""
    transparency_score: float
    traceability_score: float
    responsibility_clarity: float
    review_recommendations: List[str]
    compliance_notes: List[str]

class AccountabilityPrism:
    """
    Accountability Prism for AEPF Mk1
    Ensures transparency, traceability, and responsibility in AI decision-making
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.decision_log: Dict[int, DecisionRecord] = {}
        self.responsibility_assignments: Dict[int, str] = {}
        self.review_history: Dict[int, List[Dict[str, Any]]] = {}
    
    def explain_decision(self, decision_data: Dict[str, Any]) -> str:
        """
        Generate a clear, non-technical explanation of a decision
        
        Args:
            decision_data: Dictionary containing decision details
            
        Returns:
            Human-readable explanation of the decision process
        """
        try:
            # Extract key decision components
            context = decision_data.get('context', {})
            rationale = decision_data.get('rationale', '')
            impact = decision_data.get('impact', '')
            
            # Construct explanation template
            explanation = (
                f"Decision Context: {self._simplify_context(context)}\n"
                f"Reasoning: {self._simplify_text(rationale)}\n"
                f"Expected Impact: {self._simplify_text(impact)}"
            )
            
            self.logger.info(f"Generated explanation for decision")
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating decision explanation: {str(e)}")
            return "Unable to generate explanation due to an error"
    
    def log_decision(self, decision_id: int, decision_data: Dict[str, Any]) -> bool:
        """
        Log a decision with full context and traceability information
        
        Args:
            decision_id: Unique identifier for the decision
            decision_data: Complete decision data including context and rationale
            
        Returns:
            Boolean indicating success of logging operation
        """
        try:
            record = DecisionRecord(
                decision_id=decision_id,
                timestamp=datetime.now(),
                context=decision_data.get('context', {}),
                rationale=decision_data.get('rationale', ''),
                responsible_entity=decision_data.get('responsible_entity', 'system'),
                impact_level=decision_data.get('impact_level', 'unknown'),
                review_status='pending',
                explanation=self.explain_decision(decision_data)
            )
            
            self.decision_log[decision_id] = record
            self.logger.info(f"Decision {decision_id} logged successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging decision {decision_id}: {str(e)}")
            return False
    
    def assign_responsibility(self, decision_id: int, responsible_entity: str) -> bool:
        """
        Assign responsibility for a decision to a specific entity
        
        Args:
            decision_id: ID of the decision
            responsible_entity: Entity (user, team, system) responsible for the decision
            
        Returns:
            Boolean indicating success of responsibility assignment
        """
        try:
            if decision_id not in self.decision_log:
                raise ValueError(f"Decision {decision_id} not found in log")
                
            # Validate responsible entity
            try:
                ResponsibilityLevel(responsible_entity)
            except ValueError:
                raise ValueError(f"Invalid responsibility level: {responsible_entity}")
            
            self.responsibility_assignments[decision_id] = responsible_entity
            self.decision_log[decision_id].responsible_entity = responsible_entity
            
            self.logger.info(
                f"Responsibility for decision {decision_id} assigned to {responsible_entity}"
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Error assigning responsibility for decision {decision_id}: {str(e)}"
            )
            return False
    
    def get_responsibility(self, decision_id: int) -> Optional[str]:
        """
        Retrieve the responsible entity for a decision
        
        Args:
            decision_id: ID of the decision
            
        Returns:
            String identifying the responsible entity, or None if not found
        """
        return self.responsibility_assignments.get(decision_id)
    
    def review_decision(self, decision_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete information about a decision for review
        
        Args:
            decision_id: ID of the decision to review
            
        Returns:
            Dictionary containing all decision information, or None if not found
        """
        try:
            if decision_id not in self.decision_log:
                return None
                
            record = self.decision_log[decision_id]
            review_data = {
                'decision_id': record.decision_id,
                'timestamp': record.timestamp.isoformat(),
                'context': record.context,
                'rationale': record.rationale,
                'responsible_entity': record.responsible_entity,
                'impact_level': record.impact_level,
                'review_status': record.review_status,
                'explanation': record.explanation,
                'review_history': self.review_history.get(decision_id, [])
            }
            
            self.logger.info(f"Retrieved review data for decision {decision_id}")
            return review_data
            
        except Exception as e:
            self.logger.error(f"Error reviewing decision {decision_id}: {str(e)}")
            return None
    
    def evaluate(self, action: str, context: Dict) -> AccountabilityAssessment:
        """
        Evaluate an action from an accountability perspective
        
        Args:
            action: Proposed action to evaluate
            context: Current context information
            
        Returns:
            AccountabilityAssessment containing the evaluation results
        """
        try:
            # Calculate accountability metrics
            transparency_score = self._calculate_transparency(action, context)
            traceability_score = self._calculate_traceability(action, context)
            responsibility_clarity = self._assess_responsibility_clarity(action, context)
            
            # Generate recommendations and compliance notes
            recommendations = self._generate_recommendations(action, context)
            compliance_notes = self._check_compliance(action, context)
            
            assessment = AccountabilityAssessment(
                transparency_score=transparency_score,
                traceability_score=traceability_score,
                responsibility_clarity=responsibility_clarity,
                review_recommendations=recommendations,
                compliance_notes=compliance_notes
            )
            
            self.logger.info("Completed accountability assessment")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error during accountability assessment: {str(e)}")
            raise
    
    def _simplify_context(self, context: Dict) -> str:
        """Convert complex context data into human-readable format"""
        # Placeholder for context simplification logic
        return str(context)
    
    def _simplify_text(self, text: str) -> str:
        """Convert technical text into simpler language"""
        # Placeholder for text simplification logic
        return text
    
    def _calculate_transparency(self, action: str, context: Dict) -> float:
        """Calculate transparency score for an action"""
        # Placeholder for transparency calculation
        return 0.0
    
    def _calculate_traceability(self, action: str, context: Dict) -> float:
        """Calculate traceability score for an action"""
        # Placeholder for traceability calculation
        return 0.0
    
    def _assess_responsibility_clarity(self, action: str, context: Dict) -> float:
        """Assess clarity of responsibility assignment"""
        # Placeholder for responsibility clarity assessment
        return 0.0
    
    def _generate_recommendations(self, action: str, context: Dict) -> List[str]:
        """Generate accountability recommendations"""
        # Placeholder for recommendations generation
        return []
    
    def _check_compliance(self, action: str, context: Dict) -> List[str]:
        """Check compliance with accountability requirements"""
        # Placeholder for compliance checking
        return [] 