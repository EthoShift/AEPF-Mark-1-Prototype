from typing import Dict, List, Optional, Any, Union
import logging
from scripts.context_models import ContextEntry, StakeholderData, RealTimeMetrics
from scripts.location_context import LocationContextManager, RegionalContext
from datetime import datetime

class ContextEngine:
    """
    Context Engine for AEPF Mk1 - Handles context processing, management and analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.context_store: Dict[str, ContextEntry] = {}
        self.context_history: List[Dict[str, Any]] = []
        self.location_manager = LocationContextManager()
        self.current_region: Optional[str] = None
        
    def add_context_entry(self, entry: ContextEntry) -> None:
        """
        Add or update a context entry
        
        Args:
            entry: ContextEntry instance containing either StakeholderData or RealTimeMetrics
        """
        self.context_store[entry.id] = entry
        self.context_history.append({
            'action': 'add',
            'entry_type': entry.entry_type,
            'entry_id': entry.id
        })
        self.logger.debug(f"Added context entry: {entry.id}")
        
    def get_context(self, key: str) -> Optional[Any]:
        """
        Retrieve context information by key
        
        Args:
            key: Context identifier
            
        Returns:
            Context value if found, None otherwise
        """
        value = self.context_store.get(key)
        if value is None:
            self.logger.debug(f"Context key not found: {key}")
        return value
        
    def remove_context(self, key: str) -> bool:
        """
        Remove context information
        
        Args:
            key: Context identifier
            
        Returns:
            True if context was removed, False if key wasn't found
        """
        if key in self.context_store:
            del self.context_store[key]
            self.context_history.append({
                'action': 'remove',
                'key': key
            })
            self.logger.debug(f"Removed context: {key}")
            return True
        return False
        
    def clear_context(self) -> None:
        """Clear all context information"""
        self.context_store.clear()
        self.context_history.append({
            'action': 'clear'
        })
        self.logger.debug("Cleared all context")
        
    def get_context_history(self) -> List[Dict[str, Any]]:
        """
        Get history of context operations
        
        Returns:
            List of context operations with their details
        """
        return self.context_history
        
    def analyze_context(self) -> Dict[str, Any]:
        """
        Analyze current context state
        
        Returns:
            Dictionary containing context analysis results
        """
        analysis = {
            'total_keys': len(self.context_store),
            'keys': list(self.context_store.keys()),
            'history_length': len(self.context_history)
        }
        return analysis
    
    def set_location_context(self, region_id: str) -> bool:
        """Set the current location context"""
        context = self.location_manager.get_context(region_id)
        if context:
            self.current_region = region_id
            self.context_history.append({
                'action': 'set_location',
                'region_id': region_id,
                'timestamp': datetime.now().isoformat()
            })
            return True
        return False
    
    def get_location_context(self) -> Optional[RegionalContext]:
        """Get current location context"""
        if self.current_region:
            return self.location_manager.get_context(self.current_region)
        return None
    
    def adjust_for_location(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust weights based on location context"""
        context = self.get_location_context()
        if context:
            return self.location_manager.adjust_weights(weights, context)
        return weights
    
    def get_decision_context(self, action: str) -> Dict:
        """Gather comprehensive context for decision-making"""
        context = {
            'stakeholders': self._get_relevant_stakeholders(),
            'metrics': self._get_relevant_metrics(),
            'location': self.get_location_context(),
            'compliance': self._get_compliance_requirements(),
            'historical_data': self._get_historical_context(action)
        }
        
        # Add probability-relevant context
        context['risk_factors'] = self._analyze_risk_factors(action)
        context['previous_decisions'] = self._get_related_decisions(action)
        
        return context