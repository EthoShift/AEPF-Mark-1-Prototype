from typing import Dict, List, Optional, Any, Union
import logging
from scripts.context_models import ContextEntry, StakeholderData, RealTimeMetrics

class ContextEngine:
    """
    Context Engine for AEPF Mk1 - Handles context processing, management and analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.context_store: Dict[str, ContextEntry] = {}
        self.context_history: List[Dict[str, Any]] = []
        
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