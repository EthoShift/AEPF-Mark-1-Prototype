from dataclasses import dataclass
from typing import Dict, List, Any
import yaml
from pathlib import Path
from datetime import datetime

@dataclass
class ModelConfig:
    """Persistent model configuration"""
    # Target configuration
    target_column: str = 'Termd'
    
    # Sensitive features
    sensitive_features: List[str] = (
        'GenderID', 'RaceDesc', 'MaritalStatusID', 'Sex', 
        'HispanicLatino', 'DeptID', 'State', 'RecruitmentSource'
    )
    
    # Performance indicators
    performance_features: List[str] = (
        'EngagementSurvey', 'EmpSatisfaction', 
        'SpecialProjectsCount', 'DaysLateLast30', 'Absences'
    )
    
    # Feature weights
    feature_weights: Dict[str, float] = {
        'EngagementSurvey': 0.20,
        'EmpSatisfaction': 0.20,
        'SpecialProjectsCount': 0.15,
        'DaysLateLast30': 0.25,
        'Absences': 0.20
    }

class ConfigManager:
    def __init__(self):
        """Initialize configuration manager"""
        self.config_dir = Path('config')
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / 'model_config.yaml'
        
        # Load or create configuration
        self.config = self._load_or_create_config()
        
        # Initialize sensitivity tracking
        self.sensitivity_log = []
    
    def _load_or_create_config(self) -> Dict:
        """Load existing config or create default"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Create default configuration
        default_config = {
            'model': {
                'target_column': ModelConfig.target_column,
                'sensitive_features': ModelConfig.sensitive_features,
                'performance_features': ModelConfig.performance_features,
                'feature_weights': ModelConfig.feature_weights
            },
            'sensitivity': {
                'demographic_features': {
                    'GenderID': {'weight': 0.15, 'balance_target': 0.5},
                    'RaceDesc': {'weight': 0.15, 'balance_target': 0.3},
                    'MaritalStatusID': {'weight': 0.10, 'monitor': True},
                    'Sex': {'weight': 0.15, 'balance_target': 0.5},
                    'HispanicLatino': {'weight': 0.15, 'balance_target': 0.3}
                },
                'organizational_features': {
                    'DeptID': {'weight': 0.10, 'monitor': True},
                    'State': {'weight': 0.05, 'monitor': True},
                    'RecruitmentSource': {'weight': 0.10, 'monitor': True}
                },
                'performance_thresholds': {
                    'EngagementSurvey': {'min': 3.0, 'target': 4.0},
                    'EmpSatisfaction': {'min': 3.0, 'target': 4.0},
                    'SpecialProjectsCount': {'min': 1, 'target': 3},
                    'DaysLateLast30': {'max': 3, 'target': 0},
                    'Absences': {'max': 5, 'target': 2}
                }
            }
        }
        
        # Save configuration
        with open(self.config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def get_target_column(self) -> str:
        """Get configured target column"""
        return self.config['model']['target_column']
    
    def get_sensitive_features(self) -> List[str]:
        """Get list of sensitive features"""
        return self.config['model']['sensitive_features']
    
    def get_performance_features(self) -> List[str]:
        """Get list of performance features"""
        return self.config['model']['performance_features']
    
    def get_feature_weights(self) -> Dict[str, float]:
        """Get feature weights"""
        return self.config['model']['feature_weights']
    
    def get_sensitivity_settings(self) -> Dict[str, Any]:
        """Get sensitivity settings"""
        return self.config['sensitivity']
    
    def log_sensitivity_adjustment(self, feature: str, original: float, adjusted: float, reason: str):
        """Log sensitivity adjustment"""
        self.sensitivity_log.append({
            'timestamp': datetime.now().isoformat(),
            'feature': feature,
            'original_value': original,
            'adjusted_value': adjusted,
            'adjustment_factor': adjusted - original,
            'reason': reason
        })
    
    def get_sensitivity_summary(self) -> Dict[str, Any]:
        """Get summary of sensitivity adjustments"""
        if not self.sensitivity_log:
            return {"message": "No sensitivity adjustments recorded"}
        
        summary = {
            "total_adjustments": len(self.sensitivity_log),
            "features_adjusted": {},
            "average_adjustment": 0.0,
            "significant_adjustments": []
        }
        
        total_adjustment = 0
        for log in self.sensitivity_log:
            feature = log['feature']
            adjustment = log['adjustment_factor']
            
            # Track feature adjustments
            if feature not in summary['features_adjusted']:
                summary['features_adjusted'][feature] = []
            summary['features_adjusted'][feature].append(adjustment)
            
            # Track significant adjustments
            if abs(adjustment) > 0.1:
                summary['significant_adjustments'].append(log)
            
            total_adjustment += adjustment
        
        # Calculate averages
        summary['average_adjustment'] = total_adjustment / len(self.sensitivity_log)
        
        # Calculate feature-specific statistics
        for feature in summary['features_adjusted']:
            adjustments = summary['features_adjusted'][feature]
            summary['features_adjusted'][feature] = {
                'count': len(adjustments),
                'average': sum(adjustments) / len(adjustments),
                'max': max(adjustments),
                'min': min(adjustments)
            }
        
        return summary

def get_config() -> ConfigManager:
    """Get configuration manager instance"""
    return ConfigManager() 