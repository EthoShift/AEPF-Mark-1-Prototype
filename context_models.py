from dataclasses import dataclass
from datetime import datetime
from typing import Union, Optional
from enum import Enum

class StakeholderRole(Enum):
    """Enumeration of possible stakeholder roles"""
    USER = "user"
    REGULATOR = "regulator"
    DEVELOPER = "developer"
    MANAGER = "manager"
    CUSTOMER = "customer"

class MetricSource(Enum):
    """Enumeration of possible metric sources"""
    SENSOR = "sensor"
    USER_FEEDBACK = "user_feedback"
    SYSTEM = "system"
    EXTERNAL_API = "external_api"
    CALCULATION = "calculation"

@dataclass
class StakeholderData:
    """Data class representing stakeholder information in AEPF Mk1"""
    id: int
    name: str
    role: StakeholderRole
    region: str
    priority_level: int
    impact_score: float

    def __post_init__(self):
        """Validate data after initialization"""
        if not 1 <= self.priority_level <= 5:
            raise ValueError("Priority level must be between 1 and 5")
        if not 0 <= self.impact_score <= 100:
            raise ValueError("Impact score must be between 0 and 100")
        if not isinstance(self.role, StakeholderRole):
            self.role = StakeholderRole(self.role)

@dataclass
class RealTimeMetrics:
    """Data class representing real-time metrics in AEPF Mk1"""
    metric_name: str
    value: float
    timestamp: datetime = datetime.now()
    source: MetricSource = MetricSource.SYSTEM

    def __post_init__(self):
        """Validate data after initialization"""
        if not isinstance(self.source, MetricSource):
            self.source = MetricSource(self.source)

@dataclass
class ContextEntry:
    """Wrapper class for context entries in AEPF Mk1"""
    entry_type: str
    data: Union[StakeholderData, RealTimeMetrics]
    metadata: Optional[dict] = None

    def __post_init__(self):
        """Validate entry type"""
        valid_types = {'stakeholder', 'metric'}
        if self.entry_type not in valid_types:
            raise ValueError(f"Entry type must be one of: {valid_types}")
        
        # Validate data type matches entry_type
        if self.entry_type == 'stakeholder' and not isinstance(self.data, StakeholderData):
            raise TypeError("Data must be StakeholderData for stakeholder entry type")
        if self.entry_type == 'metric' and not isinstance(self.data, RealTimeMetrics):
            raise TypeError("Data must be RealTimeMetrics for metric entry type")

    @property
    def id(self) -> str:
        """Generate a unique identifier for the context entry"""
        if isinstance(self.data, StakeholderData):
            return f"stakeholder_{self.data.id}"
        return f"metric_{self.data.metric_name}_{self.data.timestamp.isoformat()}" 