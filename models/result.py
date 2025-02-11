from dataclasses import dataclass
from typing import Dict

@dataclass
class ResultData:
    Total_Distance: float = 0.0
    Total_Distance_Allow: float = 0.0
    Structure_Force: float = 0.0
    Structure_Force_Allow: float = 0.0
    Soil_Force: float = 0.0
    Soil_Force_Allow: float = 0.0
    Penalty: float = 0.0
    Cost: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert dataclass to dictionary"""
        return {
            'Total_Distance': self.Total_Distance,
            'Total_Distance_Allow': self.Total_Distance_Allow,
            'Structure_Force': self.Structure_Force,
            'Structure_Force_Allow': self.Structure_Force_Allow,
            'Soil_Force': self.Soil_Force,
            'Soil_Force_Allow': self.Soil_Force_Allow,
            'Penalty': self.Penalty,
            'Cost': self.Cost
        }