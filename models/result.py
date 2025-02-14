from dataclasses import dataclass
from typing import Dict

@dataclass
class ResultData:
    Total_Displacement: float = 0.0
    Total_Displacement_Allow: float = 0.0
    max_Structure_Ratio: float = 0.0
    max_Soil_Ratio: float = 0.0
    Penalty: float = 0.0
    Cost: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert dataclass to dictionary"""
        return {
            'Total_Displacement': self.Total_Displacement,
            'Total_Displacement_Allow': self.Total_Displacement_Allow,
            'Structure_Ratio': self.Structure_Ratio,
            'Soil_Ratio': self.Soil_Ratio,
            'Penalty': self.Penalty,
            'Cost': self.Cost
        }