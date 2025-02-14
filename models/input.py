from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class InputData:
    rebar_dia: float = 0.0
    nail_len: list[int] = field(default_factory=list)
    nail_teta: float = 0.0
    nail_h_space: float = 0.0
    nail_v_space: float = 0.0
    Algo_Name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary"""
        return {
            'rebar_dia': self.rebar_dia,
            'nail_len': self.nail_len,
            'nail_teta': self.nail_teta,
            'nail_h_space': self.nail_h_space,
            'nail_v_space': self.nail_v_space,
            'Algo_Name': self.Algo_Name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InputData':
        """Create instance from dictionary"""
        return cls(**data)