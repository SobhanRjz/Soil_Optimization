from dataclasses import dataclass
from .input import InputData
from typing import Dict, List, Optional

@dataclass
class OptimizationConfig:
    population_size: int
    mutation_rate: float
    crossover_rate: float
    max_iterations: int
    
@dataclass
class OptimizationBounds:
    rebar_dia: tuple[float, float]
    nail_len: tuple[float, float]
    nail_teta: tuple[float, float]
    nail_h_space: tuple[float, float]
    nail_v_space: tuple[float, float]

@dataclass
class OptimizationResult:
    best_solution: InputData
    best_fitness: float
    iteration_history: List[float]
    execution_time: float

@dataclass
class ParticleConfig:
    input_hash: str = ""
    position: InputData = None
    velocity: List[float] = None
    best_position: Optional[InputData] = None 
    best_score: float = float("inf")
    Cost: float = float("inf")
    best_penalty: float = float("inf")