import random
import numpy as np
import json
from typing import List, Dict, Any

from models import InputData, ParticleConfig
from Config.config import MainConfig


class ParticleGenerator:
    """Class to generate particles for optimization algorithms."""
    
    def __init__(self):
        """Initialize the particle generator with parameter ranges."""
        # Set random seed for reproducibility
        random.seed(42)
        
        # Define parameter choices
        self.rebar_dia_list = [28, 30, 32, 36]
        self.nail_len_list = list(range(3, 30, 3))  # 3 to 27 meters, step 3
        self.nail_teta_list = [5, 10, 15, 20]  # Degrees
        self.nail_h_space_list = [1.2, 1.5, 1.8, 2, 2.2, 2.5]  # m
        self.nail_v_space_list = [1.2, 1.5, 1.8, 2, 2.2, 2.5]  # m
    
    def initialize_particle(self, algo_name: str) -> ParticleConfig:
        """
        Initialize a particle with pattern-based nail lengths and non-zero velocity.
        
        Args:
            algo_name: Name of the algorithm for tracking
            
        Returns:
            Initialized particle configuration
        """
        # First select vertical spacing
        nail_v_space = random.choice(self.nail_v_space_list)
        
        # Then select horizontal spacing ensuring nail_v_space * nail_h_space < 4
        valid_h_spaces = [h for h in self.nail_h_space_list if h * nail_v_space < 4]
        if not valid_h_spaces:  # Fallback if no valid combinations
            nail_v_space = min(self.nail_v_space_list)
            valid_h_spaces = [h for h in self.nail_h_space_list if h * nail_v_space < 4]
        
        nail_h_space = random.choice(valid_h_spaces)
        rebar_dia = random.choice(self.rebar_dia_list)
        
        # Nail length pattern parameters
        max_length = random.choice([l for l in self.nail_len_list if l > min(self.nail_len_list)])
        min_length = random.choice([l for l in self.nail_len_list if l <= max_length and l >= 3])
        
        pattern_type = random.choice(["linear", "exponential", "stepped", "random"])
        
        position = InputData(
            rebar_dia=rebar_dia,
            nail_len_pattern={
                "max_length": max_length,
                "min_length": min_length,
                "pattern_type": pattern_type
            },
            nail_teta=random.choice(self.nail_teta_list),
            nail_h_space=nail_h_space,
            nail_v_space=nail_v_space,
            Algo_Name=f"Random {algo_name}"
        )
        
        # Generate actual nail lengths based on pattern
        position.nail_len = self._generate_nail_lengths_from_pattern(position)
        
        # Initialize with random non-zero velocity
        velocity = np.array([
            random.uniform(-0.5, 0.5) * (max(self.rebar_dia_list) - min(self.rebar_dia_list)),
            random.uniform(-0.5, 0.5) * (max(self.nail_len_list) - min(self.nail_len_list)),
            random.uniform(-0.5, 0.5) * (max(self.nail_len_list) - min(self.nail_len_list)),
            random.uniform(-0.5, 0.5) * 4.0,  # Pattern type range
            random.uniform(-0.5, 0.5) * (max(self.nail_teta_list) - min(self.nail_teta_list)),
            random.uniform(-0.5, 0.5) * (max(self.nail_h_space_list) - min(self.nail_h_space_list)),
            random.uniform(-0.5, 0.5) * (max(self.nail_v_space_list) - min(self.nail_v_space_list))
        ])
        
        return ParticleConfig(
            position=position,
            velocity=velocity,
            best_position=None,
            best_score=float("inf")
        )
    
    def _generate_nail_lengths_from_pattern(self, position: InputData) -> List[float]:
        """
        Generate nail lengths based on pattern parameters.
        
        Args:
            position: Input data containing pattern parameters
            
        Returns:
            List of nail lengths
        """
        main_config = MainConfig()
        num_nails = int(((main_config.MODEL_GEOMETRY.plate_length - 1.5) // position.nail_v_space) + 1)
        
        pattern_data = position.nail_len_pattern
        if isinstance(pattern_data, str):
            pattern_data = json.loads(pattern_data)
            
        max_len = pattern_data["max_length"]
        min_len = pattern_data["min_length"]
        pattern = pattern_data["pattern_type"]
        
        # Ensure max_len is greater than 3
        if max_len <= 3:
            max_len = min([l for l in self.nail_len_list if l > 3])
        
        if pattern == "linear":
            # Linear decrease from max to min
            step = (max_len - min_len) / (num_nails - 1) if num_nails > 1 else 0
            nail_lengths = [max_len - i * step for i in range(num_nails)]
        elif pattern == "exponential":
            # Exponential decrease
            factor = (min_len / max_len) ** (1 / (num_nails - 1)) if num_nails > 1 else 1
            nail_lengths = [max_len * (factor ** i) for i in range(num_nails)]
        elif pattern == "stepped":
            # Step function with random transitions
            steps = random.randint(2, min(4, num_nails))
            step_sizes = sorted(random.sample(range(1, num_nails), steps-1), reverse=True)
            step_sizes = [0] + step_sizes + [num_nails]
            lengths = [random.uniform(min_len, max_len) for _ in range(steps)]
            lengths.sort(reverse=True)
            
            nail_lengths = []
            for i in range(steps):
                nail_lengths.extend([lengths[i]] * (step_sizes[i+1] - step_sizes[i]))
        else:  # random pattern
            # Completely random values between min and max
            nail_lengths = [random.uniform(min_len, max_len) for _ in range(num_nails)]
            # Sort in descending order to maintain general top-to-bottom decrease
            nail_lengths.sort(reverse=True)
        
        # Convert to valid discrete values
        nail_lengths = [self._closest_value(length, self.nail_len_list) for length in nail_lengths]
        
        # Ensure only the last nail can be 3, all others must be greater
        for i in range(len(nail_lengths) - 1):
            if nail_lengths[i] <= 3:
                nail_lengths[i] = min([l for l in self.nail_len_list if l > 3])
        
        return nail_lengths
    
    def _closest_value(self, value: float, choices: List[float]) -> float:
        """
        Find the closest valid discrete value from a list of choices.
        
        Args:
            value: The target value
            choices: List of valid choices
            
        Returns:
            The closest value from the choices list
        """
        return min(choices, key=lambda x: abs(x - value))
    
    def get_parameter_ranges(self) -> Dict[str, List[float]]:
        """
        Get the parameter ranges for the optimization.
        
        Returns:
            Dictionary of parameter ranges
        """
        return {
            "rebar_dia": self.rebar_dia_list,
            "nail_len": self.nail_len_list,
            "nail_teta": self.nail_teta_list,
            "nail_h_space": self.nail_h_space_list,
            "nail_v_space": self.nail_v_space_list
        } 