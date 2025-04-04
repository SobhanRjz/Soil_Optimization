import numpy as np
import random
import sys
import os
from typing import List, Tuple, Optional, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Database.PostgreSQL import PostgreSQLDatabase
from models import InputData, ResultData, ParticleConfig
import PLAXIS.Input_PLAXIS as InputModel
import PLAXIS.Output as OutputModel
from Config.config import MainConfig
import math
from dataclasses import dataclass
import logging
import json

# Import the new modular components
from Optimization.algorithms.base import BaseOptimizer
from Optimization.evaluation.model_evaluator import ModelEvaluator
from Optimization.algorithms.particle_generator import ParticleGenerator

@dataclass
class Penalty:
    elementPenalty_Structures: float
    elementPenalty_Soil: float
    uTotalpenalty: float

class HybridPSOHS(BaseOptimizer):
    """
    Hybrid Particle Swarm Optimization and Harmony Search algorithm for 
    optimizing geotechnical reinforcement designs.
    """
    
    def __init__(self, population_size: int = 20, harmony_memory_size: int = 40, 
                 max_iter: int = 100, inertia_weight: float = 0.7, 
                 cognitive_weight: float = 1.5, social_weight: float = 2.0,
                 harmony_consideration_rate: float = 0.9, 
                 pitch_adjustment_rate: float = 0.3):
        """
        Initialize the hybrid PSO-HS optimizer.
        
        Args:
            population_size: Size of the PSO population
            harmony_memory_size: Size of the harmony memory
            max_iter: Maximum number of iterations
            inertia_weight: Inertia weight for PSO
            cognitive_weight: Cognitive weight for PSO
            social_weight: Social weight for PSO
            harmony_consideration_rate: Harmony consideration rate for HS
            pitch_adjustment_rate: Pitch adjustment rate for HS
        """
        super().__init__(max_iter=max_iter)
        
        # PSO Parameters
        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        # HS Parameters
        self.harmony_memory_size = harmony_memory_size
        self.harmony_consideration_rate = harmony_consideration_rate
        self.pitch_adjustment_rate = pitch_adjustment_rate
        
        # Initialize helper classes
        self.particle_generator = ParticleGenerator()
        self.model_evaluator = ModelEvaluator()
        
        # Initialize algorithm state
        self.particles = []
        self.harmony_memory = []
        self.all_particles = []
        self.hs_best_solution = None
        self.hs_best_score = float("inf")
        self.hs_best_penalty = 0.05
        self.penalty_weight = 20.0

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Define parameter choices
        # Set random seed for reproducibility
        random.seed(42)
        self.rebar_dia_List = [28, 30, 32, 36]
        self.nail_len_List = list(range(3, 30, 3))  # 1 to 40 meters, step 5
        self.nail_teta_List = [5, 10, 15, 20]  # Degrees
        self.nail_h_space_List = [1.2, 1.5, 1.8, 2, 2.2, 2.5]  # cm
        self.nail_v_space_List = [1.2, 1.5, 1.8, 2, 2.2, 2.5]  # cm
        
        # Initialize PSO particles
        
        self.global_best_position = None
        self.global_best_score = float("inf")
        self.global_best_penalty = 0.05
        self.global_best_combined_score = float("inf")

        #connect to the DataBase
        self.DataBase = PostgreSQLDatabase()
        self.DataBase.initialize_tables()

        # Initialize missing variables
        self.processing_queue = []  # Queue for particles to be processed

    def _getBestLength(self, RebarDiameter):
        config = MainConfig()
        FS_Structures = 1.8
        RebarArea = (RebarDiameter / 1000) ** 2 * math.pi / 4
        Fy = config.STRUCTURAL_MATERIALS.geogrid.Fy * 1000 # KN/m2
        structure_Strength = Fy * RebarArea  / FS_Structures
    
        FS_Soil = 2.0
        NominalBond = 60 * 6.89475728 # Kn / m2
        HollowDiameter = config.STRUCTURAL_MATERIALS.geogrid.HollowDiameter # M
        NailLength = config.STRUCTURAL_MATERIALS.geogrid.nail_length # M

        BestLength = structure_Strength * FS_Soil/ (math.pi * HollowDiameter * NominalBond)

        return BestLength
    
    def initialize_particle(self, algoName):
        """Initialize a particle with pattern-based nail lengths and non-zero velocity."""
        
        # First select vertical spacing
        nail_v_space = random.choice(self.nail_v_space_List)
        
        # Then select horizontal spacing ensuring nail_v_space * nail_h_space < 4
        valid_h_spaces = [h for h in self.nail_h_space_List if h * nail_v_space < 4]
        if not valid_h_spaces:  # Fallback if no valid combinations
            nail_v_space = min(self.nail_v_space_List)
            valid_h_spaces = [h for h in self.nail_h_space_List if h * nail_v_space < 4]
        
        nail_h_space = random.choice(valid_h_spaces)
        rebar_dia = random.choice(self.rebar_dia_List)
        
        # Nail length pattern parameters
        max_length = random.choice([l for l in self.nail_len_List if l > min(self.nail_len_List)])
        min_length = random.choice([l for l in self.nail_len_List if l <= max_length and l >= 3])
        
        pattern_type = random.choice(["linear", "exponential", "stepped", "random"])
        
        position = InputData(
            rebar_dia=rebar_dia,
            nail_len_pattern={
                "max_length": max_length,
                "min_length": min_length,
                "pattern_type": pattern_type
            },
            nail_teta=random.choice(self.nail_teta_List),
            nail_h_space=nail_h_space,
            nail_v_space=nail_v_space,
            Algo_Name=f"Random {algoName}"
        )
        
        # Generate actual nail lengths based on pattern
        position.nail_len = self._generate_nail_lengths_from_pattern(position)
        
        # Initialize with random non-zero velocity instead of zeros
        velocity = np.array([
            random.uniform(-0.5, 0.5) * (max(self.rebar_dia_List) - min(self.rebar_dia_List)),
            random.uniform(-0.5, 0.5) * (max(self.nail_len_List) - min(self.nail_len_List)),
            random.uniform(-0.5, 0.5) * (max(self.nail_len_List) - min(self.nail_len_List)),
            random.uniform(-0.5, 0.5) * 4.0,  # Pattern type range
            random.uniform(-0.5, 0.5) * (max(self.nail_teta_List) - min(self.nail_teta_List)),
            random.uniform(-0.5, 0.5) * (max(self.nail_h_space_List) - min(self.nail_h_space_List)),
            random.uniform(-0.5, 0.5) * (max(self.nail_v_space_List) - min(self.nail_v_space_List))
        ])
        
        return ParticleConfig(
            position=position,
            velocity=velocity,  # Use random velocity instead of zeros
            best_position=None,
            best_score=float("inf")
        )

    def _generate_nail_lengths_from_pattern(self, position):
        """Generate nail lengths based on pattern parameters."""
        main_config = MainConfig()
        num_nails = int(((main_config.MODEL_GEOMETRY.plate_length - 1.5) // position.nail_v_space) + 1)
        
        max_len = position.nail_len_pattern["max_length"]
        min_len = position.nail_len_pattern["min_length"]
        pattern = position.nail_len_pattern["pattern_type"]
        
        # Ensure max_len is greater than 3
        if max_len <= 3:
            max_len = min([l for l in self.nail_len_List if l > 3])
        
        
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
        nail_lengths = [self.closest_value(length, self.nail_len_List) for length in nail_lengths]
        
        # Ensure only the last nail can be 3, all others must be greater
        for i in range(len(nail_lengths) - 1):
            if nail_lengths[i] <= 3:
                nail_lengths[i] = min([l for l in self.nail_len_List if l > 3])
        
        return nail_lengths

    def evaluate_fitness(self, inputData : InputData):
        """Define the objective function (modify for real-world constraints)."""
        cost = (inputData.rebar_dia * 10) + (inputData.nail_len * 15) + (abs(inputData.nail_teta - 60) * 5) + (inputData.nail_h_space * 2) + (inputData.nail_v_space * 1.5)
        return cost
    
    def update_velocity_and_position(self, particle, algoName="", first_iteration=False, current_inertia=None):
        """Update particle velocity and position in PSO."""
        # Generate random coefficients for cognitive and social components
        r1, r2 = np.random.rand(2)
        
        # Extract pattern parameters from current position
        current_pattern = particle.position.nail_len_pattern
        
        # Check if current_pattern is a string and needs to be parsed
        if isinstance(current_pattern, str):
            current_pattern = json.loads(current_pattern)
            
        # Create vectorized representation of current position
        current_position = np.array([
            float(particle.position.rebar_dia),
            float(current_pattern["max_length"]),
            float(current_pattern["min_length"]),
            float(1 if current_pattern["pattern_type"] == "linear" else 
                 2 if current_pattern["pattern_type"] == "exponential" else 
                 3 if current_pattern["pattern_type"] == "stepped" else 4),  # Encode pattern type
            float(particle.position.nail_teta),
            float(particle.position.nail_h_space),
            float(particle.position.nail_v_space)
        ])
        
        # Handle best position vectors
        if particle.best_position:
            best_pattern = particle.best_position.nail_len_pattern
            best_position = np.array([
                float(particle.best_position.rebar_dia),
                float(best_pattern["max_length"]),
                float(best_pattern["min_length"]),
                float(1 if best_pattern["pattern_type"] == "linear" else 
                     2 if best_pattern["pattern_type"] == "exponential" else 
                     3 if best_pattern["pattern_type"] == "stepped" else 4),
                float(particle.best_position.nail_teta),
                float(particle.best_position.nail_h_space),
                float(particle.best_position.nail_v_space)
            ])
        else:
            # If no best position yet, use global best or add randomness
            if self.global_best_position:
                # Extract pattern from global best position
                global_pattern = self.global_best_position.nail_len_pattern
                # Create vectorized representation of global best position
                best_position = np.array([
                    float(self.global_best_position.rebar_dia),
                    float(global_pattern["max_length"]),
                    float(global_pattern["min_length"]),
                    float(1 if global_pattern["pattern_type"] == "linear" else 
                         2 if global_pattern["pattern_type"] == "exponential" else 
                         3 if global_pattern["pattern_type"] == "stepped" else 4),
                    float(self.global_best_position.nail_teta),
                    float(self.global_best_position.nail_h_space),
                    float(self.global_best_position.nail_v_space)
                ])  # Use global best as reference
            else:
                best_position = current_position.copy()
                # Add significant randomness for exploration when no best exists
                exploration_factor = 0.2
                best_position += np.random.uniform(-exploration_factor, exploration_factor, size=best_position.shape)
        
        # Handle global best position
        if self.global_best_position:
            global_pattern = self.global_best_position.nail_len_pattern
            global_best_position = np.array([
                float(self.global_best_position.rebar_dia),
                float(global_pattern["max_length"]),
                float(global_pattern["min_length"]),
                float(1 if global_pattern["pattern_type"] == "linear" else 
                     2 if global_pattern["pattern_type"] == "exponential" else 
                     3 if global_pattern["pattern_type"] == "stepped" else 4),
                float(self.global_best_position.nail_teta),
                float(self.global_best_position.nail_h_space),
                float(self.global_best_position.nail_v_space)
            ])
        else:
            global_best_position = current_position.copy()
            # For first iteration, add some randomness to global_best_position
            if first_iteration:
                exploration_factor = 0.1
                global_best_position += np.random.uniform(-exploration_factor, exploration_factor, size=global_best_position.shape)
        
        # Apply velocity clamping to prevent excessive exploration
        max_velocity = 0.1 * (np.array([
            max(self.rebar_dia_List) - min(self.rebar_dia_List),
            max(self.nail_len_List) - min(self.nail_len_List),
            max(self.nail_len_List) - min(self.nail_len_List),
            4.0,  # Pattern type range
            max(self.nail_teta_List) - min(self.nail_teta_List),
            max(self.nail_h_space_List) - min(self.nail_h_space_List),
            max(self.nail_v_space_List) - min(self.nail_v_space_List)
        ]))
        
        # Use the passed inertia value
        inertia = current_inertia if current_inertia is not None else self.inertia_weight
        
        particle.velocity = (
            inertia * particle.velocity +
            self.cognitive_weight * r1 * (best_position - current_position) +
            self.social_weight * r2 * (global_best_position - current_position)
        )
        
        # Apply velocity clamping
        particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)
        
        # Update position based on velocity
        new_position = current_position + particle.velocity
        
        # Decode pattern type from numerical representation
        pattern_type_value = new_position[3]
        if pattern_type_value < 1.5:
            pattern_type = "linear"
        elif pattern_type_value < 2.5:
            pattern_type = "exponential"
        elif pattern_type_value < 3.5:
            pattern_type = "stepped"
        else:
            pattern_type = "random"
        
        # Ensure max_length >= min_length
        max_length = self.closest_value(new_position[1], self.nail_len_List)
        min_length = self.closest_value(new_position[2], self.nail_len_List)
        if min_length > max_length:
            min_length, max_length = max_length, min_length
        
        # Create new position with updated values
        particle.position = InputData(
            rebar_dia=self.closest_value(new_position[0], self.rebar_dia_List),
            nail_len_pattern={
                "max_length": max_length,
                "min_length": min_length,
                "pattern_type": pattern_type
            },
            nail_len=[],  # Will be generated from pattern
            nail_teta=self.closest_value(new_position[4], self.nail_teta_List),
            nail_h_space=self.closest_value(new_position[5], self.nail_h_space_List),
            nail_v_space=self.closest_value(new_position[6], self.nail_v_space_List),
            Algo_Name=algoName
        )
        
        # Generate nail lengths based on pattern
        particle.position.nail_len = self._generate_nail_lengths_from_pattern(particle.position)
        if isinstance(particle.position.nail_len_pattern, dict):
            particle.position.nail_len_pattern = json.dumps(particle.position.nail_len_pattern)

        # Reset best position and score if needed
        if particle.best_position is None or particle.best_score == float("inf"):
            particle.best_position = None
            particle.best_score = float("inf")
        
        # Log the particle's updated position
        logging.debug(f"Updated particle position: {particle.position}")
        
        return particle

    def generate_new_harmony(self):
        """Generate a new harmony in Harmony Search algorithm."""
        # Initialize pattern parameters
        pattern_type_value = random.uniform(0, 4)  # Random value for pattern type
        
        # Select a random harmony from memory for reference
        selected_harmony = random.choice(self.harmony_memory)
        
        # Extract parameters from the selected harmony
        rebar_dia = selected_harmony.position.rebar_dia
        
        # Get pattern information if available
        if hasattr(selected_harmony.position, 'nail_len_pattern') and selected_harmony.position.nail_len_pattern:
            try:
                if isinstance(selected_harmony.position.nail_len_pattern, str):
                    pattern_data = json.loads(selected_harmony.position.nail_len_pattern)
                else:
                    pattern_data = selected_harmony.position.nail_len_pattern
                    
                max_length = pattern_data.get('max_length', random.choice(self.nail_len_List))
                min_length = pattern_data.get('min_length', random.choice(self.nail_len_List))
            except (json.JSONDecodeError, AttributeError):
                max_length = random.choice(self.nail_len_List)
                min_length = random.choice([l for l in self.nail_len_List if l < max_length])
        else:
            # If no pattern info, use mean of nail lengths or random values
            if selected_harmony.position.nail_len:
                avg_length = np.mean(selected_harmony.position.nail_len)
                max_length = self.closest_value(avg_length * 1.2, self.nail_len_List)
                min_length = self.closest_value(avg_length * 0.8, self.nail_len_List)
            else:
                max_length = random.choice(self.nail_len_List)
                min_length = random.choice([l for l in self.nail_len_List if l < max_length])
        
        nail_teta = selected_harmony.position.nail_teta
        nail_h_space = selected_harmony.position.nail_h_space
        nail_v_space = selected_harmony.position.nail_v_space
        
        # Apply harmony consideration and pitch adjustment
        params = [rebar_dia, max_length, min_length, pattern_type_value, nail_teta, nail_h_space, nail_v_space]
        choices_lists = [
            self.rebar_dia_List, 
            self.nail_len_List, 
            self.nail_len_List, 
            None,  # Pattern type is continuous
            self.nail_teta_List, 
            self.nail_h_space_List, 
            self.nail_v_space_List
        ]
        
        new_params = []
        for i, (param, choices) in enumerate(zip(params, choices_lists)):
            if random.random() < self.harmony_consideration_rate:
                # Use value from memory with possible pitch adjustment
                if random.random() < self.pitch_adjustment_rate and choices is not None:
                    # Adjust the parameter value within valid range
                    adjustment = random.uniform(-0.1, 0.1) * (max(choices) - min(choices))
                    new_param = self.closest_value(param + adjustment, choices)
                else:
                    new_param = param
            else:
                # Use random value from parameter range
                if choices is not None:
                    new_param = random.choice(choices)
                else:
                    # For pattern type, generate a new random value
                    new_param = random.uniform(0, 4)
            
            new_params.append(new_param)
        
        # Ensure max_length >= min_length
        if new_params[1] < new_params[2]:
            new_params[1], new_params[2] = new_params[2], new_params[1]
        
        # Decode pattern type
        if new_params[3] < 1.0:
            pattern_type = "linear"
        elif new_params[3] < 2.0:
            pattern_type = "exponential"
        elif new_params[3] < 3.0:
            pattern_type = "stepped"
        else:
            pattern_type = "random"
        
        # Create new harmony particle
        particle = ParticleConfig()
        particle.position = InputData(
            rebar_dia=new_params[0],
            nail_len_pattern={
                "max_length": new_params[1],
                "min_length": new_params[2],
                "pattern_type": pattern_type
            },
            nail_len=[],  # Will be generated from pattern
            nail_teta=new_params[4],
            nail_h_space=new_params[5],
            nail_v_space=new_params[6],
            Algo_Name="HS"
        )
        
        # Generate nail lengths based on pattern
        particle.position.nail_len = self._generate_nail_lengths_from_pattern(particle.position)
        
        # Convert pattern to JSON string for storage
        if isinstance(particle.position.nail_len_pattern, dict):
            particle.position.nail_len_pattern = json.dumps(particle.position.nail_len_pattern)
        
        # Initialize particle metrics
        particle.best_position = None
        particle.best_score = float("inf")
        particle.Cost = float("inf")
        
        return particle

    def closest_value(self, value, choices):
        """Find the closest valid discrete value."""
        return min(choices, key=lambda x: abs(x - value))

    def _calculate_Weight(self, output, config):
        """Calculate the cost of the particle."""
        rebarDiameter = config.STRUCTURAL_MATERIALS.geogrid.RebarDiameter
        nail_teta = config.MODEL_GEOMETRY.geogrid_teta
        nail_h_space = config.STRUCTURAL_MATERIALS.geogrid.HorizentalSpace
        nail_v_space = config.STRUCTURAL_MATERIALS.geogrid.VerticalSpace
        SteelVolume = 0
        ConcreteVolume = 0
        for i, nail_length in enumerate(config.STRUCTURAL_MATERIALS.geogrid.nail_length):
            SteelVolume += nail_length * (rebarDiameter / 1000) ** 2 * math.pi / 4 # Steel
            ConcreteVolume += nail_length * ((0.12 **2) * math.pi / 4  - (rebarDiameter / 1000) ** 2 * math.pi / 4) # Concrete

        weightSteel = SteelVolume * 7850  # kg/m3 density of steel and concrete
        weightConcrete = ConcreteVolume * 2400  # kg/m3 density of steel and concrete

        weightSteel = weightSteel / (nail_h_space * nail_v_space) / 1000 # ton per m2
        weightConcrete = weightConcrete / (nail_h_space * nail_v_space) / 1000 # ton per m2

        return weightSteel, weightConcrete
    
    def _calculate_penalty(self, output) -> Penalty:
        """Calculate the penalty of the particle using vectorized operations."""
        max_ratio_Structures = 1.0
        max_ratio_Soil = 1.0
        max_uTotal = 0.03  # M

        # Get all ratios and displacements
        structure_ratios = np.array([data.max_ratio_Structures for data in output.ElementData.elementForceData.values()])
        soil_ratios = np.array([data.max_ratio_Soil for data in output.ElementData.elementForceData.values()])
        displacements = np.array([data.utotal for data in output.ElementData.displacementData])
        
        # Calculate penalties using vectorized operations
        elementPenalty_Structures = np.sum(np.maximum(0, structure_ratios - max_ratio_Structures))
        elementPenalty_Soil = np.sum(np.maximum(0, soil_ratios - max_ratio_Soil))
        uTotalpenalty = np.sum(np.maximum(0, displacements - max_uTotal))

        return Penalty(elementPenalty_Structures, elementPenalty_Soil, uTotalpenalty)

    def _calculate_Price(self, config, weightSteel, weightConcrete):
        """Calculate the price of the particle."""
        PriceSteel = weightSteel * config.MATERIAL_PRICE.Steel
        PriceConcrete = weightConcrete * config.MATERIAL_PRICE.Concrete
        return PriceSteel + PriceConcrete
    
    def _calculate_Cost(self, output: Any, config: MainConfig) -> Tuple[float, float]:
        """
        Calculate the cost of the particle.
        
        Args:
            output: The output data from the model
            config: The configuration object
            
        Returns:
            Tuple containing:
                - Cost: The total cost including penalties
                - sum_penalty: The sum of all penalties
        """
        weightSteel, weightConcrete = self._calculate_Weight(output, config)
        Price = self._calculate_Price(config, weightSteel, weightConcrete)
        Penalty = self._calculate_penalty(output)
        ElementPenaltyWeight = 1
        elementPenalty_Soil  = 1
        uTotalPenaltyWeight = 20

        sum_penalty = ElementPenaltyWeight * Penalty.elementPenalty_Structures + elementPenalty_Soil * Penalty.elementPenalty_Soil + uTotalPenaltyWeight * Penalty.uTotalpenalty
        sum_penalty *= 2
        
        Cost = Price * (1 + sum_penalty)
        # Convert Cost from np.float64 to float to ensure compatibility
        if isinstance(Cost, np.float64):
            Cost = float(Cost)
        if isinstance(sum_penalty, np.float64):
            sum_penalty = float(sum_penalty)
        return Cost, sum_penalty

    def _save_status(self, iteration_count, processed_count, max_to_process, initialize=False):
        """Save the optimization status to a JSON file."""
        # Create output data dictionary
        output_data = {
            "iteration": iteration_count,
            "global_best_score": self.global_best_score,
            "processed_count": processed_count,
            "max_to_process": max_to_process,
            "_IsTerminate": False if initialize else None
        }
        
        # Add report data with current best solution details
        if hasattr(self, 'global_best_position') and self.global_best_position:
            output_data["report"] = {
                "best_rebar_dia": self.global_best_position.rebar_dia,
                "best_nail_pattern": self.global_best_position.nail_len_pattern,
                "best_nail_teta": self.global_best_position.nail_teta,
                "best_nail_h_space": self.global_best_position.nail_h_space,
                "best_nail_v_space": self.global_best_position.nail_v_space,
                "best_score": self.global_best_score
            }
        
        # Check if file exists and read existing data
        file_path = 'optimization_status.json'
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.logger.warning(f"Could not read existing status file. Creating new file.")
        
        # Update existing data with new values
        existing_data.update(output_data)
        
        # Preserve termination flag if it exists and we're not initializing
        if not initialize and "_IsTerminate" in existing_data:
            existing_data["_IsTerminate"] = existing_data.get("_IsTerminate", False)
        
        # Write updated data back to file
        try:
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
        except IOError as e:
            self.logger.error(f"Failed to write optimization status: {str(e)}")
        
    def close_plaxis_output(self):
        """
        Closes any running PLAXIS Output processes to free up resources.
        This is useful to call periodically during optimization to prevent memory leaks.
        """
        self.logger.info("Attempting to close PLAXIS Output processes...")
        try:
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                # For Windows - forcefully terminate Plaxis2DXOutput.exe
                result = subprocess.run(
                    ['taskkill', '/F', '/IM', 'Plaxis2DOutput.exe'],
                    shell=True, 
                    stderr=subprocess.PIPE, 
                    stdout=subprocess.PIPE
                )
                if result.returncode == 0:
                    self.logger.info("Successfully closed PLAXIS Output processes")
                else:
                    self.logger.warning(f"No PLAXIS Output processes found to close: {result.stderr.decode()}")
            else:
                # For Linux/Mac
                result = subprocess.run(
                    ['pkill', '-f', 'Plaxis2DXOutput'],
                    shell=True, 
                    stderr=subprocess.PIPE, 
                    stdout=subprocess.PIPE
                )
                if result.returncode == 0:
                    self.logger.info("Successfully closed PLAXIS Output processes")
                else:
                    self.logger.warning("No PLAXIS Output processes found to close")
                    
            # Wait briefly to ensure processes are fully terminated
            import time
            time.sleep(1)
            
            return True
        except Exception as e:
            self.logger.error(f"Error closing PLAXIS Output processes: {str(e)}")
            return False

    def optimize(self):
        """
        Run the hybrid PSO-HS optimization algorithm.
        
        Returns:
            Tuple containing:
                - best_position: The best position found
                - best_score: The best score found
        """
        # Initialize configuration and model
        config = MainConfig()
        #model = InputModel.PlaxisModelInput()
        
        # Initialize first particle with specific parameters
        first_position = InputData(
            rebar_dia=36,
            nail_len=[27, 27, 27, 27, 27, 27],
            nail_len_pattern=json.dumps({
                "max_length": 27, 
                "min_length": 27, 
                "pattern_type": "random"
            }),
            nail_teta=5,
            nail_h_space=1.2,
            nail_v_space=1.2,
            Algo_Name="PSO HS First Model"
        )

        # Insert first particle into database
        input_id, input_hash = self.DataBase.insert_input(first_position, status=0)
        first_particle = self.particle_generator.initialize_particle("First")
        first_particle.input_hash = input_hash
        
        # Add first particle to particles and harmony memory
        self.particles.append(first_particle)
        self.harmony_memory.append(first_particle)
        
        # Initialize additional particles and harmonies
        self._initialize_population()
        
        # Main optimization loop
        iteration_count = 0
        first_iteration = True
        current_inertia = self.inertia_weight
        
        while iteration_count < self.max_iter:
            processed_count = 0
            max_to_process = len(self.particles) + len(self.harmony_memory)
            
            # Process particles
            while processed_count < max_to_process:
                # Get next input to process
                input_hash, input_data = self.DataBase._get_input_Data()
                if not input_hash:
                    break
                
                try:
                    # Evaluate the model using ModelEvaluator
                    cost, penalty, output = self.model_evaluator.evaluate_model(input_data, config)
                    
                    # Create result data
                    result_data = ResultData(
                        Total_Displacement=round(max(d.utotal for d in output.ElementData.displacementData), 4),
                        Total_Displacement_Allow=0.03,
                        max_Structure_Ratio=round(max(d.max_ratio_Structures for d in output.ElementData.elementForceData.values()), 4),
                        max_PullOut_Ratio=round(max(d.max_ratio_Soil for d in output.ElementData.elementForceData.values()), 4),
                        Penalty=round(penalty, 4),
                        Cost=round(cost, 4)
                    )

                    # Insert result into database
                    self.DataBase.insert_result(input_hash, status=2, result_data=result_data)
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating model: {str(e)}")
                    self.DataBase._update_input_status(input_hash=input_hash, status=-1)
                    continue
                
                # Process results
                self._process_results(input_hash, cost, penalty, first_iteration, current_inertia)
            
            # Update iteration parameters
            first_iteration = False
            iteration_count += 1
            
            # Periodically close PLAXIS output
            if iteration_count % 5 == 0:
                self.close_plaxis_output()
            
            # Save optimization status
            self._save_status(iteration_count, processed_count, max_to_process)
        
        # Return best solution found
        return self.global_best_position, self.global_best_score

    def _initialize_population(self):
        """
        Initialize PSO particles and HS harmonies.
        """
        # Initialize HS harmonies
        for i in range(self.harmony_memory_size):
            particle = self.particle_generator.initialize_particle(f"HS_{i}")
            
            # Convert nail_len_pattern to string if it's a dict
            if isinstance(particle.position.nail_len_pattern, dict):
                particle.position.nail_len_pattern = json.dumps(particle.position.nail_len_pattern)
            
            # Insert into database
            input_id, input_hash = self.DataBase.insert_input(particle.position, status=0)
            particle.input_hash = input_hash
            
            # Add to harmony memory if not a duplicate
            if not any(existing_particle.input_hash == input_hash for existing_particle in self.harmony_memory):
                self.harmony_memory.append(particle)
                self.all_particles.append(particle)
        
        # Initialize PSO particles
        for i in range(self.population_size):
            particle = self.particle_generator.initialize_particle(f"PSO_{i}")
            
            # Convert nail_len_pattern to string if it's a dict
            if isinstance(particle.position.nail_len_pattern, dict):
                particle.position.nail_len_pattern = json.dumps(particle.position.nail_len_pattern)
            
            # Insert into database
            input_id, input_hash = self.DataBase.insert_input(particle.position, status=0)
            particle.input_hash = input_hash
            
            # Add to particles and harmony memory if not a duplicate
            if not any(existing_particle.input_hash == input_hash for existing_particle in self.particles):
                self.particles.append(particle)
                self.harmony_memory.append(particle)

    def _process_results(self, input_hash, cost, penalty, first_iteration, current_inertia):
        """
        Process optimization results and update particles.
        
        Args:
            input_hash: Hash of the current input
            cost: Cost of the current solution
            penalty: Penalty of the current solution
            first_iteration: Whether this is the first iteration
            current_inertia: Current inertia weight
        """
        # Check for results that need processing
        rowNumber = self.DataBase._get_result_NonCheck()
        if rowNumber != 0:
            inputHash_Result = self.DataBase._get_lastHash_Result()
            result = self.DataBase._get_result_baseHash(inputHash_Result)

            # Find matching particle in harmony memory
            particle_idx = next((idx for idx, p in enumerate(self.harmony_memory) if p.input_hash == inputHash_Result), None)
            
            if particle_idx is not None:
                # Update particle cost
                self.harmony_memory[particle_idx].Cost = cost
                
                # Update personal best if better
                if result.Cost < self.harmony_memory[particle_idx].best_score and result.Penalty < self.global_best_penalty:
                    self.harmony_memory[particle_idx].best_score = result.Cost
                    self.harmony_memory[particle_idx].best_position = self.harmony_memory[particle_idx].position

                # Update global best
                if result.Cost < self.global_best_score:
                    self.global_best_score = result.Cost
                    self.global_best_position = self.harmony_memory[particle_idx].position
                    self.logger.info(f"New global best ***** : {self.global_best_score} *****")

                # Generate new particle using PSO
                new_particle = self.update_velocity_and_position(
                    self.harmony_memory[particle_idx], 
                    "PSO",
                    first_iteration=first_iteration,
                    current_inertia=current_inertia
                )
                
                # Try to insert new particle
                max_attempts = 10
                attempt_count = 0
                new_input_hash = None
                
                while attempt_count < max_attempts and new_input_hash is None:
                    input_id, new_input_hash = self.DataBase.insert_input(new_particle.position, status=0)
                    
                    # Check for duplicates
                    if any(existing_particle.input_hash == new_input_hash for existing_particle in self.processing_queue):
                        # Modify particle to make it unique
                        if isinstance(new_particle.position.nail_len_pattern, str):
                            pattern_dict = json.loads(new_particle.position.nail_len_pattern)
                        else:
                            pattern_dict = new_particle.position.nail_len_pattern
                        
                        # Adjust parameters
                        if "min_length" in pattern_dict and pattern_dict["min_length"] < max(self.particle_generator.nail_len_list):
                            pattern_dict["min_length"] += 1
                        elif "max_length" in pattern_dict and pattern_dict["max_length"] < max(self.particle_generator.nail_len_list):
                            pattern_dict["max_length"] += 1
                        
                        new_particle.position.nail_len_pattern = json.dumps(pattern_dict)
                        new_input_hash = None  # Reset to try again
                    
                    attempt_count += 1
                
                if new_input_hash:
                    new_particle.input_hash = new_input_hash
                    # Replace old particle and add to processing queue
                    self.harmony_memory[particle_idx] = new_particle
                    self.processing_queue.append(new_particle)
                else:
                    self.logger.warning("Failed to create a unique particle")

                # Generate new harmony
                new_harmony = self.generate_new_harmony()
                
                # Try to insert new harmony
                harmony_input_hash = self._insert_new_harmony(new_harmony)
                
                # Mark result as processed
                self.DataBase._update_result_status(input_hash=inputHash_Result, status=3)

    def _insert_new_harmony(self, new_harmony):
        """
        Insert a new harmony into the database and harmony memory.
        
        Args:
            new_harmony: New harmony particle to insert
            
        Returns:
            Input hash of the inserted harmony
        """
        max_attempts = 10
        attempt_count = 0
        harmony_input_hash = None
        
        while attempt_count < max_attempts and harmony_input_hash is None:
            input_id, harmony_input_hash = self.DataBase.insert_input(new_harmony.position, status=0)
            
            # Check for duplicates
            if any(existing_particle.input_hash == harmony_input_hash for existing_particle in self.harmony_memory):
                # Modify harmony to make it unique
                if isinstance(new_harmony.position.nail_len_pattern, str):
                    pattern_dict = json.loads(new_harmony.position.nail_len_pattern)
                else:
                    pattern_dict = new_harmony.position.nail_len_pattern
                
                # Adjust parameters
                if "min_length" in pattern_dict and pattern_dict["min_length"] < max(self.particle_generator.nail_len_list):
                    pattern_dict["min_length"] += 1
                elif "max_length" in pattern_dict and pattern_dict["max_length"] < max(self.particle_generator.nail_len_list):
                    pattern_dict["max_length"] += 1
                
                new_harmony.position.nail_len_pattern = json.dumps(pattern_dict)
                harmony_input_hash = None  # Reset to try again
            
            attempt_count += 1
        
        if harmony_input_hash:
            new_harmony.input_hash = harmony_input_hash
            
            # Update harmony memory
            if len(self.harmony_memory) >= self.harmony_memory_size:
                # Replace worst harmony
                worst_idx = max(range(len(self.harmony_memory)), 
                               key=lambda i: getattr(self.harmony_memory[i], 'Cost', float('inf')))
                self.harmony_memory[worst_idx] = new_harmony
            else:
                # Add new harmony
                self.harmony_memory.append(new_harmony)
            
            # Add to processing queue
            self.processing_queue.append(new_harmony)
        
        return harmony_input_hash

# Optional: Run the optimizer
if __name__ == "__main__":
    hybrid_optimizer = HybridPSOHS(max_iter=1000)
    best_solution, best_score = hybrid_optimizer.optimize()
    hybrid_optimizer.logger.info(f"Hybrid PSO-HS Best Solution: {best_solution}, Score: {best_score}")
