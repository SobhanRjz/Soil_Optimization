import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Any

from Config.config import MainConfig
import PLAXIS.Input_PLAXIS as InputModel
import PLAXIS.Output as OutputModel


@dataclass
class Penalty:
    """Class to store penalty values for constraint violations."""
    elementPenalty_Structures: float
    elementPenalty_Soil: float
    uTotalpenalty: float


class ModelEvaluator:
    """Class to handle model evaluation and fitness calculation."""
    
    def __init__(self, penalty_weight: float = 20.0):
        """
        Initialize the model evaluator.
        
        Args:
            penalty_weight: Weight for penalty in combined score
        """
        self.penalty_weight = penalty_weight
        self.model = InputModel.PlaxisModelInput()
    
    def evaluate_model(self, input_data, config: MainConfig) -> Tuple[float, float, Any]:
        """
        Evaluate a model with the given input parameters.
        
        Args:
            input_data: Input data for the model
            config: Configuration object
            
        Returns:
            Tuple containing:
                - cost: The calculated cost
                - penalty: The calculated penalty
                - output: The model output data
        """
        # Configure model with input parameters
        config.STRUCTURAL_MATERIALS.geogrid.nail_length = input_data.nail_len
        config.STRUCTURAL_MATERIALS.geogrid.RebarDiameter = input_data.rebar_dia
        config.STRUCTURAL_MATERIALS.geogrid.HorizentalSpace = input_data.nail_h_space 
        config.STRUCTURAL_MATERIALS.geogrid.VerticalSpace = input_data.nail_v_space
        config.MODEL_GEOMETRY.geogrid_teta = input_data.nail_teta
        
        # Run model and calculate results
        self.model.config = config
        
        # Try to create model and get output with retry logic
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                self.model.Create_Model()
                
                output = OutputModel.PlaxisModelOutput()
                output.GetOutput()
                success = True
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"Failed to execute PLAXIS model after {max_retries} attempts: {str(e)}")
                # Wait briefly before retrying
                import time
                time.sleep(2)
        
        cost, penalty = self._calculate_cost(output, config)
        return cost, penalty, output
    
    def _calculate_weight(self, output, config) -> Tuple[float, float]:
        """
        Calculate the weight of steel and concrete.
        
        Args:
            output: The output data from the model
            config: The configuration object
            
        Returns:
            Tuple containing:
                - weightSteel: Weight of steel in tons per m²
                - weightConcrete: Weight of concrete in tons per m²
        """
        rebarDiameter = config.STRUCTURAL_MATERIALS.geogrid.RebarDiameter
        nail_h_space = config.STRUCTURAL_MATERIALS.geogrid.HorizentalSpace
        nail_v_space = config.STRUCTURAL_MATERIALS.geogrid.VerticalSpace
        
        steel_volume = 0
        concrete_volume = 0
        
        for nail_length in config.STRUCTURAL_MATERIALS.geogrid.nail_length:
            # Calculate steel volume (m³)
            steel_volume += nail_length * (rebarDiameter / 1000) ** 2 * math.pi / 4
            
            # Calculate concrete volume (m³)
            hollow_diameter = 0.12  # 12 cm diameter in meters
            concrete_volume += nail_length * ((hollow_diameter **2) * math.pi / 4 - 
                                             (rebarDiameter / 1000) ** 2 * math.pi / 4)

        # Convert to weight (kg)
        weight_steel = steel_volume * 7850    # kg/m³ density of steel
        weight_concrete = concrete_volume * 2400  # kg/m³ density of concrete

        # Convert to tons per m²
        weight_steel = weight_steel / (nail_h_space * nail_v_space) / 1000
        weight_concrete = weight_concrete / (nail_h_space * nail_v_space) / 1000

        return weight_steel, weight_concrete
    
    def _calculate_penalty(self, output) -> Penalty:
        """
        Calculate the penalty for constraint violations.
        
        Args:
            output: The output data from the model
            
        Returns:
            Penalty object with calculated penalties
        """
        max_ratio_Structures = 1.0
        max_ratio_Soil = 1.0
        max_uTotal = 0.03  # M

        # Get all ratios and displacements using vectorized operations
        structure_ratios = np.array([data.max_ratio_Structures for data in output.ElementData.elementForceData.values()])
        soil_ratios = np.array([data.max_ratio_Soil for data in output.ElementData.elementForceData.values()])
        displacements = np.array([data.utotal for data in output.ElementData.displacementData])
        
        # Calculate penalties using vectorized operations
        elementPenalty_Structures = np.sum(np.maximum(0, structure_ratios - max_ratio_Structures))
        elementPenalty_Soil = np.sum(np.maximum(0, soil_ratios - max_ratio_Soil))
        uTotalpenalty = np.sum(np.maximum(0, displacements - max_uTotal))

        return Penalty(elementPenalty_Structures, elementPenalty_Soil, uTotalpenalty)
    
    def _calculate_price(self, config, weight_steel, weight_concrete) -> float:
        """
        Calculate the price based on material weights.
        
        Args:
            config: The configuration object
            weight_steel: Weight of steel in tons per m²
            weight_concrete: Weight of concrete in tons per m²
            
        Returns:
            Total price
        """
        price_steel = weight_steel * config.MATERIAL_PRICE.Steel
        price_concrete = weight_concrete * config.MATERIAL_PRICE.Concrete
        return price_steel + price_concrete
    
    def _calculate_cost(self, output: Any, config: MainConfig) -> Tuple[float, float]:
        """
        Calculate the cost including penalties.
        
        Args:
            output: The output data from the model
            config: The configuration object
            
        Returns:
            Tuple containing:
                - Cost: The total cost including penalties
                - sum_penalty: The sum of all penalties
        """
        weight_steel, weight_concrete = self._calculate_weight(output, config)
        price = self._calculate_price(config, weight_steel, weight_concrete)
        penalty = self._calculate_penalty(output)
        
        # Penalty weights
        element_penalty_weight = 1
        element_penalty_soil = 1
        u_total_penalty_weight = 20

        # Calculate total penalty
        sum_penalty = (element_penalty_weight * penalty.elementPenalty_Structures + 
                      element_penalty_soil * penalty.elementPenalty_Soil + 
                      u_total_penalty_weight * penalty.uTotalpenalty)
        sum_penalty *= 2
        
        # Calculate final cost with penalty
        cost = price * (1 + sum_penalty)
        
        # Convert from np.float64 to float for compatibility
        if isinstance(cost, np.float64):
            cost = float(cost)
        if isinstance(sum_penalty, np.float64):
            sum_penalty = float(sum_penalty)
            
        return cost, sum_penalty 