import numpy as np
import random
import sys
import os
from typing import List
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Database.PostgreSQL import PostgreSQLDatabase
from models import InputData, ResultData, ParticleConfig
import PLAXIS.Input_PLAXIS as InputModel
import PLAXIS.Output as OutputModel
from Config.config import MainConfig
import math
from dataclasses import dataclass

@dataclass
class Penalty:
    elementPenalty_Structures: float
    elementPenalty_Soil: float
    uTotalpenalty: float

class HybridPSOHS:
    def __init__(self, population_size=10, harmony_memory_size=10, max_iter=100,
                 inertia_weight=0.7, cognitive_weight=1.5, social_weight=2.0,
                 harmony_consideration_rate=0.9, pitch_adjustment_rate=0.3):
        # Define parameter choices
        # Set random seed for reproducibility
        random.seed(42)
        self.rebar_dia_List = [16, 18, 20, 22, 25, 28, 30]
        self.nail_len_List = list(range(3, 30, 3))  # 1 to 40 meters, step 5
        self.nail_teta_List = [5, 10, 15, 20]  # Degrees
        self.nail_h_space_List = [1.2, 1.5, 1.8, 2, 2.2, 2.5]  # cm
        self.nail_v_space_List = [1.2, 1.5, 1.8, 2, 2.2, 2.5]  # cm
        
        self.max_iter = max_iter
        # PSO Parameters
        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        # HS Parameters
        self.harmony_memory_size = harmony_memory_size
        self.harmony_consideration_rate = harmony_consideration_rate
        self.pitch_adjustment_rate = pitch_adjustment_rate
        
        # Initialize PSO particles
        
        self.global_best_position = None
        self.global_best_score = float("inf")

        # Initialize HS harmonies
        self.hs_best_solution = None
        self.hs_best_score = float("inf")

        #connect to the DataBase
        self.DataBase = PostgreSQLDatabase()
        self.DataBase.initialize_tables()

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
        """Initialize a particle with a random valid combination of parameters."""

       
        
        position = InputData(
            rebar_dia=random.choice(self.rebar_dia_List),
            nail_teta=random.choice(self.nail_teta_List),
            nail_h_space=random.choice(self.nail_h_space_List),
            nail_v_space=random.choice(self.nail_v_space_List),
            Algo_Name=f"Random {algoName}"
        )

        
        position.nail_len = self._generate_nail_lengths(position)
        
        return ParticleConfig(
            position=position,
            velocity=np.zeros(5),
            best_position=None,
            best_score=float("inf")
        )

    def _generate_nail_lengths(self, position):
        """Generate a list of nail lengths based on rebar diameter and number of nails."""
        main_config = MainConfig()
        min_len = self._getBestLength(position.rebar_dia)
        num_nails = int(((main_config.MODEL_GEOMETRY.plate_length - 1.5) // position.nail_v_space) + 1)
        nail_len_list = [self.closest_value(random.uniform(min_len, max(self.nail_len_List)), self.nail_len_List) 
                        for _ in range(num_nails)]
        nail_len_list.sort(reverse=True)

        if nail_len_list[0] - nail_len_list[-1] > 20:
            avg_len = sum(nail_len_list) / len(nail_len_list)
            nail_len_list = [self.closest_value(random.uniform(avg_len - 10, avg_len + 10), self.nail_len_List) 
                            for _ in range(len(nail_len_list))]
            nail_len_list.sort(reverse=True)

        return nail_len_list


    def evaluate_fitness(self, inputData : InputData):
        """Define the objective function (modify for real-world constraints)."""
        cost = (inputData.rebar_dia * 10) + (inputData.nail_len * 15) + (abs(inputData.nail_teta - 60) * 5) + (inputData.nail_h_space * 2) + (inputData.nail_v_space * 1.5)
        return cost
    def update_velocity_and_position(self, particle, algoName = ""):
        """Update particle velocity and position in PSO."""
        r1, r2 = np.random.rand(2)
        
        # Handle nail_len list specially
        current_nail_len = np.mean(particle.position.nail_len)
        
        current_position = np.array([
            float(particle.position.rebar_dia),
            float(current_nail_len), 
            float(particle.position.nail_teta),
            float(particle.position.nail_h_space),
            float(particle.position.nail_v_space)
        ])

        if particle.best_position:
            best_nail_len = np.mean(particle.best_position.nail_len)
            best_position = np.array([
                float(particle.best_position.rebar_dia),
                float(best_nail_len),
                float(particle.best_position.nail_teta),
                float(particle.best_position.nail_h_space),
                float(particle.best_position.nail_v_space)
            ])
        else:
            best_position = current_position

        if self.global_best_position:
            global_nail_len = np.mean(self.global_best_position.nail_len)
            global_best_position = np.array([
                float(self.global_best_position.rebar_dia),
                float(global_nail_len),
                float(self.global_best_position.nail_teta),
                float(self.global_best_position.nail_h_space),
                float(self.global_best_position.nail_v_space)
            ])
        else:
            global_best_position = current_position

        velocity = (
            self.inertia_weight * particle.velocity
            + self.cognitive_weight * r1 * (best_position - current_position)
            + self.social_weight * r2 * (global_best_position - current_position)
        )

        particle.velocity = velocity
        new_position = current_position + velocity
        

        particle.position = InputData(
            rebar_dia=self.closest_value(new_position[0], self.rebar_dia_List),
            nail_len=[],
            nail_teta=self.closest_value(new_position[2], self.nail_teta_List),
            nail_h_space=self.closest_value(new_position[3], self.nail_h_space_List),
            nail_v_space=self.closest_value(new_position[4], self.nail_v_space_List),
            Algo_Name=algoName
        )
                # Generate nail lengths based on vertical spacing
        nail_len_list = self._generate_nail_lengths(particle.position)
        particle.position.nail_len = nail_len_list

        particle.best_position = None
        particle.best_score = float("inf")
        particle.Cost = float("inf")
        return particle

    def generate_new_harmony(self):
        """Generate a new harmony in HS."""
        new_harmony = []
                        # Get position from random harmony memory particle
        selected_particle = random.choice(self.harmony_memory)
        param_values = [
            selected_particle.position.rebar_dia,
            np.mean(selected_particle.position.nail_len),  # Take mean of nail_len list
            selected_particle.position.nail_teta,
            selected_particle.position.nail_h_space,
            selected_particle.position.nail_v_space
        ]
        
        for i, choices in enumerate([
            self.rebar_dia_List, self.nail_len_List, self.nail_teta_List, self.nail_h_space_List, self.nail_v_space_List
        ]):
            if random.random() < self.harmony_consideration_rate:
                param_value = param_values[i]
                if random.random() < self.pitch_adjustment_rate:
                    # Adjust the parameter value
                    param_value = self.closest_value(param_value + random.uniform(-1, 1), choices)
                new_harmony.append(param_value)
            else:
                new_harmony.append(random.choice(choices))
                
        particle = ParticleConfig()
        particle.position = InputData(
            rebar_dia=new_harmony[0],
            nail_len=[], # Initialize empty nail_len list
            nail_teta=new_harmony[2],
            nail_h_space=new_harmony[3],
            nail_v_space=new_harmony[4],
            Algo_Name="HS"
        )
        
        # Generate nail lengths based on vertical spacing
        nail_len_list = self._generate_nail_lengths(particle.position)
        particle.position.nail_len = nail_len_list

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
        """Calculate the penalty of the particle."""
        max_ratio_Structures = 1.0
        max_ratio_Soil = 1.0
        max_uTotal = 5

        uTotalpenalty = 0
        elementPenalty_Structures = 0
        elementPenalty_Soil = 0
        for data in output.ElementData.elementForceData.values():
            if data.max_ratio_Structures > max_ratio_Structures:
                elementPenalty_Structures += data.max_ratio_Structures - max_ratio_Structures
            if data.max_ratio_Soil > max_ratio_Soil:
                elementPenalty_Soil += max_ratio_Soil - data.max_ratio_Soil
        for data in output.ElementData.displacementData:
            if data.utotal > max_uTotal:
                uTotalpenalty +=  max_uTotal - data.utotal

        return Penalty(elementPenalty_Structures, elementPenalty_Soil, uTotalpenalty)

    def _calculate_Price(self, config, weightSteel, weightConcrete):
        """Calculate the price of the particle."""
        PriceSteel = weightSteel * config.MATERIAL_PRICE.Steel
        PriceConcrete = weightConcrete * config.MATERIAL_PRICE.Concrete
        return PriceSteel + PriceConcrete
    
    def _calculate_Cost(self, output, config):
        """Calculate the cost of the particle."""
        weightSteel, weightConcrete = self._calculate_Weight(output, config)
        Price = self._calculate_Price(config, weightSteel, weightConcrete)
        Penalty = self._calculate_penalty(output)
        sum_penalty = Penalty.elementPenalty_Structures + Penalty.elementPenalty_Soil + Penalty.uTotalpenalty
        Cost = Price * (1 + sum_penalty)
        return Cost, sum_penalty

    def optimize(self):
        """Run the hybrid PSO-HS optimization."""
        config = MainConfig()
        # Initialize particle population
        self.particles: List[ParticleConfig] = []
        self.harmony_memory: List[ParticleConfig] = []
        model = InputModel.PlaxisModelInput()

        for _ in range(self.population_size):
            particle = self.initialize_particle("PSO")
            
            # Store initial particle positions in database
            input_id, input_hash = self.DataBase.insert_input(particle.position, status = 0)
            particle.input_hash = input_hash
            self.particles.append(particle)

        for _ in range(self.population_size):
            particle = self.initialize_particle("HS")
            
            # Store initial particle positions in database
            input_id, input_hash = self.DataBase.insert_input(particle.position, status = 0)
            particle.input_hash = input_hash
            self.harmony_memory.append(particle)

        
        for _ in range(self.max_iter):
            GlobalRowNum = 0
            while True:
                # Insert initial results into database
                input_hash, input_Data = self.DataBase._get_input_Data()
                config.STRUCTURAL_MATERIALS.geogrid.nail_length = input_Data.nail_len
                config.STRUCTURAL_MATERIALS.geogrid.RebarDiameter = input_Data.rebar_dia
                config.STRUCTURAL_MATERIALS.geogrid.HorizentalSpace = input_Data.nail_h_space 
                config.STRUCTURAL_MATERIALS.geogrid.VerticalSpace = input_Data.nail_v_space
                config.MODEL_GEOMETRY.geogrid_teta = input_Data.nail_teta
                model.config = config
                model.Create_Model()
                output = OutputModel.PlaxisModelOutput()
                output.GetOutput()
                Cost, Penalty = self._calculate_Cost(output, config)

                result_data = ResultData(
                                Total_Displacement = round(max(d.utotal for d in output.ElementData.displacementData), 2),
                                Total_Displacement_Allow = 5,
                                max_Structure_Ratio = round(max(d.max_ratio_Structures for d in output.ElementData.elementForceData.values()), 2),
                                max_Soil_Ratio = round(max(d.max_ratio_Soil for d in output.ElementData.elementForceData.values()), 2),
                                Penalty = round(Penalty, 2),
                                Cost= round(Cost, 2))

                self.DataBase.insert_result(input_hash, status = 2, result_data = result_data)
                
                # result_data = ResultData(   
                #                             Total_Distance= 2,
                #                             Total_Distance_Allow= 10,
                #                             Structure_Force= 30,
                #                             Structure_Force_Allow= 40,
                #                             Soil_Force= 60,
                #                             Soil_Force_Allow= 70,
                #                             Penalty= 2,
                #                             Cost= 6000)
                

                # self.DataBase.insert_result(self.harmony_memory[0].input_hash, status = 2, result_data = result_data)

                rowNumber = self.DataBase._get_result_NonCheck()
                if rowNumber != 0:
                    inputHash = self.DataBase._get_lastHash_Result()
                    result: ResultData = self.DataBase._get_result_baseHash(inputHash)
                    AlgoName =  self.DataBase._get_algoNameBaseHash(inputHash)
                    # Find particle index in list
                    
                    

                        

                    if AlgoName == "PSO" or  AlgoName == "Random PSO":
                        particle_idx = next((idx for idx, p in enumerate(self.particles) if p.input_hash == inputHash), None)
                        if particle_idx is not None:
                            # Update particle in list
                            self.particles[particle_idx].Cost = Cost
                            if result.Cost < self.particles[particle_idx].best_score:
                                self.particles[particle_idx].best_score = result.Cost
                                self.particles[particle_idx].best_position = self.particles[particle_idx].position

                            if result.Cost < self.global_best_score:
                                self.global_best_score = result.Cost
                                self.global_best_position = self.particles[particle_idx].position

                        NewParticle = self.update_velocity_and_position(self.particles[particle_idx], "PSO")
                        input_id, input_hash = self.DataBase.insert_input(NewParticle.position, status = 0)
                        NewParticle.input_hash = input_hash
                        self.particles.append(NewParticle)
                        self.DataBase._update_result_status(input_hash=inputHash, status=3)
                        
                    if AlgoName == "HS" or AlgoName == "Random HS":
                        particle_idx = next((idx for idx, p in enumerate(self.harmony_memory) if p.input_hash == inputHash), None)
                        self.harmony_memory[particle_idx].Cost = Cost
                        
                        # HS update
                        new_harmony = self.generate_new_harmony()
                        input_id, new_input_hash = self.DataBase.insert_input(new_harmony.position, status = 0)
                        new_harmony.input_hash = new_input_hash


                        #new_score = self.evaluate_fitness(new_harmony)
                        if result.Cost < self.hs_best_score:
                            self.hs_best_score = result.Cost
                            self.hs_best_solution = new_harmony
                        
                        if result.Cost < self.global_best_score:
                            self.global_best_score = result.Cost
                            self.global_best_position = self.harmony_memory[particle_idx].position

                        # Sort harmony memory by cost in descending order and keep only harmony_memory_size items
                        self.harmony_memory = sorted(self.harmony_memory, key=lambda x: x.Cost)[:self.harmony_memory_size - 1]
                        self.harmony_memory.append(new_harmony)
                        self.DataBase._update_result_status(input_hash=inputHash, status=3)

                    break


# Run the hybrid optimizer
hybrid_optimizer = HybridPSOHS(max_iter=1000)
best_solution, best_score = hybrid_optimizer.optimize()
print(f"Hybrid PSO-HS Best Solution: {best_solution}, Score: {best_score}")
