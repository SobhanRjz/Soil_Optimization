import numpy as np
import random
import sys
import os
from typing import List
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Database.PostgreSQL import PostgreSQLDatabase
from models import InputData, ResultData, ParticleConfig
import PLAXIS.Input as InputModel
import PLAXIS.Output as OutputModel
class HybridPSOHS:
    def __init__(self, population_size=30, harmony_memory_size=30, max_iter=100,
                 inertia_weight=0.7, cognitive_weight=1.5, social_weight=2.0,
                 harmony_consideration_rate=0.9, pitch_adjustment_rate=0.3):
        # Define parameter choices
        # Set random seed for reproducibility
        random.seed(42)
        self.rebar_dia_List = [16, 18, 20, 22, 25, 28, 30]
        self.nail_len_List = list(range(1, 40, 5))  # 1 to 40 meters, step 5
        self.nail_teta_List = [5, 10, 15, 20]  # Degrees
        self.nail_h_space_List = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]  # cm
        self.nail_v_space_List = [100, 150, 200, 250, 300, 350, 400, 450, 500]  # cm
        
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


    def initialize_particle(self, algoName):
        """Initialize a particle with a random valid combination of parameters."""


        
        position = InputData(
            rebar_dia=random.choice(self.rebar_dia_List),
            nail_len=random.choice(self.nail_len_List), 
            nail_teta=random.choice(self.nail_teta_List),
            nail_h_space=random.choice(self.nail_h_space_List),
            nail_v_space=random.choice(self.nail_v_space_List),
            Algo_Name=f"Random {algoName}"
        )
        return ParticleConfig(
            position=position,
            velocity=np.zeros(5),
            best_position=None,
            best_score=float("inf")
        )


    def evaluate_fitness(self, inputData : InputData):
        """Define the objective function (modify for real-world constraints)."""
        cost = (inputData.rebar_dia * 10) + (inputData.nail_len * 15) + (abs(inputData.nail_teta - 60) * 5) + (inputData.nail_h_space * 2) + (inputData.nail_v_space * 1.5)
        return cost

    def update_velocity_and_position(self, particle, algoName = ""):
        """Update particle velocity and position in PSO."""
        r1, r2 = np.random.rand(2)
        current_position = np.array([
            particle.position.rebar_dia,
            particle.position.nail_len,
            particle.position.nail_teta, 
            particle.position.nail_h_space,
            particle.position.nail_v_space
        ])
        best_position = np.array([
            particle.best_position.rebar_dia,
            particle.best_position.nail_len,
            particle.best_position.nail_teta,
            particle.best_position.nail_h_space,
            particle.best_position.nail_v_space
        ]) if particle.best_position else current_position

        global_best_position = np.array([
            self.global_best_position.rebar_dia,
            self.global_best_position.nail_len, 
            self.global_best_position.nail_teta,
            self.global_best_position.nail_h_space,
            self.global_best_position.nail_v_space
        ]) if self.global_best_position else current_position

        
        velocity = (
            self.inertia_weight * particle.velocity
            + self.cognitive_weight * r1 * (best_position - current_position)
            + self.social_weight * r2 * (global_best_position - current_position)
        )

        particle.velocity = velocity
        new_position = current_position + velocity

        particle.position = InputData(
            rebar_dia=self.closest_value(new_position[0], self.rebar_dia_List),
            nail_len=self.closest_value(new_position[1], self.nail_len_List),
            nail_teta=self.closest_value(new_position[2], self.nail_teta_List),
            nail_h_space=self.closest_value(new_position[3], self.nail_h_space_List),
            nail_v_space=self.closest_value(new_position[4], self.nail_v_space_List),
            Algo_Name=algoName
        )
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
            selected_particle.position.nail_len,
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
            nail_len=new_harmony[1], 
            nail_teta=new_harmony[2],
            nail_h_space=new_harmony[3],
            nail_v_space=new_harmony[4],
            Algo_Name="HS"
        )
        particle.best_position = None
        particle.best_score = float("inf")
        particle.Cost = float("inf")
        return particle

    def closest_value(self, value, choices):
        """Find the closest valid discrete value."""
        return min(choices, key=lambda x: abs(x - value))

    def optimize(self):
        """Run the hybrid PSO-HS optimization."""
        
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
                _ , input_Data = self.DataBase._get_input_Data()
                model.MODEL_GEOMETRY.plate_length = input_Data.nail_len
                model.STRUCTURAL_MATERIALS.geogrid.RebarDiameter = input_Data.rebar_dia
                model.STRUCTURAL_MATERIALS.geogrid.HorizentalSpace = input_Data.nail_h_space / 100
                model.STRUCTURAL_MATERIALS.geogrid.VerticalSpace = input_Data.nail_v_space / 100
                model.STRUCTURAL_MATERIALS.geogrid.teta = input_Data.nail_teta

                model.Create_Model()
                output = OutputModel.PlaxisModelOutput()
                output.GetOutput()
                Total_Distance = max(data.utotal for data in output._output_data)
                resultOutput = output._Check_Total_Displacement()

                result_data = ResultData(   
                                Total_Distance= Total_Distance,
                                Total_Distance_Allow= resultOutput,
                                Penalty= 1,
                                Cost= 5000)
                

                # self.DataBase.insert_result(self.particles[0].input_hash, status = 2, result_data = result_data)
                
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
                            self.particles[particle_idx].Cost = self.evaluate_fitness(self.particles[particle_idx].position)
                            if result.Cost < self.particles[particle_idx].best_score:
                                self.particles[particle_idx].best_score = result.Cost
                                self.particles[particle_idx].best_position = self.particles[particle_idx].position

                            if result.Cost < self.global_best_score:
                                self.global_best_score = result.Cost
                                self.global_best_position = self.particles[particle_idx].position

                        NewParticle = self.update_velocity_and_position(self.particles[particle_idx], "PSO")
                        input_id, input_hash = self.DataBase.insert_input(NewParticle.position)
                        NewParticle.input_hash = input_hash
                        self.particles.append(NewParticle)
                        self.DataBase._update_result_status(input_hash=inputHash, status=2)
                        
                    if AlgoName == "HS" or AlgoName == "Random HS":
                        particle_idx = next((idx for idx, p in enumerate(self.harmony_memory) if p.input_hash == inputHash), None)
                        self.harmony_memory[particle_idx].Cost = self.evaluate_fitness(self.particles[particle_idx].position)
                        
                        # HS update
                        new_harmony = self.generate_new_harmony()
                        input_id, new_input_hash = self.DataBase.insert_input(new_harmony.position)
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
                        self.DataBase._update_result_status(input_hash=inputHash, status=2)

                    break


# Run the hybrid optimizer
hybrid_optimizer = HybridPSOHS(population_size=30, harmony_memory_size=30, max_iter=50)
best_solution, best_score = hybrid_optimizer.optimize()
print(f"Hybrid PSO-HS Best Solution: {best_solution}, Score: {best_score}")
