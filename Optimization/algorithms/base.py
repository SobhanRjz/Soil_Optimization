import logging
import json
import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from models import InputData, ResultData, ParticleConfig
from Config.config import MainConfig
from Database.PostgreSQL import PostgreSQLDatabase


class BaseOptimizer(ABC):
    """Base class for optimization algorithms."""
    
    def __init__(self, max_iter: int = 100):
        """
        Initialize the base optimizer.
        
        Args:
            max_iter: Maximum number of iterations
        """
        # Set up logging
        self.logger = self._setup_logger()
        self.max_iter = max_iter
        
        # Initialize database connection
        self.database = PostgreSQLDatabase()
        self.database.initialize_tables()
        
        # Initialize tracking variables
        self.global_best_position = None
        self.global_best_score = float("inf")
        self.global_best_penalty = 0.05
        self.global_best_combined_score = float("inf")
        
        # Status tracking
        self.processing_queue = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up and return a logger for the optimizer."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _save_status(self, iteration_count: int, processed_count: int, 
                    max_to_process: int, initialize: bool = False) -> None:
        """
        Save the optimization status to a JSON file.
        
        Args:
            iteration_count: Current iteration number
            processed_count: Number of particles processed
            max_to_process: Total number of particles to process
            initialize: Whether this is the initial status save
        """
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
                self.logger.warning("Could not read existing status file. Creating new file.")
        
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
    
    def close_plaxis_output(self) -> bool:
        """
        Close any running PLAXIS Output processes to free up resources.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Attempting to close PLAXIS Output processes...")
        try:
            import subprocess
            import platform
            import time
            
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
            time.sleep(1)
            
            return True
        except Exception as e:
            self.logger.error(f"Error closing PLAXIS Output processes: {str(e)}")
            return False
    
    def closest_value(self, value: float, choices: List[float]) -> float:
        """
        Find the closest valid discrete value from a list of choices.
        
        Args:
            value: The target value
            choices: List of valid choices
            
        Returns:
            The closest value from the choices list
        """
        return min(choices, key=lambda x: abs(x - value))
    
    @abstractmethod
    def optimize(self):
        """Run the optimization algorithm. Must be implemented by subclasses."""
        pass
