# Soil Parameter Optimization Framework

## Overview
A sophisticated Python framework for optimizing soil parameters through integration with PLAXIS geotechnical engineering software. This project implements advanced optimization algorithms, particularly Particle Swarm Optimization (PSO), to determine optimal soil parameters that match observed field measurements.

## Key Features

### PLAXIS Integration
- Automated input generation and parameter modification for PLAXIS models
- Structured output processing and analysis
- Seamless workflow between optimization algorithms and geotechnical simulations

### Optimization Algorithms
- Particle Swarm Optimization (PSO) implementation with customizable parameters
- Modular algorithm architecture for easy extension
- Intelligent particle generation with constraint handling
- Comprehensive evaluation metrics for optimization performance

### Database Integration
- PostgreSQL database for persistent storage of optimization results
- Efficient tracking of optimization runs and parameter evolution
- Query capabilities for historical optimization data

### Server Architecture
- Dedicated server implementation for running optimization processes
- Configurable server settings for different deployment environments
- Testing infrastructure for validation

## Project Structure

### Core Modules

#### PLAXIS Interface
- `PLAXIS/Input_PLAXIS.py`: Handles the generation and modification of PLAXIS input models
- `PLAXIS/Output.py`: Processes simulation results from PLAXIS for evaluation

#### Optimization Engine
- `Optimization/algorithms/base.py`: Base classes for all optimization algorithms
- `Optimization/algorithms/pso.py`: Implementation of Particle Swarm Optimization
- `Optimization/algorithms/particle_generator.py`: Generates and manages particles for PSO
- `Optimization/evaluation/model_evaluator.py`: Evaluates model performance against target data

#### Data Management
- `Database/PostgreSQL.py`: Handles database connections and operations
- `Config/config.py`: Centralized configuration management

#### Server Components
- `startServer.py`: Entry point for starting the optimization server
- `testServer.py`: Testing utilities for server validation



### Workflow
1. Define soil parameters to optimize and their bounds
2. Configure the optimization algorithm parameters
3. Start the optimization process
4. Monitor convergence through database queries
5. Extract optimal parameters once convergence is achieved

## Technical Details

### Particle Swarm Optimization
The implementation follows standard PSO principles with:
- Inertia weight for balancing exploration and exploitation
- Cognitive and social parameters for particle movement
- Velocity clamping to prevent excessive parameter jumps
- Constraint handling for geotechnically valid parameters

### Performance Optimization
- Vectorized operations using NumPy for computational efficiency
- Parallel processing for multiple PLAXIS simulations
- Efficient database operations with connection pooling

## Requirements
- Python 3.7+
- PLAXIS (with appropriate licensing)
- PostgreSQL 12+
- NumPy, pandas, psycopg2, and other dependencies listed in requirements.txt
