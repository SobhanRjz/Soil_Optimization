from enum import Enum

# Mesh size options
class MeshSize(Enum):
    VERY_COARSE = 0.12
    COARSE = 0.08
    MEDIUM = 0.06 
    FINE = 0.04
    VERY_FINE = 0.03

# PLAXIS server configuration
PLAXIS_CONFIG = {
    'input': {
        'host': 'localhost',
        'port': 10000,
        'password': 'vkrAnSku2/x^$f8~'
    },
    'output': {
        'host': 'localhost', 
        'port': 10001,
        'password': 'vkrAnSku2/x^$f8~'
    }
}

# Model geometry parameters
MODEL_GEOMETRY = {
    'plate_length': 10,  # meters
    'step_phase': 2,     # meters
    'load_value': 15,    # kN/m
    'geogrid_teta': 10,   # degrees
    'mesh_size': MeshSize.MEDIUM.value  # meters
}


# Soil material properties
SOIL_PROPERTIES = {
    'model': 'HardeningSoil',
    'gamma_unsat': 20,
    'gamma_sat': 20,
    'e50_ref': 20000,
    'eoed_ref': 20000,
    'eur_ref': 60000,
    'nu': 0.3,
    'phi': 30,
    'c_ref': 15,
    'psi': 0
}

# Structural materials
STRUCTURAL_MATERIALS = {
    'geogrid': {
        'type': 'Elastic',
        'ea1': 123000, #KN/m
        'HorizentalSpace': 10, #M
        'VerticalSpace': 2 #M
    },
    'plate': {
        'type': 'Elastic',
        'UnitWeight': 2.3,  #unit Weight KN/m/m
        'ea1': 2000000, #KN/m
        'ei': 1667 #KN/m2/m

    }
}
