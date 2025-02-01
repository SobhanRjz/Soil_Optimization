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
    'contour_width': 40, # meters
    'step_phase': 2,     # meters
    'load_value': 15,    # kN/m
    'geogrid_teta': 10   # degrees
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
        'ea1': 123000
    },
    'plate': {
        'type': 'Elastic',
        'width': 2.3,
        'ea1': 2000000,
        'ei': 1667
    }
}
