from enum import Enum

# Mesh size options
class MeshSize(Enum):
    VERY_COARSE = 0.12
    COARSE = 0.08
    MEDIUM = 0.06 
    FINE = 0.04
    VERY_FINE = 0.03

class PlaxisConfig:
    def __init__(self):
        self.input = {
            'host': 'localhost',
            'port': 10000,
            'password': 'vkrAnSku2/x^$f8~'
        }
        self.output = {
            'host': 'localhost',
            'port': 10001, 
            'password': 'vkrAnSku2/x^$f8~'
        }

class ModelGeometry:
    def __init__(self):
        self.plate_length = 10  # meters
        self.step_phase = 2     # meters 
        self.load_value = 10    # kN/m
        self.geogrid_teta = 10  # degrees
        self.mesh_size = MeshSize.VERY_COARSE.value  # meters

class SoilProperties:
    def __init__(self):
        self.model = 'HardeningSoil'
        self.gamma_unsat = 20   # kN/m³
        self.gamma_sat = 20     # kN/m³
        self.e50_ref = 20000    # kN/m²
        self.eoed_ref = 20000   # kN/m²
        self.eur_ref = 60000    # kN/m²
        self.nu = 0.3          # Poisson's ratio
        self.phi = 30          # Friction angle (degrees)
        self.c_ref = 15        # Cohesion (kN/m²)
        self.psi = 0           # Dilatancy angle (degrees)

class GeogridMaterial:
    def __init__(self):
        self.type = 'Elastic'
        self.E = 2.1 * 10**8  # KN/m2
        self.Fy = 400  # MPA = 400,000 KN/m² 
        self.RebarDiameter = 28  # mm
        self.ea1 = 123000  # KN/m
        self.HorizentalSpace = 1  # M
        self.VerticalSpace = 2  # M
        self.AlphaBond = 300
        self.CodeRatio = 2.0
        self.nail_length = [10, 10, 10, 10, 10]  # m
        self.HollowDiameter = 0.12  # M

class PlateMaterial:
    def __init__(self):
        self.type = 'Elastic'
        self.UnitWeight = 2.3    # kN/m/m
        self.ea1 = 2000000      # Axial stiffness (kN/m)
        self.ei = 1667          # Bending stiffness (kN·m²/m)

class StructuralMaterials:
    def __init__(self):
        self.geogrid = GeogridMaterial()
        self.plate = PlateMaterial()

class MaterialPrice:
    def __init__(self):
        self.Concrete = 1500    # Price per m³
        self.Steel = 30000      # Price per ton

# Singleton pattern
class MainConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MainConfig, cls).__new__(cls)
            cls._instance.PLAXIS_CONFIG = PlaxisConfig()
            cls._instance.MODEL_GEOMETRY = ModelGeometry()
            cls._instance.SOIL_PROPERTIES = SoilProperties()
            cls._instance.STRUCTURAL_MATERIALS = StructuralMaterials()
            cls._instance.MATERIAL_PRICE = MaterialPrice()
            
        return cls._instance

    def update(self, key, value):
        """Update configuration dynamically"""
        # Split key by dots to handle nested attributes
        keys = key.split('.')
        
        # Start with the current instance
        current = self
        
        # Navigate through nested attributes except the last one
        for k in keys[:-1]:
            if hasattr(current, k):
                current = getattr(current, k)
            else:
                raise KeyError(f"'{k}' not found in configuration")
        
        # Set the final attribute
        if hasattr(current, keys[-1]):
            setattr(current, keys[-1], value)
        else:
            raise KeyError(f"'{keys[-1]}' not found in configuration")

    def get(self, key):
        """Get a configuration value"""
        return getattr(self, key, None)
