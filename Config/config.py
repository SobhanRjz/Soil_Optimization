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
        self.load_value = 15    # kN/m
        self.geogrid_teta = 10  # degrees
        self.mesh_size = MeshSize.VERY_COARSE.value  # meters

class SoilProperties:
    def __init__(self):
        self.model = 'HardeningSoil'
        self.gamma_unsat = 20
        self.gamma_sat = 20
        self.e50_ref = 20000
        self.eoed_ref = 20000
        self.eur_ref = 60000
        self.nu = 0.3
        self.phi = 30
        self.c_ref = 15
        self.psi = 0

class GeogridMaterial:
    def __init__(self):
        self.type = 'Elastic'
        self.E = 2.1 * 10**8  # KN/m2
        self.Fy = 0.2353596  # KN/mm2
        self.RebarDiameter = 28  # mm
        self.ea1 = 123000  # KN/m
        self.HorizentalSpace = 1  # M
        self.VerticalSpace = 2  # M
        self.AlphaBond = 300
        self.CodeRatio = 2.0
        self.nail_length = 10  # m

class PlateMaterial:
    def __init__(self):
        self.type = 'Elastic'
        self.UnitWeight = 2.3  # unit Weight KN/m/m
        self.ea1 = 2000000  # KN/m
        self.ei = 1667  # KN/m2/m

class StructuralMaterials:
    def __init__(self):
        self.geogrid = GeogridMaterial()
        self.plate = PlateMaterial()
class MaterialPrice:
    def __init__(self):
        self.Concrete = 1500
        self.Steel = 30000

# Create instances
PLAXIS_CONFIG = PlaxisConfig()
MODEL_GEOMETRY = ModelGeometry()
SOIL_PROPERTIES = SoilProperties()
STRUCTURAL_MATERIALS = StructuralMaterials()
MATERIAL_PRICE = MaterialPrice()
