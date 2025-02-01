import os
import plxscripting.easy as plx
import math
import numpy as np
from shapely.geometry import LineString, box
import logging
import time
from Config.config import PLAXIS_CONFIG, MODEL_GEOMETRY, SOIL_PROPERTIES, STRUCTURAL_MATERIALS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlaxisModelInput:
    def __init__(self, host=PLAXIS_CONFIG['input']['host'], 
                 port=PLAXIS_CONFIG['input']['port'], 
                 password=PLAXIS_CONFIG['input']['password']):
        self.__host = host
        self.__port = port
        self.__password = password
        self.__s_i = None
        self.__g_i = None
        
        # Load geometry parameters from config
        self.__plate_length = MODEL_GEOMETRY['plate_length']
        self.__geogrid_VerticalSpace = STRUCTURAL_MATERIALS['geogrid']['VerticalSpace']
        self.__geogrid_length = self.__plate_length
        self.__geogrid_teta = MODEL_GEOMETRY['geogrid_teta']
        self.__step_phase = MODEL_GEOMETRY['step_phase']

        self.__load_value = MODEL_GEOMETRY['load_value']
        self.__contour_points = None
        self.phase_names = []

    def __connect(self):
        """Establish connection to PLAXIS server"""
        logger.info("Connecting to PLAXIS server...")
        self.__s_i, self.__g_i = plx.new_server(self.__host, self.__port, password=self.__password)
        
    def __create_project(self, name="MyNewProject"):
        """Create new PLAXIS project"""
        logger.info("Creating new PLAXIS project...")
        self.__s_i.new()
        self.__g_i.ModelType = "Plane strain"
        
    def __define_geometry(self):
        """Define model geometry and soil contour"""
        logger.info("Defining geometry...")
        self.__contour_points = [
            (-3 * self.__plate_length, -2.5 * self.__plate_length),  # Bottom-left corner
            (1.5 * self.__plate_length, -2.5 * self.__plate_length),   # Bottom-right corner
            (1.5 * self.__plate_length, 0),     # Top-right corner
            (-3 * self.__plate_length, 0)     # Top-left corner
        ]

        self.__soil_contour = self.__g_i.SoilContour.initializerectangular(*self.__contour_points[0], *self.__contour_points[2])
        
    def __create_soil_profile(self):
        """Create soil layers and materials"""
        logger.info("Creating soil profile...")
        borhole = self.__g_i.borehole(0)
        borhole.Head = -20
        self.__soillayer = self.__g_i.soillayer(20)
        
        # Create soil material using config
        self.__soil_material = self.__g_i.soilmat()
        self.__soil_material.Identification = "SoilMat"
        self.__soil_material.MaterialName = "Clay"
        self.__soil_material.SoilModel = SOIL_PROPERTIES['model']
        self.__soil_material.gammaUnsat = SOIL_PROPERTIES['gamma_unsat']
        self.__soil_material.gammaSat = SOIL_PROPERTIES['gamma_sat']
        self.__soil_material.E50Ref = SOIL_PROPERTIES['e50_ref']
        self.__soil_material.EOedRef = SOIL_PROPERTIES['eoed_ref']
        self.__soil_material.EURRef = SOIL_PROPERTIES['eur_ref']
        self.__soil_material.nu = SOIL_PROPERTIES['nu']
        self.__soil_material.phi = SOIL_PROPERTIES['phi']
        self.__soil_material.cRef = SOIL_PROPERTIES['c_ref']
        self.__soil_material.psi = SOIL_PROPERTIES['psi']
        
    def __create_structural_materials(self):
        """Create geogrid and plate materials"""
        logger.info("Creating structural materials...")
        # Geogrid material
        self.__geogrid = self.__g_i.geogridmat()
        self.__geogrid.MaterialType = STRUCTURAL_MATERIALS['geogrid']['type']
        self.__geogrid.Identification = "Geogrid_PHI28"
        self.__geogrid.setproperties("EA1", STRUCTURAL_MATERIALS['geogrid']['ea1'])
        
        # Plate material
        self.__plate = self.__g_i.platemat()
        self.__plate.MaterialType = STRUCTURAL_MATERIALS['plate']['type']
        self.__plate.Identification = "Plate"
        self.__plate.setproperties("w", STRUCTURAL_MATERIALS['plate']['UnitWeight'])
        self.__plate.setproperties("EA1", STRUCTURAL_MATERIALS['plate']['ea1'])
        self.__plate.setproperties("EI", STRUCTURAL_MATERIALS['plate']['ei'])
        self.__plate.setproperties("StructNu", 0.2)
        

    def __assign_materials(self):
        """Assign materials to layers"""
        logger.info("Assigning materials...")
        self.__g_i.setmaterial(self.__g_i.Soillayers[-1], self.__g_i.SoilMat)
        
    def __create_structures(self):
        """Create structural elements"""
        logger.info("Creating structures...")
        self.__g_i.gotostructures()
        
        # Create shotcrete line
        shotcrete_line = self.__g_i.plate((0, 0), (0, -self.__plate_length))
        shotcrete_line[2].setproperties("Plate.Material", self.__plate)
        

        # Create horizontal lines
        for y in range(-self.__step_phase, - self.__plate_length - 1, -self.__step_phase):
            self.__g_i.line((0, y), (self.__contour_points[2][0], y))
            
        # Create geogrids
        geogrid_y_offset = self.__geogrid_length * math.tan(math.radians(self.__geogrid_teta))

        for y in np.arange(-1.5, -self.__plate_length, -self.__geogrid_VerticalSpace):
            geogrid = self.__g_i.geogrid((0, y), (-self.__geogrid_length, y - geogrid_y_offset))
            geogrid[2].setproperties("Geogrid.Material", self.__geogrid)

        # Create Line Loads
        load = self.__g_i.lineload((0,0), (-self.__plate_length * 1.5 ,0))
        load[3].qy_start = -self.__load_value # Kn / m
        
    def __automesh(self):
        logger.info("Generating mesh...")
        self.__g_i.gotomesh()
        mesh = self.__g_i.mesh(MODEL_GEOMETRY['mesh_size']) # medium

    def __create_phase(self):
        logger.info("Creating phases...")
        self.__g_i.gotostages()


        #Activate line load for IntialPhases
        line_loads = self.__g_i.LineLoads
        initial_phase = self.__g_i.Phases[0]
        self.__g_i.activate(line_loads , initial_phase)

        soils = self.__g_i.Soils
        lines = self.__g_i.Lines
        geogrid_lines = self.__g_i.GeoGrids
        # Filter lines that are associated with plates
        lines = self.__g_i.Lines

        # Filter lines that are associated with geogrid materials
        geogrid_lines = [line for line in lines if hasattr(line, "Geogrid")]

        # Create map of soil names to soil objects
        soil_map = {soil.Name.echo().split(".")[0]: soil for soil in soils}
        line_map = {line.Name.echo().split(".")[0]: line for line in lines}
        Geo_map = {Geo.Name.echo().split(".")[0]: Geo for Geo in geogrid_lines}
        firstIndexGepMap = min(int(name.split('_')[1]) for name in Geo_map.keys())

        for indexS in range(1, int(abs(self.__plate_length / 2)) + 1):
            #create New Phase
            new_phase = self.__g_i.phase(self.__g_i.Phases[-1])
            self.phase_names.append(new_phase.Identification.value)

            # Deactivate Soils
            soilName = f"Soil_1_{indexS}"
            Soil = soil_map[soilName]
            xMinSoil, yMinSoil, xMaxSoil, yMaxSoil = Soil.Parent.BoundingBox.xMin, Soil.Parent.BoundingBox.yMin, Soil.Parent.BoundingBox.xMax, Soil.Parent.BoundingBox.yMax
            soil_box = box(xMinSoil.value, yMinSoil.value, xMaxSoil.value, yMaxSoil.value)
            
            self.__g_i.deactivate(Soil, new_phase)

            # Activate ShotCrete
            found2Line = 0
            for line_name, line in line_map.items():
                if hasattr(line, "Plate"):
                    x1, y1 = line.Geometry.x1.value, line.Geometry.y1.value
                    x2, y2 = line.Geometry.x2.value, line.Geometry.y2.value
                    plate_line = LineString([(x1, y1), (x2, y2)])

                    if (plate_line.within(soil_box) or plate_line.touches(soil_box)) and (y1 >= yMinSoil.value and y2 >= yMinSoil.value):
                        self.__g_i.activate(line, new_phase)

            # Activate Nailing
            for GeoName, Nail in Geo_map.items():
                x1, y1 = Nail.Geometry.x1.value, Nail.Geometry.y1.value
                x2, y2 = Nail.Geometry.x2.value, Nail.Geometry.y2.value
                nail_line = LineString([(x1, y1), (x2, y2)])

                if nail_line.intersects(soil_box):
                    self.__g_i.activate(Nail, new_phase)

    def __run(self):
        logger.info("Running calculation...")
        self.__g_i.calculate()

    def __save(self, project_name="my_plaxis_project.p2dx"):
        """Save project and close connection"""
        logger.info("Saving and closing project...")
        save_path = os.path.join(os.getcwd(), project_name)
        self.__g_i.save(save_path)
        #self.__g_i.close()

    def Create_Model(self):
        start_time = time.time()
        logger.info("Starting PLAXIS model creation...")
        
        # Create and run model
        self.__connect()
        self.__create_project()
        self.__define_geometry()
        self.__create_soil_profile()
        self.__create_structural_materials()
        self.__assign_materials()
        self.__create_structures()
        self.__automesh()
        self.__create_phase()
        self.__run()
        self.__save()
        #add view InitialPhase
        self.__g_i.gotoviews()
        self.__g_i.view(self.__g_i.Phases[0])
        logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
