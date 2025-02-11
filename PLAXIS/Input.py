import os
import plxscripting.easy as plx
import math
import numpy as np
from shapely.geometry import LineString, box
import logging
import time
from Config.config import PLAXIS_CONFIG, MODEL_GEOMETRY, SOIL_PROPERTIES, STRUCTURAL_MATERIALS, MATERIAL_PRICE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlaxisModelInput:
    def __init__(self):
        

        self.PLAXIS_CONFIG = PLAXIS_CONFIG
        self.MODEL_GEOMETRY = MODEL_GEOMETRY
        self.SOIL_PROPERTIES = SOIL_PROPERTIES
        self.STRUCTURAL_MATERIALS = STRUCTURAL_MATERIALS
        self.MATERIAL_PRICE = MATERIAL_PRICE

        self.__host = self.PLAXIS_CONFIG.input['host']
        self.__port = self.PLAXIS_CONFIG.input['port']
        self.__password = self.PLAXIS_CONFIG.input['password']
        self.__s_i = None

        self.__g_i = None
        

        # Load geometry parameters from config
        self.__plate_length = self.MODEL_GEOMETRY.plate_length
        self.__geogrid_VerticalSpace = self.STRUCTURAL_MATERIALS.geogrid.VerticalSpace
        self.__geogrid_length = self.STRUCTURAL_MATERIALS.geogrid.nail_length
        self.__geogrid_teta = self.MODEL_GEOMETRY.geogrid_teta
        self.__step_phase = self.MODEL_GEOMETRY.step_phase


        self.__load_value = self.MODEL_GEOMETRY.load_value
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
        self.__soil_material.SoilModel = self.SOIL_PROPERTIES.model
        self.__soil_material.gammaUnsat = self.SOIL_PROPERTIES.gamma_unsat
        self.__soil_material.gammaSat = self.SOIL_PROPERTIES.gamma_sat
        self.__soil_material.E50Ref = self.SOIL_PROPERTIES.e50_ref

        self.__soil_material.EOedRef = self.SOIL_PROPERTIES.eoed_ref
        self.__soil_material.EURRef = self.SOIL_PROPERTIES.eur_ref
        self.__soil_material.nu = self.SOIL_PROPERTIES.nu
        self.__soil_material.phi = self.SOIL_PROPERTIES.phi
        self.__soil_material.cRef = self.SOIL_PROPERTIES.c_ref

        self.__soil_material.psi = self.SOIL_PROPERTIES.psi
        

    def __create_structural_materials(self):
        """Create geogrid and plate materials"""
        logger.info("Creating structural materials...")
        # Geogrid material
        E = self.STRUCTURAL_MATERIALS.geogrid.E
        RebarDiameter = self.STRUCTURAL_MATERIALS.geogrid.RebarDiameter

        EA1 = E * (math.pi * (RebarDiameter / 1000) ** 2 / 4) / self.STRUCTURAL_MATERIALS.geogrid.HorizentalSpace



        self.__geogrid = self.__g_i.geogridmat()
        self.__geogrid.MaterialType = self.STRUCTURAL_MATERIALS.geogrid.type
        self.__geogrid.Identification = f"Geogrid_PHI:{RebarDiameter}_HSpace:{self.STRUCTURAL_MATERIALS.geogrid.HorizentalSpace}"
        self.__geogrid.setproperties("EA1", EA1)
        


        # Plate material
        self.__plate = self.__g_i.platemat()
        self.__plate.MaterialType = self.STRUCTURAL_MATERIALS.plate.type
        self.__plate.Identification = "Plate"
        self.__plate.setproperties("w", self.STRUCTURAL_MATERIALS.plate.UnitWeight)
        self.__plate.setproperties("EA1", self.STRUCTURAL_MATERIALS.plate.ea1)
        self.__plate.setproperties("EI", self.STRUCTURAL_MATERIALS.plate.ei)

        self.__plate.setproperties("StructNu", 0.2)
        

    def __assign_materials(self):
        """Assign materials to layers"""
        logger.info("Assigning materials...")
        self.__g_i.setmaterial(self.__g_i.Soillayers[-1], self.__g_i.SoilMat)
        
    def __create_structures(self):
        """Create structural elements"""
        logger.info("Creating structures...")
        #self.__g_i.gotostructures()
        
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
        mesh = self.__g_i.mesh(self.MODEL_GEOMETRY.mesh_size) # medium


    def __create_phase(self):
        logger.info("Creating phases...")
        self.__g_i.gotostages()

        # Get initial references to avoid repeated lookups
        initial_phase = self.__g_i.Phases[0]
        line_loads = self.__g_i.LineLoads
        soils = self.__g_i.Soils
        lines = self.__g_i.Lines

        # Activate line loads in initial phase
        self.__g_i.activate(line_loads, initial_phase)

        # Create maps once upfront
        soil_map = {soil.Name.echo().split(".")[0]: soil for soil in soils}
        
        # Filter and map lines by type
        plate_lines = []
        geogrid_lines = []
        for line in lines:
            if hasattr(line, "Plate"):
                plate_lines.append(line)
            elif hasattr(line, "Geogrid"):
                geogrid_lines.append(line)
                
        # Cache geometry values to avoid repeated property access
        plate_geometries = [(line, 
                           line.Geometry.x1.value, 
                           line.Geometry.y1.value,
                           line.Geometry.x2.value, 
                           line.Geometry.y2.value) for line in plate_lines]
                           
        geogrid_geometries = [(line,
                              line.Geometry.x1.value,
                              line.Geometry.y1.value, 
                              line.Geometry.x2.value,
                              line.Geometry.y2.value) for line in geogrid_lines]

        # Main phase creation loop
        for indexS in range(1, int(abs(self.__plate_length / 2)) + 1):
            new_phase = self.__g_i.phase(self.__g_i.Phases[-1])
            self.phase_names.append(new_phase.Identification.value)

            # Get soil and its bounding box
            soil = soil_map[f"Soil_1_{indexS}"]
            bbox = soil.Parent.BoundingBox
            xmin, ymin = bbox.xMin.value, bbox.yMin.value
            xmax, ymax = bbox.xMax.value, bbox.yMax.value
            soil_box = box(xmin, ymin, xmax, ymax)

            # Deactivate soil
            self.__g_i.deactivate(soil, new_phase)

            # Activate relevant plates
            for line, x1, y1, x2, y2 in plate_geometries:
                plate_line = LineString([(x1, y1), (x2, y2)])
                if ((plate_line.within(soil_box) or plate_line.touches(soil_box)) and 
                    y1 >= ymin and y2 >= ymin):
                    self.__g_i.activate(line, new_phase)

            # Activate relevant geogrids
            for line, x1, y1, x2, y2 in geogrid_geometries:
                nail_line = LineString([(x1, y1), (x2, y2)])
                if nail_line.intersects(soil_box):
                    self.__g_i.activate(line, new_phase)

    def __run(self):
        logger.info("Running calculation...")
        self.__g_i.calculate()

    def __save(self, project_name="my_plaxis_project.p2dx"):
        """Save project and close connection"""
        logger.info("Saving and closing project...")
        save_path = os.path.join(os.getcwd(), project_name)
        self.__g_i.save(save_path)
        #self.__g_i.close()

    def __Output_View(self):
        self.__g_i.view(self.__g_i.Phases[-1])
        logger.info("Viewing output project...")

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
        self.__Output_View()
        
        logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

