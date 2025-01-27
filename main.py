import os
import plxscripting.easy as plx
import math

class PlaxisModel:
    def __init__(self, host='localhost', port=10000, password='vkrAnSku2/x^$f8~'):
        self.host = host
        self.port = port 
        self.password = password
        self.s_i = None
        self.g_i = None
        self.contour_width = 40
        self.ybot_plate = -10
        self.step_phase = 2
        
    def connect(self):
        """Establish connection to PLAXIS server"""
        self.s_i, self.g_i = plx.new_server(self.host, self.port, password=self.password)
        
    def create_project(self, name="MyNewProject"):
        """Create new PLAXIS project"""
        self.s_i.new()
        self.g_i.ModelType = "Plane strain"
        
    def define_geometry(self):
        """Define model geometry and soil contour"""
        contour_points = [
            (-30, -20),  # Bottom-left corner
            (10, -20),   # Bottom-right corner
            (10, 0),     # Top-right corner
            (-30, 0)     # Top-left corner
        ]
        self.soil_contour = self.g_i.SoilContour.initializerectangular(*contour_points[0], *contour_points[2])
        
    def create_soil_profile(self):
        """Create soil layers and materials"""
        borhole = self.g_i.borehole(0)
        borhole.Head = -20
        self.soillayer = self.g_i.soillayer(20)
        
        # Create soil material
        self.soil_material = self.g_i.soilmat()
        self.soil_material.Identification = "SoilMat"
        self.soil_material.MaterialName = "Clay"
        self.soil_material.SoilModel = "HardeningSoil"
        self.soil_material.gammaUnsat = 20
        self.soil_material.gammaSat = 20
        self.soil_material.E50Ref = 20000
        self.soil_material.EOedRef = 20000
        self.soil_material.EURRef = 60000
        self.soil_material.nu = 0.3
        self.soil_material.phi = 30
        self.soil_material.cRef = 15
        self.soil_material.psi = 0
        
    def create_structural_materials(self):
        """Create geogrid and plate materials"""
        # Geogrid material
        self.geogrid = self.g_i.geogridmat()
        self.geogrid.MaterialType = "Elastic"
        self.geogrid.Identification = "Geogrid_PHI28"
        self.geogrid.setproperties("EA1", 123000)
        
        # Plate material
        self.plate = self.g_i.platemat()
        self.plate.MaterialType = "Elastic"
        self.plate.Identification = "Plate"
        self.plate.setproperties("w", 2.3)
        self.plate.setproperties("EA1", 2000000)
        self.plate.setproperties("EI", 1667)
        self.plate.setproperties("StructNu", 0.2)
        
    def assign_materials(self):
        """Assign materials to layers"""
        self.g_i.setmaterial(self.g_i.Soillayers[-1], self.g_i.SoilMat)
        print("Hardening Soil material defined.")
        
    def create_structures(self):
        """Create structural elements"""
        self.g_i.gotostructures()
        
        # Create shotcrete line
        shotcrete_line = self.g_i.plate((0, 0), (0, -10))
        shotcrete_line[2].setproperties("Plate.Material", self.plate)
        
        # Create horizontal lines
        for y in range(-self.step_phase, self.ybot_plate - 1, -self.step_phase):
            self.g_i.line((0, y), (10, y))
            
        # Create geogrids
        geogrid_length = self.contour_width / 2 - 10
        for y in range(-self.step_phase, self.ybot_plate, -self.step_phase):
            geogrid = self.g_i.geogrid((0, y), (-geogrid_length, y - self.step_phase))
            geogrid[2].setproperties("Geogrid.Material", self.geogrid)

        # Create Line Loads
        load = self.g_i.lineload((0,0), (-self.contour_width / 2 ,0))
        load[3].qy_start = -15 # Kn / m

        
        print("Soil layer created.")

        
    def save_and_close(self, project_name="my_plaxis_project.p2dx"):
        """Save project and close connection"""
        save_path = os.path.join(os.getcwd(), project_name)
        self.g_i.save(save_path)
        self.g_i.close()
        print(f"2D Project '{project_name}' created and saved at '{save_path}'")

    def Automesh(self):
        self.g_i.gotomesh()
        mesh = self.g_i.mesh(0.06) # medium
        print(mesh)
    
    def createPhase(self):
        self.g_i.gotoflow()

        #Activate line load for IntialPhases
        line_loads = self.g_i.LineLoads
        initial_phase = self.g_i.Phases[0]
        self.g_i.activate(line_loads , initial_phase)


        soils = self.g_i.Soils
        lines = self.g_i.Lines
        geogrid_lines = self.g_i.GeoGrids
        # Filter lines that are associated with plates
        # Get all lines in the model
        lines = self.g_i.Lines

        # Filter lines that are associated with geogrid materials
        geogrid_lines = [line for line in lines if hasattr(line, "Geogrid")]

        # Create map of soil names to soil objects
        soil_map = {soil.Name.echo().split(".")[0]: soil for soil in soils}
        line_map = {line.Name.echo().split(".")[0]: line for line in lines}
        Geo_map = {Geo.Name.echo().split(".")[0]: Geo for Geo in geogrid_lines}
        firstIndexGepMap = min(int(name.split('_')[1]) for name in Geo_map.keys())

        for indexS in range(1, int(abs(self.ybot_plate / 2)) + 1): # FIX round

            #create New Phase
            new_phase = self.g_i.phase(self.g_i.Phases[-1])

            # Deactivate Soils
            soilName = f"Soil_1_{indexS}"
            Soil = soil_map[soilName]
            self.g_i.deactivate(Soil, new_phase)

            # Activate ShotCrete
            lineName = f"Line_1_{indexS}"
            ShotCrete = line_map[lineName]
            self.g_i.activate(ShotCrete, new_phase)

            # Activate Nailing
            GeoName = f"Line_{firstIndexGepMap + indexS - 1}_1"
            if GeoName in Geo_map:
                Nail = Geo_map[GeoName]
                self.g_i.activate(Nail, new_phase)

        print("")
    def Run(self):
        self.g_i.calculate()
        print("")

def main():
    # Create and run model
    model = PlaxisModel()
    model.connect()
    model.create_project()
    model.define_geometry()
    model.create_soil_profile()
    model.create_structural_materials()
    model.assign_materials()
    model.create_structures()
    model.Automesh()
    model.createPhase()
    model.Run()
    model.save_and_close()

if __name__ == "__main__":
    main()