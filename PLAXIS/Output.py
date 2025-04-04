import os
import plxscripting.easy as plx
import math
import numpy as np
from shapely.geometry import LineString, box
import logging
import time
from Config.config import MainConfig
from dataclasses import dataclass
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
@dataclass
class DisplacementData:
    x: float = 0.0
    y: float = 0.0
    utotal: float = 0.0
    phaseName: str = ""
@dataclass
class ElementForceData:
    GeoName: str = ""
    axialForceX: float = 0.0
    x: float = 0.0
    y: float = 0.0
    max_ratio_Structures: float = 0.0
    max_ratio_Soil: float = 0.0

@dataclass
class ElementData:
    elementForceData: ElementForceData = ElementForceData()
    displacementData: DisplacementData = DisplacementData()

class PlaxisModelOutput:
    def __init__(self, model_input = None):
        self.ElementData = ElementData()
        self.config = MainConfig()
        self.__host = self.config.PLAXIS_CONFIG.output['host']
        self.__port = self.config.PLAXIS_CONFIG.output['port']
        self.__password = self.config.PLAXIS_CONFIG.output['password']
        self.__s_i = None
        self.__g_i = None
        self.__output_data = []

        # Load geometry parameters from config
        self.__plate_length = self.config.MODEL_GEOMETRY.plate_length
        self.__phase_names = model_input.phase_names if model_input else None
        self.__start_time = time.time()
        self.__connect()

    def __connect(self):
        """Establish connection to PLAXIS server"""
        logger.info("Connecting to PLAXIS server...")
        self.__s_i, self.__g_i = plx.new_server(self.__host, self.__port, password=self.__password)
    
    def __getnodeid_x(self, phase, nodeid=None):
        """
        returns the result for X-coordinate for the specific phase and node number
        if nodeid is None, returns all X coordinates
        """
        if nodeid is None:
            resultLocation = 'Node'
            return self.__g_i.getresults(phase, self.__g_i.ResultTypes.Soil.X, resultLocation)[:]
        return self.getnodeid_result(phase, self.__g_i.ResultTypes.Soil.X, nodeid)

    def __getnodeid_y(self, phase, nodeid=None):
        """
        returns the result for Y-coordinate for the specific phase and node number
        if nodeid is None, returns all Y coordinates
        """
        if nodeid is None:
            resultLocation = 'Node'
            return self.__g_i.getresults(phase, self.__g_i.ResultTypes.Soil.Y, resultLocation)[:]
        return self.getnodeid_result(phase, self.__g_i.ResultTypes.Soil.Y, nodeid)

    def __getnodeid_utotal(self, phase, nodeid=None):
        """
        returns the result for Y-coordinate for the specific phase and node number
        if nodeid is None, returns all Y coordinates
        """
        if nodeid is None:
            resultLocation = 'Node'
            return self.__g_i.getresults(phase, self.__g_i.ResultTypes.Soil.Utot, resultLocation)[:]
        return self.getnodeid_result(phase, self.__g_i.ResultTypes.Soil.Utot, nodeid)

    def GetOutput(self):
        """Get output from PLAXIS server"""
        logger.info("Getting output from PLAXIS server...")
        self._check_displacement()
        self._check_Axial_Forces()

       

    def _check_displacement(self):
        """Get displacement from PLAXIS server"""
        logger.info("Getting displacement from PLAXIS server...")
        phases = [self.__g_i.Phases[-1]]
        for phase in phases:
            # retrieve values for this phase
            x_coords = self.__getnodeid_x(phase)
            y_coords = self.__getnodeid_y(phase)

            utotal = self.__getnodeid_utotal(phase)

            # Get indices of top 100 utotal values
            top_100_indices = np.argsort(utotal)[-10:]
            
            new_data = [
                DisplacementData(x=x_coords[i], y=y_coords[i], utotal=utotal[i], phaseName=phase.Name.value)
                for i in top_100_indices
            ]
            self.__output_data.extend(new_data)

        # Sort output data based on total displacement (utotal)
        self.__output_data.sort(key=lambda x: x.utotal, reverse=True)
        self.ElementData.displacementData = self.__output_data
        
        logger.info(f"Processed {len(self.__output_data)} points of data")


    def _get_axialForces(self):
        """Get axial forces from PLAXIS server"""
        logger.info("Getting axial forces from PLAXIS server...")
        
        # Create dictionary to store forces by element ID
        element_forces = {}
        # Get only last phase
        phase = self.__g_i.Phases[-1]
        logger.info(f"Processing axial forces for phase: {phase.Name.value}")
        for Geo in self.__g_i.Geogrids[:]:
            axialForcesX = self.__g_i.getresults(Geo, phase, self.__g_i.ResultTypes.Geogrid.Nx2D, 'Node')[:]
            CoordinatesX = self.__g_i.getresults(Geo, phase, self.__g_i.ResultTypes.Geogrid.X, 'Node')[:]
            CoordinatesY = self.__g_i.getresults(Geo, phase, self.__g_i.ResultTypes.Geogrid.Y, 'Node')[:]
            ElementID = Geo.Name.value

            # Process data for each element
            
            element_id = ElementID
            if element_id not in element_forces:
                element_forces[element_id] = []
            
            force_data = ElementForceData(
                GeoName=Geo.Name.value,
                axialForceX=[axialForcesX[i] for i in range(len(axialForcesX))],
                x=CoordinatesX,
                y=CoordinatesY
            )
            element_forces[element_id] = force_data

        # Store the processed data
        self.__axial_forces_data = element_forces
        
        logger.info(f"Processed axial forces for {len(element_forces)} elements")

    def _check_Axial_Forces(self):
        """Check if all axial forces are below 10"""
        self._get_axialForces()
        self._check_Axial_Structures()
        self._check_Axial_Soil()
        self.ElementData.elementForceData = self.__axial_forces_data

    def _check_Axial_Structures(self):
        FS = 1.8
        RebarArea = (self.config.STRUCTURAL_MATERIALS.geogrid.RebarDiameter / 1000) ** 2 * math.pi / 4
        Fy = self.config.STRUCTURAL_MATERIALS.geogrid.Fy * 1000 # KN/m2
        AllowableForce = Fy * RebarArea  / (FS * self.config.STRUCTURAL_MATERIALS.geogrid.HorizentalSpace)

        for element_id, force_data in self.__axial_forces_data.items():
            max_ratio = max(force_data.axialForceX) / AllowableForce
            force_data.max_ratio_Structures = max_ratio
                

    def _check_Axial_Soil(self):
        FS = 2.0
        NominalBond = 50 * 6.89475728 # Kn / m2
        HollowDiameter = self.config.STRUCTURAL_MATERIALS.geogrid.HollowDiameter # M
        NailLength = self.config.STRUCTURAL_MATERIALS.geogrid.nail_length # M
        SoilStrength = [0] * len(NailLength)
        for i, nail_length in enumerate(NailLength):
            SoilStrength[i] = math.pi * HollowDiameter * nail_length * NominalBond / (FS * self.config.STRUCTURAL_MATERIALS.geogrid.HorizentalSpace)

        for i, (_, force_data) in enumerate(self.__axial_forces_data.items()):
            max_ratio = max(force_data.axialForceX) / SoilStrength[i]
            force_data.max_ratio_Soil = max_ratio




    def _Check_Total_Displacement(self):
        """Check if all total displacements are below 10"""
        return all(data.utotal < 10 for data in self.__output_data)


if __name__ == "__main__":
    plaxis_model = PlaxisModelOutput()
    plaxis_model.GetOutput()

