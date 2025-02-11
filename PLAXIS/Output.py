import os
import plxscripting.easy as plx
import math
import numpy as np
from shapely.geometry import LineString, box
import logging
import time
from Config.config import PLAXIS_CONFIG, MODEL_GEOMETRY, STRUCTURAL_MATERIALS
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OutputClassData:
    x: float
    y: float
    ux: float
    uy: float
    utotal: float
    phaseName: str
@dataclass
class ElementForceData:
    GeoName: str = ""
    axialForceX: float = 0.0
    x: float = 0.0
    y: float = 0.0
    max_ratio_Structures: float = 0.0
    max_ratio_Soil: float = 0.0
class PlaxisModelOutput:
    def __init__(self, model_input = None, host=PLAXIS_CONFIG.output['host'], 
                 port=PLAXIS_CONFIG.output['port'], 
                 password=PLAXIS_CONFIG.output['password']):
        self.__host = host
        self.__port = port
        self.__password = password
        self.__s_i = None
        self.__g_i = None
        self.__output_data = []

        # Load geometry parameters from config
        self.__plate_length = MODEL_GEOMETRY.plate_length
        self.__phase_names = model_input.phase_names if model_input else None
        self.__start_time = time.time()

    def __connect(self):
        """Establish connection to PLAXIS server"""
        logger.info("Connecting to PLAXIS server...")
        self.__s_i, self.__g_i = plx.new_server(self.__host, self.__port, password=self.__password)
    
    def __getnodeid_result(self, phase, resulttype, nodeid):
        """
        finds the index in the results lists for the node id 
        and then returns the value for the resulttype for soil data
        :param phase: Output phase object
        :param resulttype: PLAXIS Output Soil Result Type
        :param nodeid: integer of the node ID
        :return: result value
        """
        nodeindex = None
        resultLocation = 'Node'
        _resulttypeID = self.__g_i.ResultTypes.Soil.NodeID
        _nodeids = self.__g_i.getresults(phase, _resulttypeID, resultLocation)
        # use local list to find the index, otherwise an index command will be send to PLAXIS:
        nodeindex = _nodeids[:].index(nodeid)
        if nodeindex is not None:
            _resultvalues = self.__g_i.getresults(phase, resulttype, resultLocation)
            _requestedvalue = _resultvalues[nodeindex]
            return _requestedvalue

        logger.warning('Could not find the requested node number in the results of this phase')
        return None

    def __getnodeid_ux(self, phase, nodeid=None):
        """
        returns the result for Ux for the specific phase and node number
        if nodeid is None, returns all Ux values
        """
        if nodeid is None:
            resultLocation = 'Node'
            return self.__g_i.getresults(phase, self.__g_i.ResultTypes.Soil.Ux, resultLocation)[:]
        return self.getnodeid_result(phase, self.__g_i.ResultTypes.Soil.Ux, nodeid)

    def __getnodeid_uy(self, phase, nodeid=None):
        """
        returns the result for Uy for the specific phase and node number
        if nodeid is None, returns all Uy values
        """
        if nodeid is None:
            resultLocation = 'Node'
            return self.__g_i.getresults(phase, self.__g_i.ResultTypes.Soil.Uy, resultLocation)[:]
        return self.getnodeid_result(phase, self.__g_i.ResultTypes.Soil.Uy, nodeid)

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

    def GetOutput(self):
        """Get output from PLAXIS server"""
        self.__connect()
        logger.info("Getting output from PLAXIS server...")

        # Iterate through all phases
        # Get only first and last phase
        phases = [self.__g_i.Phases[-1]]
        for phase in phases:
            # retrieve values for this phase
            x_coords = self.__getnodeid_x(phase)
            y_coords = self.__getnodeid_y(phase)
            ux_values = self.__getnodeid_ux(phase)
            uy_values = self.__getnodeid_uy(phase)

            # Process and store data for each point
            # Use numpy for faster vector operations
            
            utotal = np.sqrt(np.array(ux_values)**2 + np.array(uy_values)**2)
            
            # Pre-allocate list size for better performance
            # Get indices of top 100 utotal values
            top_100_indices = np.argsort(utotal)[-10:]
            
            new_data = [
                OutputClassData(x=x_coords[i], y=y_coords[i], ux=ux_values[i], uy=uy_values[i], 
                              utotal=utotal[i], phaseName=phase.Name.value)
                for i in top_100_indices
            ]
            self.__output_data.extend(new_data)

        # Sort output data based on total displacement (utotal)
        self.__output_data.sort(key=lambda x: x.utotal, reverse=True)
        
        logger.info(f"Processed {len(self.__output_data)} points of data")



    def _get_axialForces(self):
        """Get axial forces from PLAXIS server"""
        self.__connect()
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
                axialForceX=[axialForcesX[i] * STRUCTURAL_MATERIALS.geogrid.HorizentalSpace for i in range(len(axialForcesX))],
                x=CoordinatesX,
                y=CoordinatesY
            )
            element_forces[element_id] = force_data

        # Store the processed data
        self.__axial_forces_data = element_forces
        self._Check_Axial_Forces()
        logger.info(f"Processed axial forces for {len(element_forces)} elements")

    def _Check_Axial_Forces(self):
        """Check if all axial forces are below 10"""
        self._check_Axial_Structures()
        self._check_Axial_Soil()

        return all(data.axialForceX < 10 for data in self.__axial_forces_data)
    def _check_Axial_Structures(self):
        RebarArea = STRUCTURAL_MATERIALS.geogrid.RebarDiameter ** 2 * math.pi / 4
        BaseForce = STRUCTURAL_MATERIALS.geogrid.Fy * RebarArea  / 1.8
        
        max_ratios = {}
        for element_id, force_data in self.__axial_forces_data.items():
            max_ratio = max(force_data.axialForceX) / BaseForce
            force_data.max_ratio_Structures = max_ratio
                

    def _check_Axial_Soil(self):
        AlphaBond = 0.3
        Su = 100 # Kn / m2
        CodeRatio = 2.0
        RebarDiameter = STRUCTURAL_MATERIALS.geogrid.RebarDiameter / 100 # m
        SoilStrength = 1.2 * math.pi * RebarDiameter * STRUCTURAL_MATERIALS.geogrid.nail_length * AlphaBond * Su  / CodeRatio

        for element_id, force_data in self.__axial_forces_data.items():
            max_ratio = max(force_data.axialForceX) / SoilStrength
            force_data.max_ratio_Soil = max_ratio
        


    def _Check_Total_Displacement(self):
        """Check if all total displacements are below 10"""
        return all(data.utotal < 10 for data in self.__output_data)
