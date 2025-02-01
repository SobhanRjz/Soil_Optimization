import os
import plxscripting.easy as plx
import math
import numpy as np
from shapely.geometry import LineString, box
import logging
import time
from Config.config import PLAXIS_CONFIG, MODEL_GEOMETRY
from dataclasses import dataclass

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

class PlaxisModelOutput:
    def __init__(self, model_input, host=PLAXIS_CONFIG['output']['host'], 
                 port=PLAXIS_CONFIG['output']['port'], 
                 password=PLAXIS_CONFIG['output']['password']):
        self.__host = host
        self.__port = port
        self.__password = password
        self.__s_i = None
        self.__g_i = None
        self.__output_data = []

        # Load geometry parameters from config
        self.__plate_length = MODEL_GEOMETRY['plate_length']
        self.__phase_names = model_input.phase_names
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
        phases = [self.__g_i.Phases[-1], self.__g_i.Phases[-2]]
        for phase in phases:
            # retrieve values for this phase
            x_coords = self.__getnodeid_x(phase)
            y_coords = self.__getnodeid_y(phase)
            ux_values = self.__getnodeid_ux(phase)
            uy_values = self.__getnodeid_uy(phase)

            # Process and store data for each point
            for x, y, ux, uy in zip(x_coords, y_coords, ux_values, uy_values):
                utotal = math.sqrt(ux**2 + uy**2)  # Calculate total displacement
                point_data = OutputClassData(x=x, y=y, ux=ux, uy=uy, utotal=utotal, phaseName=phase.Name.value)
                self.__output_data.append(point_data)

        # Sort output data based on total displacement (utotal)
        self.__output_data.sort(key=lambda x: x.utotal, reverse=True)
        logger.info(f"Processed {len(self.__output_data)} points of data")
