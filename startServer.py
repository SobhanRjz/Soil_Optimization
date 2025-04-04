from plxscripting.easy import *
import subprocess
import time
import logging

class PlaxisServer:
    """
    Class to manage PLAXIS server instances and connections.
    """
    def __init__(self, plaxis_path=None, password=None, input_port=10000, output_port=10001):
        """
        Initialize the PLAXIS server manager.
        
        Args:
            plaxis_path (str): Path to PLAXIS executable
            password (str): Password for PLAXIS server connection
            input_port (int): Port for input connection
            output_port (int): Port for output connection
        """
        self.plaxis_path = plaxis_path or r'C:\Program Files\Seequent\PLAXIS 2D 2024\\Plaxis2DXInput.exe'
        self.password = password or 'vkrAnSku2/x$f8~'
        self.input_port = input_port
        self.output_port = output_port
        self.processes = []
        self.s_i = None
        self.g_i = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def start_plaxis_instances(self, additional_ports=None):
        """
        Start PLAXIS instances with specified ports.
        
        Args:
            additional_ports (list): Additional ports to start PLAXIS instances on
        
        Returns:
            bool: True if all instances started successfully
        """
        try:
            # Start main instance
            self.processes.append(
                subprocess.Popen([self.plaxis_path, f'--AppServerPassword={self.password}', 
                                 f'--AppServerPort={self.input_port}'], shell=True)
            )
            self.logger.info(f"Started PLAXIS instance on port {self.input_port}")
            
            # Start additional instances if specified
            if additional_ports:
                for port in additional_ports:
                    self.processes.append(
                        subprocess.Popen([self.plaxis_path, f'--AppServerPassword={self.password}', 
                                         f'--AppServerPort={port}'], shell=True)
                    )
                    self.logger.info(f"Started PLAXIS instance on port {port}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to start PLAXIS instances: {e}")
            return False
    
    def connect_to_plaxis(self, max_attempts=30, delay=1):
        """
        Connect to PLAXIS server with retry mechanism.
        
        Args:
            max_attempts (int): Maximum number of connection attempts
            delay (int): Delay between attempts in seconds
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        attempt = 0
        connected = False
        
        while not connected and attempt < max_attempts:
            try:
                self.logger.info(f"Attempting to connect to PLAXIS (attempt {attempt+1}/{max_attempts})")
                self.s_i, self.g_i = new_server('localhost', self.input_port, password=self.password)
                
                # # Initialize new project
                self.s_i.new()
                # self.g_i.ModelType = "Plane strain"
                connected = True
                self.logger.info("Successfully connected to PLAXIS")
            except Exception as e:
                attempt += 1
                self.logger.warning(f"Connection attempt {attempt} failed: {e}")
                time.sleep(delay)
        
        if not connected:
            self.logger.error("Failed to connect to PLAXIS after maximum attempts")
            raise ConnectionError("Failed to connect to PLAXIS after maximum attempts")
        
        return connected

    def Start_Server(self):
        """
        Main function to start and connect to PLAXIS servers.
        """
        # Start PLAXIS instances
        self.start_plaxis_instances()
        
        # Connect to PLAXIS
        self.connect_to_plaxis()
        
        return self

if __name__ == "__main__":
    plaxis_server = PlaxisServer()
    plaxis_server.Start_Server()
