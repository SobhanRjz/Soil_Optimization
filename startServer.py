from plxscripting.easy import *
import subprocess
import time

PLAXIS_PATH = r'C:\Program Files\Seequent\PLAXIS 2D 2024\\Plaxis2DXInput.exe'  # Specify PLAXIS path on server.

PORT_i = 10000  # Define a port number.
PORT_o = 10001

PASSWORD = 'vkrAnSku2/x^$f8~'  # Define a password (up to user choice).

# Start PLAXIS process
plaxis_process = subprocess.Popen([PLAXIS_PATH, f'--AppServerPassword={PASSWORD}', f'--AppServerPort={PORT_i}'], shell=False)

# Try connecting until PLAXIS is ready
connected = False
max_attempts = 30  # Maximum number of connection attempts
attempt = 0

while not connected and attempt < max_attempts:
    try:
        s_i, g_i = new_server('localhost', PORT_i, password=PASSWORD)
        connected = True
    except:
        attempt += 1
        time.sleep(1)

if not connected:
    raise Exception("Failed to connect to PLAXIS after maximum attempts")
