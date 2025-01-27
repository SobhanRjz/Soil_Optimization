from plxscripting.easy import new_server

try:
    # Connect to the Plaxis 2D Input server
    server, g_i = new_server('localhost', 10000, password='vkrAnSku2/x^$f8~')
    print("Connected to Plaxis 2D Input successfully!")
except Exception as e:
    print(f"Failed to connect to Plaxis 2D Input: {e}")