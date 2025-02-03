import subprocess

# Path to PLAXIS executable
plaxis_path = r"C:\Program Files\Seequent\PLAXIS 2D 2024\Plaxis2DXInput.exe"

# Command for the first instance
command_0 = [
    plaxis_path
]
command_1 = [
    plaxis_path,
    "--type=gpu-process",
    "--no-sandbox",
    "--locales-dir-path=C:\\Program Files\\Seequent\\PLAXIS 2D 2024\\dcef/locales",
    "--log-severity=disable",
    "--resources-dir-path=C:\\Program Files\\Seequent\\PLAXIS 2D 2024\\dcef",
    "--lang=en-US",
    "--user-data-dir=C:\\Users\\sobha\\AppData\\Local\\CEF\\User Data",
    "--gpu-preferences=UAAAAAAAAADgAAAYAAAAAAAAAAAAAAAAAABgAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEgAAAAAAAAASAAAAAAAAAAYAAAAAgAAABAAAAAAAAAAGAAAAAAAAAAQAAAAAAAAAAAAAAAOAAAAEAAAAAAAAAABAAAADgAAAAgAAAAAAAAACAAAAAAAAAA=",
    "--use-gl=angle",
    "--use-angle=swiftshader-webgl",
    "--log-file=C:\\Program Files\\Seequent\\PLAXIS 2D 2024\\debug.log",
    "--mojo-platform-channel-handle=2008",
    "--field-trial-handle=2028,i,7095039279584526132,4199722338841624551,131072",
    "--disable-features=BackForwardCache,CalculateNativeWinOcclusion,WinUseBrowserSpellChecker",
    "/prefetch:2"
]

command_2_2 = [
    plaxis_path
]
# Command for the second instance
command_2 = [
    plaxis_path,
    "--type=gpu-process",
    "--no-sandbox",
    "--locales-dir-path=C:\\Program Files\\Seequent\\PLAXIS 2D 2024\\dcef/locales",
    "--log-severity=disable",
    "--resources-dir-path=C:\\Program Files\\Seequent\\PLAXIS 2D 2024\\dcef",
    "--lang=en-US",
    "--user-data-dir=C:\\Users\\sobha\\AppData\\Local\\CEF\\User Data 2",
    "--gpu-preferences=UAAAAAAAAADgAAAYAAAAAAAAAAAAAAAAAABgAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEgAAAAAAAAASAAAAAAAAAAYAAAAAgAAABAAAAAAAAAAGAAAAAAAAAAQAAAAAAAAAAAAAAAOAAAAEAAAAAAAAAABAAAADgAAAAgAAAAAAAAACAAAAAAAAAA=",
    "--use-gl=angle",
    "--use-angle=swiftshader-webgl",
    "--log-file=C:\\Program Files\\Seequent\\PLAXIS 2D 2024\\debug2.log",
    "--mojo-platform-channel-handle=1745",
    "--field-trial-handle=2025,i,3379501947119101148,14410968034434445383,131072",
    "--disable-features=BackForwardCache,CalculateNativeWinOcclusion,WinUseBrowserSpellChecker",
    "/prefetch:2"
]

# Start the first instance
subprocess.Popen(command_0)
subprocess.Popen(command_1)
subprocess.Popen(command_2_2)
# Start the second instance
subprocess.Popen(command_2)


print("PLAXIS instances started successfully.")