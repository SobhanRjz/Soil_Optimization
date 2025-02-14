class MainConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MainConfig, cls).__new__(cls)
            cls._instance.settings = {
                'host': 'localhost',
                'port': 10000
            }
        return cls._instance

    def update(self, key, value):
        """Update configuration dynamically"""
        if key in self.settings:
            self.settings[key] = value
        else:
            raise KeyError(f"'{key}' not found in configuration")

    def get(self, key):
        """Get a configuration value"""
        return self.settings.get(key, None)

# Classes that use the configuration
class Class2:
    def __init__(self):
        self.config = MainConfig()

    def print_config(self):
        print(f"Class2 using: {self.config.settings}")

class Class3:
    def __init__(self):
        self.config = MainConfig()

    def print_config(self):
        print(f"Class3 using: {self.config.settings}")

# Usage
main_config = MainConfig()
class2 = Class2()
class3 = Class3()

# Before update
class2.print_config()
class3.print_config()

# Update config dynamically
main_config.update('host', 'newhost.com')
main_config.update('port', 20000)

# After update, all instances reflect the changes
class2.print_config()
class3.print_config()
