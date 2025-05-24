import sys
import os
import yaml

def load_config(yaml_path):
    if not os.path.exists(yaml_path):
        print(f"File not found: {yaml_path}")
        sys.exit(1)
        # return None
    
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Unexpected error: {e}")
