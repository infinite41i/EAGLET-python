import json

def load_config(config_file_path: str):
    with open(config_file_path, 'r') as file:
        config_dict = json.load(file)
        return config_dict
