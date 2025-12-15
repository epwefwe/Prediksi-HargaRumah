import yaml
import joblib
import platform
import os
from pathlib import Path

# Adjusting paths for the new structure where everything is relative to the app execution directory

# Robust base directory detection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_dir():
    return Path(BASE_DIR)

def load_params(lokasi_file):
    with open(lokasi_file, 'r') as file:
        params = yaml.safe_load(file)
    return params

def pickle_load(file_path: str):
    return joblib.load(file_path)

def pickle_dump(data, file_path: str) -> None:
    joblib.dump(data, file_path)

# Simplified path helpers
def get_config_path():
    # In Docker: /app/config/params.yaml (next to util.py which is in /app)
    # Locally: ROOT/api/util.py. Config is ../config/params.yaml
    
    # Check if config exists in same dir (Docker style)
    local_config = os.path.join(BASE_DIR, "config", "params.yaml")
    if os.path.exists(local_config):
        return local_config
    
    # Else check parent dir (Local style if running from api/)
    parent_config = os.path.join(os.path.dirname(BASE_DIR), "config", "params.yaml")
    return parent_config

def get_model_path(config):
    # Model path from config is 'models/production_model.pkl'
    # In Docker: /app/models/production_model.pkl (BASE_DIR/models/...)
    # Locally: ROOT/api/models/production_model.pkl (BASE_DIR/models/...)
    # Both seem to respect BASE_DIR + relative path from api folder
    
    model_rel_path = config["production_model_path"]
    
    # If path starts with 'models/', and we are in BASE_DIR (which has 'models/'), join them.
    full_path = os.path.join(BASE_DIR, model_rel_path)
    return full_path