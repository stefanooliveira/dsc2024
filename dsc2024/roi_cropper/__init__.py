import os
import json

_icao_locations = None

def get_icao_locations():
    return _icao_locations

def _load_icao_locations():
    global _icao_locations
    if _icao_locations is None:
        try:
            # Get the directory of the current script
            dir_path = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(dir_path, 'icao_locations.json')
            with open(file_path, 'r') as file:
                _icao_locations = json.load(file)
        except FileNotFoundError as fnf:
            raise FileNotFoundError(f"The file '{file_path}' was not found.") from fnf
        except json.JSONDecodeError as jde:
            raise ValueError("The file 'icao_locations.json' is not a valid JSON file.") from jde

_load_icao_locations()  # Load data when the package is imported

from . import cropper