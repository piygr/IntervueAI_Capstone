import os
import json
from datetime import datetime
import yaml

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_session(session_id, new_data):
    folder_path = "logs/sessions"
    filename = f"{session_id}.json"
    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, filename)
    
    # Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}  # empty or corrupt file
    else:
        data = {}

    # Update with new data (merge dictionaries)
    data.update(new_data)

    data.update(dict(timestamp=datetime.now().isoformat()))
    
    # Write updated data back to file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    return file_path



def fetch_session(session_id):
    folder_path = "logs/sessions"
    filename = f"{session_id}.json"
    
    file_path = os.path.join(folder_path, filename)
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}  # empty or corrupt file
    else:
        data = {}

    return data

