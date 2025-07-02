import os
import json
from datetime import datetime
import yaml
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")


def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_process_yaml(path="configs/process.yaml"):
    process = {}
    try:
        with open(path, "r") as f:
            process = yaml.safe_load(f)
    except Exception as e:
        print(e)

    return process


def save_process_yaml(data, path="configs/process.yaml"):
    try:
        with open(path, 'w') as f:
            f.write( yaml.dump(data, default_flow_style=False))
            return True
    except Exception as e:
        print(e)
        return False


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

def get_google_api_key():
    google_api_key = ''
    google_api_key_index = None
    try:
        google_api_keys = json.loads(os.getenv("GOOGLE_API_KEYS", "[]"))
        if len(google_api_keys) > 0:
            proc = load_process_yaml()
            blocked_arr = proc.get('blocked_google_api_key_index', [])
            if proc and proc.get('google_api_key_index', None) is not None:
                google_api_key_index = proc.get('google_api_key_index')
                key_count = 0
                while key_count < len(google_api_keys):
                    if google_api_key_index in blocked_arr:
                        google_api_key_index += 1
                        google_api_key_index = google_api_key_index % len(google_api_keys)
                        key_count += 1
                    else:
                        break

                if key_count < len(google_api_keys):
                    google_api_key = google_api_keys[google_api_key_index]

            else:
                google_api_key_index = 0
                google_api_key = google_api_keys[google_api_key_index]

            new_key_count = 0
            google_api_key_index += 1
            google_api_key_index = google_api_key_index % len(google_api_keys)

            while new_key_count < len(google_api_keys):
                if google_api_key_index in blocked_arr:
                    new_key_count += 1
                else:
                    break

            if new_key_count < len(google_api_keys):
                print(f"Updated GOOGLE_API_KEY_INDEX: {google_api_key_index}")
                proc['google_api_key_index'] = google_api_key_index
                save_process_yaml(proc)

        else:
            google_api_key = os.getenv('GOOGLE_API_KEY')

    except Exception as e:
        print(f"Error parsing GOOGLE_API_KEYS {e}")
        google_api_key = os.getenv('GOOGLE_API_KEY')

    return google_api_key, google_api_key_index

def block_api_key(idx):
    proc = load_process_yaml()
    blocked_arr = proc.get('blocked_google_api_key_index', [])
    blocked_arr.append(idx)
    proc['blocked_google_api_key_index'] = blocked_arr
    save_process_yaml(proc)