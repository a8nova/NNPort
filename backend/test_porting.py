import requests
import time
import os
import sys

# Dummy model
model_code = """
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)
    
    def forward(self, x):
        return x # Identity for easy convergence mock

model = SimpleModel()
"""

with open("port_model.py", "w") as f:
    f.write(model_code)

def run_test():
    base_url = "http://127.0.0.1:8000"
    
    # Upload
    files = {'file': open('port_model.py', 'rb')}
    res_up = requests.post(f"{base_url}/upload", files=files)
    print("Upload:", res_up.status_code)
    if res_up.status_code != 200:
        return False
        
    filename = res_up.json()['filename']
    
    # Port
    payload = {
        "source_filename": filename,
        "target_type": "DSP",
        "device_config": {"mock": True},
        "max_iterations": 3
    }
    
    print("Requesting Porting...")
    res_port = requests.post(f"{base_url}/port", json=payload)
    print("Porting Response:", res_port.status_code)
    
    if res_port.status_code != 200:
        print(res_port.text)
        return False
        
    logs = res_port.json()['logs']
    print("Logs Count:", len(logs))
    
    # Check if we have logs for "Result"
    success = False
    for log in logs:
        print(f"[{log['stage']}] {log['status']}")
        if log['stage'] == 'Result' and log['status'] == 'Success':
            success = True
            
    return success

if __name__ == "__main__":
    try:
        success = run_test()
    except Exception as e:
        print(e)
        success = False
    finally:
        if os.path.exists("port_model.py"):
            os.remove("port_model.py")
            
    if success:
        print("TEST PASSED")
        sys.exit(0)
    else:
        print("TEST FAILED")
        sys.exit(1)
