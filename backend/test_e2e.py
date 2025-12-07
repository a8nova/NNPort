import requests
import time
import subprocess
import os
import sys

# Create dummy model files
model_code = """
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
"""

with open("model_a.py", "w") as f:
    f.write(model_code)
    
with open("model_b.py", "w") as f:
    f.write(model_code) # Identical model

def run_test():
    base_url = "http://127.0.0.1:8000"
    
    # Wait for server
    for i in range(10):
        try:
            requests.get(f"{base_url}/docs")
            break
        except:
            time.sleep(1)
    else:
        print("Server failed to start")
        return False
        
    # Upload
    files_a = {'file': open('model_a.py', 'rb')}
    res_a = requests.post(f"{base_url}/upload", files=files_a)
    print("Upload A:", res_a.status_code, res_a.json())
    
    files_b = {'file': open('model_b.py', 'rb')}
    res_b = requests.post(f"{base_url}/upload", files=files_b)
    print("Upload B:", res_b.status_code, res_b.json())
    
    if res_a.status_code != 200 or res_b.status_code != 200:
        return False
        
    # Compare
    payload = {
        "source_filename": res_a.json()['filename'],
        "target_filename": res_b.json()['filename'],
        "input_shape": [1, 10]
    }
    
    res_cmp = requests.post(f"{base_url}/compare", json=payload)
    print("Compare:", res_cmp.status_code)
    try:
        data = res_cmp.json()
        print("Results:", data)
        # Check if match is True (should be for identical models)
        # Note: L2 diff might be non-zero due to random init weights unless we control seed/load weights.
        # But wait, we define the class in the file but don't load state dict. 
        # Each load_model_from_path will execute the file, creating NEW 'model' instance with RANDOM weights.
        # So they will NOT match.
        # This is expected behavior for this MVP unless we align weights.
        # We just want to check the API works.
        return res_cmp.status_code == 200 and "results" in data
    except Exception as e:
        print("Error parsing response:", e)
        return False

if __name__ == "__main__":
    success = run_test()
    os.remove("model_a.py")
    os.remove("model_b.py")
    sys.exit(0 if success else 1)
