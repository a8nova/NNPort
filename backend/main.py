from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import os
import tempfile
from backend.porting import PortingEngine
from backend.targets import TargetType, DeviceConfig

app = FastAPI()

# Allow CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

porting_engine = PortingEngine()

class PortRequest(BaseModel):
    source_filename: str
    target_type: TargetType
    device_config: DeviceConfig = DeviceConfig()
    max_iterations: int = 3
    input_shape: List[int] = [1, 10] # Default to what user likely needs, but they can change it

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "nnport_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/status")
async def get_system_status():
    """Get system status including NDK and ADB devices"""
    import subprocess
    
    status = {
        "ndk": {
            "available": False,
            "path": None,
            "compiler": None
        },
        "adb": {
            "available": False,
            "devices": []
        }
    }
    
    # Check NDK
    compiler = porting_engine.compiler
    if compiler.ndk_path:
        status["ndk"]["available"] = True
        status["ndk"]["path"] = compiler.ndk_path
        android_clang = compiler._get_android_clang(compiler.ndk_path)
        status["ndk"]["compiler"] = android_clang if android_clang else "Not found"
    
    # Check ADB devices
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            status["adb"]["available"] = True
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            devices = [line.split()[0] for line in lines if line.strip() and '\tdevice' in line]
            status["adb"]["devices"] = devices
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return status

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    return {"filename": file.filename, "location": file_location}

class VerifyRequest(BaseModel):
    source_filename: str = "manual_code.cpp" # Filename of uploaded C++ code
    reference_filename: str = "model.py"      # Filename of uploaded PyTorch model
    target_type: TargetType = TargetType.DSP
    device_config: DeviceConfig = DeviceConfig()
    input_shape: List[int] = [1, 5]
    max_iterations: int = 3  # Number of auto-fix iterations

@app.post("/verify")
async def verify_code(req: VerifyRequest):
    source_path = os.path.join(UPLOAD_DIR, req.source_filename)
    ref_path = os.path.join(UPLOAD_DIR, req.reference_filename)
    
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail=f"Source file {req.source_filename} not found")
    if not os.path.exists(ref_path):
         raise HTTPException(status_code=404, detail=f"Reference file {req.reference_filename} not found")
         
    logs = porting_engine.verify_manual_code(
        manual_source_path=source_path,
        reference_model_path=ref_path,
        target=req.target_type,
        config=req.device_config,
        input_shape=req.input_shape,
        max_iterations=req.max_iterations
    )
    return {"logs": logs}

@app.post("/port")
async def port_model(req: PortRequest):
    source_path = os.path.join(UPLOAD_DIR, req.source_filename)
    
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Source file not found")
        
    try:
        logs = porting_engine.run_porting_loop(
            source_path, 
            req.target_type, 
            req.device_config, 
            req.max_iterations,
            req.input_shape
        )
        return {"logs": logs}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

