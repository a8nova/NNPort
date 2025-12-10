from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import os
import tempfile
import platform
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env.dev
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.dev')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✓ Loaded environment variables from {env_path}")
else:
    print(f"⚠ Warning: .env.dev not found at {env_path}")
from backend.porting import PortingEngine
from backend.targets import TargetType, DeviceConfig
from backend.jobs.job_manager import JobManager
from backend.api.websocket import websocket_endpoint, manager as ws_manager
from backend.api.routes import jobs as jobs_router
from backend.toolchain_discovery import ToolchainDiscovery, ADBDiscovery
from backend.device_discovery import DeviceDiscovery

app = FastAPI()

# Initialize job manager
job_manager = JobManager()

# Allow CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include job routes
app.include_router(jobs_router.router, tags=["jobs"])

# WebSocket endpoint
@app.websocket("/ws/port/{job_id}")
async def websocket_port_endpoint(websocket: WebSocket, job_id: str):
    await websocket_endpoint(websocket, job_id)

porting_engine = PortingEngine()

class PortRequest(BaseModel):
    source_filename: str
    target_type: TargetType
    device_config: DeviceConfig = DeviceConfig()
    max_iterations: int = 3
    input_shape: List[int] = [1, 10]
    debug_instructions: str = ""  # User-provided debugging context

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "nncompass_uploads")
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

@app.post("/test-connection")
async def test_connection(config: DeviceConfig):
    """Test connection to target device"""
    import subprocess
    
    if config.connection_type == "ssh":
        try:
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_kwargs = {
                'hostname': config.ssh_host,
                'port': config.ssh_port,
                'username': config.ssh_user,
                'timeout': 10
            }
            
            if config.ssh_key_path:
                connect_kwargs['key_filename'] = os.path.expanduser(config.ssh_key_path)
            elif config.ssh_password:
                connect_kwargs['password'] = config.ssh_password
            
            ssh.connect(**connect_kwargs)
            stdin, stdout, stderr = ssh.exec_command('uname -a')
            output = stdout.read().decode('utf-8').strip()
            ssh.close()
            
            return {
                "success": True,
                "message": f"SSH connection successful",
                "details": f"System: {output}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": "SSH connection failed",
                "details": str(e)
            }
    
    elif config.connection_type == "adb" or config.use_adb:
        try:
            adb_path = config.adb_path if hasattr(config, 'adb_path') else 'adb'
            cmd = [adb_path, "devices"]
            if config.adb_device_id:
                cmd = [adb_path, "-s", config.adb_device_id, "shell", "getprop", "ro.product.model"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": "ADB connection successful",
                    "details": result.stdout.strip()
                }
            else:
                return {
                    "success": False,
                    "message": "ADB connection failed",
                    "details": result.stderr
                }
        except Exception as e:
            return {
                "success": False,
                "message": "ADB command failed",
                "details": str(e)
            }
    
    else:
        return {
            "success": True,
            "message": "Local execution - no connection test needed",
            "details": "Will run on host machine"
        }

@app.get("/discover-toolchains")
async def discover_toolchains():
    """Discover available toolchains on the host system"""
    try:
        discovery = ToolchainDiscovery()
        results = discovery.discover_all()
        return results
    except Exception as e:
        return {
            "error": str(e),
            "host_info": {
                "os": platform.system(),
                "architecture": platform.machine()
            },
            "toolchains": [],
            "gpu_sdks": {},
            "ndk": None
        }

@app.get("/discover-devices")
async def discover_devices():
    """Discover available compute devices (OpenCL, CUDA, etc.)"""
    try:
        devices = DeviceDiscovery.discover_all_devices()
        return devices
    except Exception as e:
        return {
            "error": str(e),
            "opencl": [],
            "cuda": [],
            "host_cpu": []
        }

@app.get("/discover-adb")
async def discover_adb():
    """Discover ADB installation on the host system"""
    try:
        adb_path = ADBDiscovery.find_adb()
        if adb_path:
            info = ADBDiscovery.get_adb_info(adb_path)
            return {
                "found": True,
                "path": adb_path,
                "version": info.get("version"),
                "devices": info.get("devices", [])
            }
        else:
            return {
                "found": False,
                "error": "ADB not found. Please install Android SDK Platform Tools."
            }
    except Exception as e:
        return {
            "found": False,
            "error": str(e)
        }

@app.post("/test-toolchain")
async def test_toolchain(config: DeviceConfig):
    """Test toolchain by compiling a simple hello world"""
    import tempfile
    import subprocess
    
    try:
        # Simple hello world C code
        test_code = """
#include <stdio.h>
int main() {
    printf("Hello World\\n");
    return 0;
}
"""
        
        # Create temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as src_file:
            src_file.write(test_code)
            src_path = src_file.name
        
        bin_path = src_path.replace('.c', '.bin')
        
        try:
            toolchain = config.toolchain if hasattr(config, 'toolchain') else None
            if toolchain:
                cmd = [toolchain.compiler_path]
                
                if toolchain.sysroot:
                    cmd.extend(["--sysroot", toolchain.sysroot])
                
                for include_path in toolchain.include_paths:
                    cmd.extend(["-I", include_path])
                
                for lib_path in toolchain.library_paths:
                    cmd.extend(["-L", lib_path])
                
                cmd.extend(toolchain.compiler_flags)
                cmd.extend(["-o", bin_path, src_path])
                cmd.extend(toolchain.linker_flags)
            else:
                cmd = ["gcc", "-o", bin_path, src_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Cleanup
                os.remove(src_path)
                if os.path.exists(bin_path):
                    os.remove(bin_path)
                
                return {
                    "success": True,
                    "message": "Toolchain test successful",
                    "details": f"Compiled test program successfully with: {' '.join(cmd[:3])}"
                }
            else:
                return {
                    "success": False,
                    "message": "Compilation failed",
                    "details": f"STDERR: {result.stderr}\nCommand: {' '.join(cmd)}"
                }
        finally:
            # Cleanup
            if os.path.exists(src_path):
                os.remove(src_path)
            if os.path.exists(bin_path):
                os.remove(bin_path)
    
    except Exception as e:
        return {
            "success": False,
            "message": "Toolchain test failed",
            "details": str(e)
        }

class VerifyRequest(BaseModel):
    source_filename: str = "manual_code.cpp"
    reference_filename: str = "model.py"
    target_type: TargetType = TargetType.DSP
    device_config: DeviceConfig = DeviceConfig()
    input_shape: List[int] = [1, 5]
    max_iterations: int = 3
    debug_instructions: str = ""  # User-provided debugging context

async def _run_verify_job(job_id: str, source_path: str, ref_path: str, req: VerifyRequest):
    """Background task to run verification"""
    # Store logs to send when WebSocket connects
    collected_logs = []
    
    async def callback(message: dict):
        collected_logs.append(message)
        await ws_manager.broadcast(job_id, message)
    
    job_manager.update_job(job_id, {"state": "running"})
    
    # Give frontend time to connect WebSocket
    await asyncio.sleep(0.5)
    
    try:
        logs = porting_engine.verify_manual_code(
            manual_source_path=source_path,
            reference_model_path=ref_path,
            target=req.target_type,
            config=req.device_config,
            input_shape=req.input_shape,
            max_iterations=req.max_iterations,
            callback=callback,
            job_id=job_id,
            debug_instructions=req.debug_instructions
        )
        job_manager.update_job(job_id, {"state": "completed", "logs": logs})
        
        # Send any missed logs
        for log in collected_logs:
            await ws_manager.broadcast(job_id, log)
        
        await ws_manager.broadcast(job_id, {"type": "job_complete", "status": "success", "logs": logs})
    except Exception as e:
        job_manager.update_job(job_id, {"state": "failed", "error": str(e)})
        await ws_manager.broadcast(job_id, {"type": "job_complete", "status": "failed", "error": str(e)})

@app.post("/verify")
async def verify_code(req: VerifyRequest, background_tasks: BackgroundTasks):
    source_path = os.path.join(UPLOAD_DIR, req.source_filename)
    ref_path = os.path.join(UPLOAD_DIR, req.reference_filename)
    
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail=f"Source file {req.source_filename} not found")
    if not os.path.exists(ref_path):
         raise HTTPException(status_code=404, detail=f"Reference file {req.reference_filename} not found")
    
    # Create job
    job_id = job_manager.create_job({
        "type": "verify",
        "source_filename": req.source_filename,
        "reference_filename": req.reference_filename,
        "target_type": req.target_type,
        "input_shape": req.input_shape
    })
    
    # Run in background
    background_tasks.add_task(_run_verify_job, job_id, source_path, ref_path, req)
    
    return {"job_id": job_id}

async def _run_port_job(job_id: str, source_path: str, req: PortRequest):
    """Background task to run porting"""
    collected_logs = []
    
    async def callback(message: dict):
        collected_logs.append(message)
        await ws_manager.broadcast(job_id, message)
    
    job_manager.update_job(job_id, {"state": "running"})
    
    # Give frontend time to connect WebSocket
    await asyncio.sleep(0.5)
    
    try:
        logs = porting_engine.run_porting_loop(
            source_path, 
            req.target_type, 
            req.device_config, 
            req.max_iterations,
            req.input_shape,
            callback=callback,
            job_id=job_id,
            debug_instructions=req.debug_instructions
        )
        job_manager.update_job(job_id, {"state": "completed", "logs": logs})
        
        # Send any missed logs
        for log in collected_logs:
            await ws_manager.broadcast(job_id, log)
        
        await ws_manager.broadcast(job_id, {"type": "job_complete", "status": "success", "logs": logs})
    except Exception as e:
        import traceback
        traceback.print_exc()
        job_manager.update_job(job_id, {"state": "failed", "error": str(e)})
        await ws_manager.broadcast(job_id, {"type": "job_complete", "status": "failed", "error": str(e)})

@app.post("/port")
async def port_model(req: PortRequest, background_tasks: BackgroundTasks):
    source_path = os.path.join(UPLOAD_DIR, req.source_filename)
    
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Source file not found")
    
    # Create job
    job_id = job_manager.create_job({
        "type": "port",
        "source_filename": req.source_filename,
        "target_type": req.target_type,
        "input_shape": req.input_shape
    })
    
    # Run in background
    background_tasks.add_task(_run_port_job, job_id, source_path, req)
    
    return {"job_id": job_id}

