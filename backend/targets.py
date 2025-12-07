from enum import Enum
from pydantic import BaseModel
from typing import Optional

class TargetType(str, Enum):
    CUDA = "CUDA"
    OPENCL = "OpenCL"
    DSP = "DSP"

class DeviceConfig(BaseModel):
    ip_address: str = "127.0.0.1"
    username: str = "user"
    password: Optional[str] = None
    use_adb: bool = False
    adb_device_id: Optional[str] = None # e.g. serial number
    remote_work_dir: str = "/data/local/tmp"
    compiler_cmd: str = "gcc" # Default to host gcc for now, user can change to arm-linux-androideabi-gcc
    mock: bool = False # Run locally by default
