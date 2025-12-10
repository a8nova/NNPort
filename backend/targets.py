from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Dict

class TargetType(str, Enum):
    CUDA = "CUDA"
    OPENCL = "OpenCL"
    DSP = "DSP"

class ConnectionType(str, Enum):
    LOCAL = "local"
    ADB = "adb"
    SSH = "ssh"

class ToolchainConfig(BaseModel):
    compiler_path: str = "gcc"
    sysroot: Optional[str] = None
    include_paths: List[str] = []
    library_paths: List[str] = []
    compiler_flags: List[str] = []
    linker_flags: List[str] = []
    architecture: str = "x86_64"
    abi: Optional[str] = None
    endianness: str = "little"

class DeviceConfig(BaseModel):
    # Connection type
    connection_type: ConnectionType = ConnectionType.LOCAL
    
    # Legacy fields (keep for backward compatibility)
    ip_address: str = "127.0.0.1"
    username: str = "user"
    password: Optional[str] = None
    
    # SSH Configuration
    ssh_host: Optional[str] = None
    ssh_port: int = 22
    ssh_user: Optional[str] = None
    ssh_password: Optional[str] = None
    ssh_key_path: Optional[str] = None
    
    # ADB Configuration
    use_adb: bool = False
    adb_device_id: Optional[str] = None
    adb_path: str = "adb"
    
    # Paths
    remote_work_dir: str = "/data/local/tmp"
    
    # Toolchain
    toolchain: ToolchainConfig = ToolchainConfig()
    
    # Legacy compiler field (keep for backward compatibility)
    compiler_cmd: str = "gcc"
    
    # Compute device selection (for OpenCL, CUDA, etc.)
    compute_backend: str = "auto"  # auto, opencl, cuda, native
    compute_device_type: str = "GPU"  # GPU, CPU, Accelerator
    compute_platform_id: int = 0
    compute_device_id: int = 0
    
    # Debugging
    enable_gdb: bool = False
    gdb_port: Optional[int] = None
    enable_profiling: bool = False
    
    # Environment
    env_vars: Dict[str, str] = {}
    
    # Mock mode
    mock: bool = False
