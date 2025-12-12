"""Device discovery for compute devices (OpenCL, CUDA, etc.)"""
from typing import Dict, List, Any
import subprocess
import os


class DeviceDiscovery:
    """Discovers available compute devices on the system"""
    
    @staticmethod
    def discover_all_devices() -> Dict[str, List[Dict[str, Any]]]:
        """Discover all available compute devices"""
        return {
            "opencl": DeviceDiscovery._discover_opencl_devices(),
            "cuda": DeviceDiscovery._discover_cuda_devices(),
            "host_cpu": DeviceDiscovery._discover_host_cpu()
        }
    
    @staticmethod
    def _discover_opencl_devices() -> List[Dict[str, Any]]:
        """Discover OpenCL devices"""
        devices = []
        
        # Try to use clinfo if available
        try:
            result = subprocess.run(
                ["clinfo", "--list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    if line.strip():
                        devices.append({
                            "id": i,
                            "name": line.strip(),
                            "type": "OpenCL"
                        })
        except FileNotFoundError:
            # clinfo not installed
            pass
        except Exception:
            pass
        
        return devices
    
    @staticmethod
    def _discover_cuda_devices() -> List[Dict[str, Any]]:
        """Discover CUDA devices"""
        devices = []
        
        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = line.split(",")
                        devices.append({
                            "id": i,
                            "name": parts[0].strip() if parts else "Unknown GPU",
                            "memory": parts[1].strip() if len(parts) > 1 else "Unknown",
                            "type": "CUDA"
                        })
        except FileNotFoundError:
            # nvidia-smi not installed
            pass
        except Exception:
            pass
        
        return devices
    
    @staticmethod
    def _discover_host_cpu() -> List[Dict[str, Any]]:
        """Discover host CPU info"""
        import platform
        
        cpu_info = {
            "id": 0,
            "name": platform.processor() or "Unknown CPU",
            "type": "CPU",
            "architecture": platform.machine(),
            "cores": os.cpu_count() or 1
        }
        
        return [cpu_info]


