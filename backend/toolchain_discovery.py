"""Toolchain discovery for cross-compilation support.

Frontend expects discovered toolchains to use the schema:
  - name
  - compiler_path
  - sysroot
  - include_paths
  - library_paths
  - architecture
  - abi
"""
import os
import subprocess
import platform
from typing import Dict, List, Optional, Any
import re


class ToolchainDiscovery:
    """Discovers available toolchains on the host system"""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
    
    def discover_all(self) -> Dict[str, Any]:
        """Discover all available toolchains"""
        return {
            "host_info": {
                "os": self.system,
                "architecture": self.machine
            },
            "toolchains": {
                "native": self._discover_native_toolchains(),
                "cross": self._discover_cross_toolchains(),
                "android_ndk": self._discover_android_ndk(),
                "embedded": []
            },
            "gpu_sdks": self.discover_gpu_sdks(),
            "ndk": self._find_ndk()
        }
    
    def _discover_native_toolchains(self) -> List[Dict[str, Any]]:
        """Discover native compilers (gcc, clang, etc.)"""
        toolchains = []
        
        # Check for gcc
        gcc_path = self._which("gcc")
        if gcc_path:
            toolchains.append({
                "name": "GCC",
                "compiler_path": gcc_path,
                "version": self._get_compiler_version(gcc_path),
                "sysroot": None,
                "include_paths": [],
                "library_paths": [],
                "architecture": self.machine,
                "abi": None
            })
        
        # Check for clang
        clang_path = self._which("clang")
        if clang_path:
            toolchains.append({
                "name": "Clang",
                "compiler_path": clang_path,
                "version": self._get_compiler_version(clang_path),
                "sysroot": None,
                "include_paths": [],
                "library_paths": [],
                "architecture": self.machine,
                "abi": None
            })
        
        return toolchains
    
    def _discover_cross_toolchains(self) -> List[Dict[str, Any]]:
        """Discover cross-compilation toolchains"""
        toolchains = []
        
        # Check for common cross compilers
        cross_prefixes = [
            "aarch64-linux-gnu-",
            "arm-linux-gnueabihf-",
            "x86_64-linux-gnu-",
        ]
        
        for prefix in cross_prefixes:
            gcc_path = self._which(f"{prefix}gcc")
            if gcc_path:
                toolchains.append({
                    "name": f"{prefix}gcc",
                    "compiler_path": gcc_path,
                    "target": prefix.rstrip("-"),
                    "sysroot": None,
                    "include_paths": [],
                    "library_paths": [],
                    "architecture": prefix.rstrip("-"),
                    "abi": None
                })
        
        return toolchains
    
    def _discover_android_ndk(self) -> List[Dict[str, Any]]:
        """Discover Android NDK toolchains (clang wrappers per ABI/API)."""
        ndk_info = self._find_ndk()
        if ndk_info:
            ndk_path = ndk_info.get("path")
            if ndk_path:
                return self._discover_ndk_toolchains(ndk_path, ndk_info.get("version"))
        return []

    def _discover_ndk_toolchains(self, ndk_path: str, ndk_version: Optional[str]) -> List[Dict[str, Any]]:
        """Enumerate Android NDK clang wrapper toolchains from an installed NDK."""
        prebuilt_root = os.path.join(ndk_path, "toolchains", "llvm", "prebuilt")
        if not os.path.isdir(prebuilt_root):
            return []

        host_tag = self._select_ndk_host_tag(prebuilt_root)
        if not host_tag:
            return []

        prebuilt_dir = os.path.join(prebuilt_root, host_tag)
        bin_dir = os.path.join(prebuilt_dir, "bin")
        sysroot = os.path.join(prebuilt_dir, "sysroot")
        if not os.path.isdir(bin_dir) or not os.path.isdir(sysroot):
            return []

        # Collect available clang wrapper compilers.
        wrappers = []
        try:
            for fname in os.listdir(bin_dir):
                if fname.endswith("-clang") and "android" in fname:
                    wrappers.append(fname)
        except OSError:
            return []

        # Map arch key -> list of available API levels.
        arch_to_apis: Dict[str, List[int]] = {}
        patterns = [
            re.compile(r"^(aarch64)-linux-android(\d+)-clang$"),
            re.compile(r"^(x86_64)-linux-android(\d+)-clang$"),
            re.compile(r"^(i686)-linux-android(\d+)-clang$"),
            re.compile(r"^(armv7a)-linux-androideabi(\d+)-clang$"),
        ]
        for w in wrappers:
            for pat in patterns:
                m = pat.match(w)
                if not m:
                    continue
                arch = m.group(1)
                api = int(m.group(2))
                arch_to_apis.setdefault(arch, []).append(api)
                break

        if not arch_to_apis:
            return []

        # Pick a sensible API level per arch (prefer 21 when present).
        preferred_api = {
            "aarch64": 21,
            "x86_64": 21,
            "i686": 21,
            "armv7a": 21,
        }

        def pick_api(arch: str, apis: List[int]) -> int:
            apis_sorted = sorted(set(apis))
            pref = preferred_api.get(arch)
            if pref in apis_sorted:
                return pref
            # Fall back to the highest available (usually most compatible with modern NDKs)
            return apis_sorted[-1]

        toolchains: List[Dict[str, Any]] = []
        arch_meta = {
            "aarch64": {"architecture": "aarch64", "abi": "arm64-v8a", "syslib_triple": "aarch64-linux-android"},
            "armv7a": {"architecture": "arm", "abi": "armeabi-v7a", "syslib_triple": "arm-linux-androideabi"},
            "x86_64": {"architecture": "x86_64", "abi": "x86_64", "syslib_triple": "x86_64-linux-android"},
            "i686": {"architecture": "x86", "abi": "x86", "syslib_triple": "i686-linux-android"},
        }

        for arch, apis in arch_to_apis.items():
            meta = arch_meta.get(arch)
            if not meta:
                continue
            api = pick_api(arch, apis)

            if arch == "armv7a":
                wrapper_name = f"armv7a-linux-androideabi{api}-clang"
            else:
                wrapper_name = f"{arch}-linux-android{api}-clang"

            compiler_path = os.path.join(bin_dir, wrapper_name)
            if not os.path.isfile(compiler_path):
                continue

            # Arch-specific sysroot libs exist under sysroot/usr/lib/<triple>/<api>
            libdir = os.path.join(sysroot, "usr", "lib", meta["syslib_triple"], str(api))
            include_dir = os.path.join(sysroot, "usr", "include")

            toolchains.append({
                "name": f"Android NDK Clang ({meta['abi']}, API {api})",
                "compiler_path": compiler_path,
                "sysroot": sysroot,
                "include_paths": [include_dir] if os.path.isdir(include_dir) else [],
                "library_paths": [libdir] if os.path.isdir(libdir) else [],
                "architecture": meta["architecture"],
                "abi": meta["abi"],
                "ndk_path": ndk_path,
                "ndk_version": ndk_version,
                "ndk_host_tag": host_tag,
                "android_api": api,
            })

        return toolchains

    def _select_ndk_host_tag(self, prebuilt_root: str) -> Optional[str]:
        """Choose the correct prebuilt host tag directory inside the NDK."""
        try:
            tags = [d for d in os.listdir(prebuilt_root) if os.path.isdir(os.path.join(prebuilt_root, d))]
        except OSError:
            return None

        if not tags:
            return None

        system = self.system.lower()
        machine = (self.machine or "").lower()

        preferred = None
        if system == "darwin":
            preferred = "darwin-arm64" if machine in ("arm64", "aarch64") else "darwin-x86_64"
        elif system == "linux":
            preferred = "linux-x86_64"
        elif system in ("windows", "msys", "cygwin"):
            preferred = "windows-x86_64"

        if preferred and preferred in tags:
            return preferred

        # Fallback: choose a deterministic tag.
        return sorted(tags)[0]
    
    def _find_ndk(self) -> Optional[Dict[str, Any]]:
        """Find Android NDK installation"""
        # Check common NDK locations
        ndk_paths = [
            os.environ.get("ANDROID_NDK_HOME"),
            os.environ.get("ANDROID_NDK"),
            os.environ.get("NDK_HOME"),
            os.path.expanduser("~/Android/Sdk/ndk"),
            os.path.expanduser("~/Library/Android/sdk/ndk"),
            "/opt/android-ndk",
        ]
        
        for path in ndk_paths:
            if path and os.path.isdir(path):
                # Check if it's an NDK directory or contains NDK versions
                if os.path.exists(os.path.join(path, "ndk-build")):
                    return {"path": path, "type": "ndk"}
                
                # Check for versioned subdirectories
                try:
                    versions = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                    if versions:
                        latest = sorted(versions)[-1]
                        ndk_path = os.path.join(path, latest)
                        if os.path.exists(os.path.join(ndk_path, "ndk-build")):
                            return {"path": ndk_path, "version": latest, "type": "ndk"}
                except OSError:
                    pass
        
        return None
    
    def discover_gpu_sdks(self) -> Dict[str, Any]:
        """Discover GPU SDKs (CUDA, OpenCL, etc.)"""
        sdks = {}
        
        # Check for CUDA
        cuda_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_path and os.path.isdir(cuda_path):
            sdks["cuda"] = {
                "path": cuda_path,
                "available": True
            }
        
        # Check for OpenCL headers
        opencl_info = self._find_opencl()
        if opencl_info:
            sdks["opencl"] = opencl_info
        
        return sdks
    
    def _find_opencl(self) -> Optional[Dict[str, Any]]:
        """Find OpenCL installation"""
        # Check common OpenCL header locations
        header_paths = [
            "/usr/include/CL",
            "/usr/local/include/CL",
            "/opt/homebrew/include/CL",
        ]
        
        for path in header_paths:
            if os.path.exists(os.path.join(path, "cl.h")):
                return {
                    "headers_path": os.path.dirname(path),
                    "headers_source": "system",
                    "available_devices": []
                }
        
        return None
    
    def pull_opencl_from_devices(self) -> List[Dict[str, Any]]:
        """Pull OpenCL libraries from connected devices"""
        # Placeholder for device-specific OpenCL discovery
        return []
    
    def _which(self, cmd: str) -> Optional[str]:
        """Find command in PATH"""
        try:
            result = subprocess.run(
                ["which", cmd],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _get_compiler_version(self, path: str) -> Optional[str]:
        """Get compiler version string"""
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.split("\n")[0]
        except Exception:
            pass
        return None


class ADBDiscovery:
    """Discover ADB and connected Android devices"""
    
    @staticmethod
    def find_adb() -> Optional[str]:
        """Find adb executable"""
        # Check PATH first
        try:
            result = subprocess.run(
                ["which", "adb"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        # Check common locations
        adb_paths = [
            os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
            os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
            "/opt/android-sdk/platform-tools/adb",
        ]
        
        for path in adb_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        
        return None
    
    @staticmethod
    def get_adb_info(adb_path: str) -> Dict[str, Any]:
        """Get ADB version and connected devices"""
        info = {"version": None, "devices": []}
        
        try:
            # Get version
            result = subprocess.run(
                [adb_path, "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                info["version"] = result.stdout.split("\n")[0]
            
            # Get devices
            result = subprocess.run(
                [adb_path, "devices", "-l"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                for line in lines:
                    if line.strip() and "device" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            device = {"id": parts[0], "state": parts[1]}
                            # Parse additional info
                            for part in parts[2:]:
                                if ":" in part:
                                    key, value = part.split(":", 1)
                                    device[key] = value
                            info["devices"].append(device)
        except Exception:
            pass
        
        return info


