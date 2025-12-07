import torch
import torch.nn as nn
import importlib.util
import os
import inspect
from typing import Dict, Any, List, Optional

class ComparisonEngine:
    def __init__(self):
        self.hooks = []
        self.activations = {}

    def load_model_from_path(self, file_path: str, model_name: str = "model"):
        """
        Loads a PyTorch model from a python file.
        Tries multiple strategies:
        1. Look for 'model' variable
        2. Look for any nn.Module instance
        3. Look for nn.Module classes and try to instantiate them
        """
        spec = importlib.util.spec_from_file_location(f"user_module_{model_name}", file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Strategy 1: Look for 'model' variable
            if hasattr(module, "model"):
                return module.model, module
            
            # Strategy 2: Look for any nn.Module instance in the module
            for name in dir(module):
                if not name.startswith('_'):  # Skip private attributes
                    obj = getattr(module, name)
                    if isinstance(obj, nn.Module):
                        print(f"Found model instance: {name}")
                        return obj, module
            
            # Strategy 3: Look for nn.Module classes and try to instantiate them
            for name in dir(module):
                if not name.startswith('_'):  # Skip private attributes
                    obj = getattr(module, name)
                    if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj != nn.Module:
                        try:
                            print(f"Found model class: {name}, attempting to instantiate...")
                            model_instance = obj()  # Try instantiating with no args
                            return model_instance, module
                        except Exception as e:
                            print(f"Could not instantiate {name}: {e}")
                            continue
            
            raise ValueError(f"Could not find 'model' variable or nn.Module class/instance in {file_path}. "
                           f"Please define a variable named 'model' or ensure your model class can be instantiated without arguments.")
        raise ValueError(f"Could not load module from {file_path}")

    def register_hooks(self, model: nn.Module, prefix: str):
        """
        Registers forward hooks to capture output of every leaf layer or specific layers.
        """
        self.activations[prefix] = {}
        
        def get_hook(name):
            def hook(model, input, output):
                # Detach and move to CPU for safety
                if isinstance(output, torch.Tensor):
                    self.activations[prefix][name] = output.detach().cpu().numpy()
                else:
                    # tuple or other struct
                    self.activations[prefix][name] = output  # Handle complex outputs later
            return hook

        for name, layer in model.named_modules():
            # You might want to filter only leaf layers or specific types
            # For now, trace everything that is a module
            self.hooks.append(layer.register_forward_hook(get_hook(name)))

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}

    def compare(self, source_path: str, target_path: str, input_shape: List[int]):
        """
        Runs both models with random input (or provided input) and compares.
        """
        try:
            model_src, _ = self.load_model_from_path(source_path, "source")
            model_tgt, _ = self.load_model_from_path(target_path, "target")
            
            model_src.eval()
            model_tgt.eval()
            
            self.clear_hooks()
            self.register_hooks(model_src, "source")
            self.register_hooks(model_tgt, "target")
            
            dummy_input = torch.randn(*input_shape)
            
            with torch.no_grad():
                _ = model_src(dummy_input)
                _ = model_tgt(dummy_input)
                
            # Compare
            results = []
            all_keys = set(self.activations["source"].keys()) | set(self.activations["target"].keys())
            
            for key in sorted(all_keys):
                val_src = self.activations["source"].get(key)
                val_tgt = self.activations["target"].get(key)
                
                res = {
                    "layer_name": key,
                    "in_source": key in self.activations["source"],
                    "in_target": key in self.activations["target"],
                    "match": False,
                    "l2_diff": None,
                    "shape_src": str(val_src.shape) if hasattr(val_src, "shape") else str(type(val_src)),
                    "shape_tgt": str(val_tgt.shape) if hasattr(val_tgt, "shape") else str(type(val_tgt)),
                }
                
                if res["in_source"] and res["in_target"]:
                    # Compute difference if they are numpy arrays
                    import numpy as np
                    if isinstance(val_src, np.ndarray) and isinstance(val_tgt, np.ndarray):
                        if val_src.shape == val_tgt.shape:
                            diff = np.linalg.norm(val_src - val_tgt)
                            res["l2_diff"] = float(diff)
                            res["match"] = bool(diff < 1e-5)
                        else:
                            res["match"] = False # Shape mismatch
                
                results.append(res)
                
            return results
        finally:
            self.clear_hooks()
