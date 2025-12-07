import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Simple linear layer: 10 inputs -> 5 outputs
        self.linear = nn.Linear(10, 5, bias=False)
        
        # Initialize with identity-like weights for predictable output
        with torch.no_grad():
            self.linear.weight.data = torch.ones(5, 10) * 0.5
    
    def forward(self, x):
        return self.linear(x)

# Create model instance at module level so it can be imported
model = SimpleModel()
model.eval()

# For testing directly
if __name__ == "__main__":
    # Test with sample input
    x = torch.randn(1, 10)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
