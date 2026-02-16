import torch
from src.model import RNAFoldingModel

# 1. Initialize the model
model = RNAFoldingModel()
print("Model initialized successfully!")

# 2. Create dummy input (Batch Size=1, Sequence Length=20)
# Random integers between 1 and 5 representing RNA bases
dummy_input = torch.randint(1, 6, (1, 20))

# 3. Forward pass
output = model(dummy_input)

print(f"\nInput Shape: {dummy_input.shape}  (1 sequence of length 20)")
print(f"Output Shape: {output.shape} (Expect: 1, 20, 3)")

# 4. Check results
if output.shape == (1, 20, 3):
    print("\n✅ SUCCESS: Model produced (x, y, z) coordinates for every base!")
else:
    print("\n❌ ERROR: Output shape is wrong.")