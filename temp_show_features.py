import numpy as np
import sys
import os
import json

filename = 'enhanced_features/mee_admin_1.wav.npz'
if len(sys.argv) > 1:
    filename = sys.argv[1]

print(f"Loading features from: {filename}")
data = np.load(filename, allow_pickle=True)
print(f"Keys: {data.files}")

for key in data.files:
    arr = data[key]
    if hasattr(arr, 'shape'):
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        
        # For object arrays, extract the object and inspect it
        if arr.dtype == np.dtype('O'):
            obj = arr.item()
            if isinstance(obj, dict):
                print("  Dictionary with keys:", list(obj.keys()))
                for k, v in obj.items():
                    if hasattr(v, 'shape'):
                        print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
                        if len(v.shape) == 1 and v.shape[0] < 10:
                            print(f"    Values: {v}")
                        elif len(v.shape) == 1:
                            print(f"    First 5 values: {v[:5]}")
                        elif len(v.shape) == 2:
                            print(f"    First 2x2 corner: {v[:2,:2]}")
                        elif len(v.shape) == 3:
                            print(f"    First corner: {v[0,0,0]}")
                    else:
                        print(f"  - {k}: {v}")
            else:
                print(f"  Object of type: {type(obj)}")
        # Print a sample of the data for arrays
        elif len(arr.shape) == 1 and arr.shape[0] < 100:
            print(f"  Values: {arr}")
        elif len(arr.shape) == 1:
            print(f"  First 5 values: {arr[:5]}")
        elif len(arr.shape) == 2:
            print(f"  First 2x2 corner: {arr[:2,:2]}")
        elif len(arr.shape) == 3:
            print(f"  First corner: {arr[0,0,0]}")
    else:
        print(f"{key}: {arr} (scalar)") 