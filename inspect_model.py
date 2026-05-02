import torch
import collections

def inspect_model(path):
    print(f"Inspecting: {path}")
    try:
        data = torch.load(path, map_location='cpu')
        
        if isinstance(data, collections.OrderedDict) or (isinstance(data, dict) and 'state_dict' in data):
            print("Detected: state_dict (weights only)")
            if isinstance(data, dict):
                print(f"Keys in dict: {data.keys()}")
            else:
                print(f"Number of layers: {len(data)}")
        elif isinstance(data, torch.nn.Module):
            print("Detected: Full PyTorch Model")
            print(f"Architecture:\n{data}")
        else:
            print(f"Detected: Unknown format ({type(data)})")
            if hasattr(data, 'keys'):
                print(f"Keys: {data.keys()}")
                
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    inspect_model("models/model.pth")
