import torch
import os
from main import ColorizerResNet

try:
    model = ColorizerResNet()
    path = os.path.join("models", "colorizer_finaleSkip.pth")
    data = torch.load(path, map_location='cpu')
    state_dict = data.get('model_state_dict', data.get('state_dict', data))
    model.load_state_dict(state_dict)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
