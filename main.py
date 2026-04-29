import io
import torch
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import cv2
import os

app = FastAPI()

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Definizione Architettura Modello ---

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.block(x)

class ColorizerResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)

        # ENCODER
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3

        # DECODER
        self.dec4 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128 + 128, 64)
        self.dec2 = DecoderBlock(64 + 64, 64)
        self.dec1 = DecoderBlock(64 + 64, 32)

        self.out = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return self.out(d1)

# --- Caricamento Modello ---

# Cerchiamo il file preferito `colorizer_finale_coco15k.pth`, poi `colorizer_finale25k.pth`, poi `model.pth`
if os.path.exists("colorizer_finale_coco15k.pth"):
    MODEL_PATH = "colorizer_finale_coco15k.pth"
elif os.path.exists("colorizer_finale25k.pth"):
    MODEL_PATH = "colorizer_finale25k.pth"
else:
    MODEL_PATH = "model.pth"

device = torch.device('cpu')
model = ColorizerResNet().to(device)

try:
    data = torch.load(MODEL_PATH, map_location=device)
    # Support common checkpoint formats
    if isinstance(data, dict):
        if 'model_state_dict' in data:
            state_dict = data['model_state_dict']
        elif 'state_dict' in data:
            state_dict = data['state_dict']
        else:
            # assume it's already a state_dict-like mapping
            state_dict = data
    else:
        state_dict = data

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Modello caricato da: {MODEL_PATH}")
except Exception as e:
    print(f"Errore caricamento pesi da {MODEL_PATH}: {e}")

# --- Processing Fedele al Notebook ---

def preprocess(image_bytes):
    img_orig = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_size = img_orig.size
    
    # Versione 256 per il modello
    img_256 = img_orig.resize((256, 256), Image.Resampling.LANCZOS)
    img_np_256 = np.array(img_256, dtype=np.float32) / 255.0
    lab_256 = rgb2lab(img_np_256).astype(np.float32)
    
    # L originale ad alta risoluzione
    img_np_full = np.array(img_orig, dtype=np.float32) / 255.0
    lab_full = rgb2lab(img_np_full).astype(np.float32)
    L_full = lab_full[:, :, 0]
    
    # Normalizzazione come nel notebook: (L / 50) - 1
    L_tensor = (lab_256[:, :, 0] / 50.0) - 1.0
    L_tensor = torch.tensor(L_tensor).unsqueeze(0).unsqueeze(0).to(device)
    
    return L_tensor, L_full, orig_size

def postprocess(output_tensor, L_full, orig_size):
    # 1. Recupero canali ab (-1 a 1)
    ab_pred = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # 2. Denormalizzazione standard dei canali ab (-128, 127)
    ab_pred = np.clip(ab_pred * 128.0, -128.0, 127.0)
    
    # 3. Resize alla risoluzione originale
    ab_full = cv2.resize(ab_pred, (orig_size[0], orig_size[1]), interpolation=cv2.INTER_LINEAR)
    
    # 4. Unione con L-full
    lab_result = np.zeros((orig_size[1], orig_size[0], 3), dtype=np.float32)
    lab_result[:, :, 0] = L_full
    lab_result[:, :, 1:] = ab_full
    
    # 5. Conversione finale
    rgb_result = lab2rgb(lab_result)
    rgb_result = (np.clip(rgb_result, 0, 1) * 255).astype(np.uint8)
    
    return Image.fromarray(rgb_result)

@app.post("/colorize")
async def colorize(file: UploadFile = File(...)):
    try:
        content = await file.read()
        L_tensor, L_full, orig_size = preprocess(content)
        with torch.no_grad():
            output = model(L_tensor)
        result_img = postprocess(output, L_full, orig_size)
        
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/")
def home():
    return {"status": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
