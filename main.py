import io
import torch
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import cv2
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3

        self.dec4 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128 + 128, 64)
        self.dec2 = DecoderBlock(64 + 64, 64)
        self.dec1 = DecoderBlock(64 + 64, 32)

        self.out = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Espansione a 3 canali per compatibilità con ResNet pre-addestrata
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

MODELS_DIR = "models"
AVAILABLE_MODELS = {
    "coco30k": os.path.join(MODELS_DIR, "colorizer_finale_coco30k.pth"),
    "gan": os.path.join(MODELS_DIR, "colorizer_finale_Gan.pth"),
    "skip": os.path.join(MODELS_DIR, "colorizer_finaleSkip.pth"),
    "final1888": os.path.join(MODELS_DIR, "colorizer_finale1888.pth")
}

device = torch.device('cpu')
model = ColorizerResNet().to(device)
current_loaded_model = None

def load_model_weights(model_id: str):
    global current_loaded_model
    if current_loaded_model == model_id:
        return
    
    file_path = AVAILABLE_MODELS.get(model_id)
    
    # Fallback al primo file .pth trovato se il percorso specificato non esiste
    if not file_path or not os.path.exists(file_path):
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            
        files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
        if not files:
            raise Exception(f"Nessun file .pth trovato nella directory {MODELS_DIR}")
        
        file_path = os.path.join(MODELS_DIR, files[0])
    
    try:
        data = torch.load(file_path, map_location=device)
        state_dict = data.get('model_state_dict', data.get('state_dict', data))
        model.load_state_dict(state_dict)
        model.eval()
        current_loaded_model = model_id
    except Exception as e:
        raise e

try:
    load_model_weights("coco30k")
except:
    try:
        load_model_weights("gan")
    except:
        pass

def preprocess(image_bytes):
    img_orig = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_size = img_orig.size
    
    img_256 = img_orig.resize((256, 256), Image.Resampling.LANCZOS)
    img_np_256 = np.array(img_256, dtype=np.float32) / 255.0
    lab_256 = rgb2lab(img_np_256).astype(np.float32)
    
    img_np_full = np.array(img_orig, dtype=np.float32) / 255.0
    lab_full = rgb2lab(img_np_full).astype(np.float32)
    L_full = lab_full[:, :, 0]
    
    # Normalizzazione coerente con i parametri di addestramento del notebook
    L_tensor = (lab_256[:, :, 0] / 50.0) - 1.0
    L_tensor = torch.tensor(L_tensor).unsqueeze(0).unsqueeze(0).to(device)
    
    return L_tensor, L_full, orig_size

def postprocess(output_tensor, L_full, orig_size):
    ab_pred = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Denormalizzazione dei canali ab nell'intervallo LAB standard (-128, 127)
    ab_pred = np.clip(ab_pred * 128.0, -128.0, 127.0)
    
    ab_full = cv2.resize(ab_pred, (orig_size[0], orig_size[1]), interpolation=cv2.INTER_LINEAR)
    
    lab_result = np.zeros((orig_size[1], orig_size[0], 3), dtype=np.float32)
    lab_result[:, :, 0] = L_full
    lab_result[:, :, 1:] = ab_full
    
    rgb_result = lab2rgb(lab_result)
    rgb_result = (np.clip(rgb_result, 0, 1) * 255).astype(np.uint8)
    
    return Image.fromarray(rgb_result)

@app.post("/colorize")
async def colorize(file: UploadFile = File(...), model_id: str = "coco30k"):
    try:
        print(f"[COLORIZE] Starting with model_id={model_id}")
        load_model_weights(model_id)
        print(f"[COLORIZE] Model loaded successfully")
        
        content = await file.read()
        print(f"[COLORIZE] File read, size={len(content)} bytes")
        
        L_tensor, L_full, orig_size = preprocess(content)
        print(f"[COLORIZE] Preprocessing done, tensor shape={L_tensor.shape}, orig_size={orig_size}")
        
        with torch.no_grad():
            output = model(L_tensor)
        print(f"[COLORIZE] Model inference done, output shape={output.shape}")
        
        result_img = postprocess(output, L_full, orig_size)
        print(f"[COLORIZE] Postprocessing done")
        
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        print(f"[COLORIZE] Image saved to bytes, size={len(img_byte_arr.getvalue())} bytes")
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    except Exception as e:
        print(f"[COLORIZE ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response(content=f"Errore: {str(e)}", status_code=500)

@app.get("/models")
def get_models():
    return [
        {"id": "coco30k", "name": "COCO Dataset (30k)", "description": "Massima precisione su scene complesse"},
        {"id": "gan", "name": "Generative Adversarial Network", "description": "Modello basato su GAN per risultati realistici"},
        {"id": "skip", "name": "Skip Connection Model", "description": "Architettura con skip connections avanzate"},
        {"id": "final1888", "name": "Finale 1888", "description": "Modello addestrato su 1888 immagini"}
    ]

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/status")
def status():
    return {"status": "ready", "current_model": current_loaded_model}

app.mount("/", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
