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

# Configurazione modelli disponibili
AVAILABLE_MODELS = {
    "coco15k": "colorizer_finale_coco15k.pth",
    "final25k": "colorizer_finale25k.pth"
}

device = torch.device('cpu')
model = ColorizerResNet().to(device)
current_loaded_model = None

def load_model_weights(model_id: str):
    global current_loaded_model
    if current_loaded_model == model_id:
        return
    
    file_path = AVAILABLE_MODELS.get(model_id)
    if not file_path or not os.path.exists(file_path):
        # Fallback se il file specifico non esiste
        files = [f for f in os.listdir(".") if f.endswith(".pth")]
        if not files:
            raise Exception("Nessun file .pth trovato nella directory")
        file_path = files[0]
        print(f"Modello {model_id} non trovato, uso fallback: {file_path}")
    
    try:
        data = torch.load(file_path, map_location=device)
        state_dict = data.get('model_state_dict', data.get('state_dict', data))
        model.load_state_dict(state_dict)
        model.eval()
        current_loaded_model = model_id
        print(f"Modello caricato: {file_path}")
    except Exception as e:
        print(f"Errore caricamento {file_path}: {e}")
        raise e

# Caricamento iniziale
try:
    load_model_weights("coco15k")
except:
    try:
        load_model_weights("final25k")
    except:
        print("Attenzione: nessun modello caricato all'avvio.")

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
async def colorize(file: UploadFile = File(...), model_id: str = "coco15k"):
    try:
        # Carica i pesi del modello selezionato
        load_model_weights(model_id)
        
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
        return Response(content=f"Errore: {str(e)}", status_code=500)

@app.get("/models")
def get_models():
    return [
        {"id": "coco15k", "name": "COCO Dataset (15k)", "description": "Ottimizzato per scene varie e oggetti"},
        {"id": "final25k", "name": "Final Model (25k)", "description": "Modello bilanciato ad alta precisione"}
    ]

@app.get("/")
def home():
    return {"status": "ready", "current_model": current_loaded_model}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
