from main import preprocess, postprocess, model
import torch

img_path = "2-900x667 (1).jpg"
out_path = "test_output_local.png"
import os
if not os.path.exists(img_path):
    print(f"Input not found: {img_path} — usando placeholder.png")
    img_path = 'placeholder.png'

with open(img_path, 'rb') as f:
    content = f.read()

L_tensor, L_full, orig_size = preprocess(content)
with torch.no_grad():
    output = model(L_tensor)

result_img = postprocess(output, L_full, orig_size)
result_img.save(out_path)
print(f"Saved: {out_path}")
