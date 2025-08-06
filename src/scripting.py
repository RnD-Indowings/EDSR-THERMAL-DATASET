import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from types import SimpleNamespace
from model.edsr import EDSR

# === CONFIGURATION ===
input_folder = "/home/nisha-/EDSR-PyTorch/thermal_input"
output_folder = "/home/nisha-/EDSR-PyTorch/thermal_output"

model_weights_path = "/home/nisha-/EDSR-PyTorch/weights/edsr_baseline_x4.pt"
target_resolution = (1280, 720)

# === DEVICE ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === CREATE OUTPUT FOLDER IF NOT EXISTS ===
os.makedirs(output_folder, exist_ok=True)

# === DEFINE MODEL ARGS ===
args = SimpleNamespace(
    scale=[4],
    n_resblocks=16,
    n_feats=64,
    res_scale=1.0,
    rgb_range=255,
    n_colors=3        # 👈 Add this line!
    
)

# === LOAD MODEL ===
model = EDSR(args)
state_dict = torch.load(model_weights_path, map_location=device)
model.load_state_dict(state_dict)
model.eval().to(device)

# === UTILITY FUNCTIONS ===
def preprocess(img):
    tensor = transforms.ToTensor()(img).unsqueeze(0) * 255
    return tensor.to(device)

def postprocess(tensor):
    img = tensor.squeeze(0).clamp(0, 255).cpu() / 255.0
    return transforms.ToPILImage()(img)

# === PROCESS EACH IMAGE ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load thermal image and convert to RGB
        img = Image.open(input_path).convert('RGB')

        # Upscale using EDSR
        with torch.no_grad():
            input_tensor = preprocess(img)
            sr_tensor = model(input_tensor)
            sr_image = postprocess(sr_tensor)

        # Resize to 1280x720
        final_image = sr_image.resize(target_resolution, Image.BICUBIC)

        # Save result
        final_image.save(output_path)
        print(f"[INFO] Saved resized image: {output_path}")
