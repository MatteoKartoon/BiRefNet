import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from huggingface_hub import hf_hub_download
import os

# --- 1. BiRefNet Compatibility Fix (From your notebook) ---
import transformers.configuration_utils
original_getattribute = transformers.configuration_utils.PretrainedConfig.__getattribute__

def patched_getattribute(self, key):
    if key == 'is_encoder_decoder':
        return False
    return original_getattribute(self, key)

transformers.configuration_utils.PretrainedConfig.__getattribute__ = patched_getattribute

# --- 2. Configuration & Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. Load Model using Notebook Logic ---
def load_toonout_model():
    global MODEL
    print(f"Checking for weights in {MODELS_DIR}...")
    
    # Download specialized ToonOut weights
    weights_path = hf_hub_download(
        repo_id="joelseytre/toonout",
        filename="birefnet_finetuned_toonout.pth",
        local_dir=MODELS_DIR
    )
    
    print("Loading BiRefNet architecture (Notebook style)...")
    # This matches the notebook logic exactly
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", 
        trust_remote_code=True,
        cache_dir=os.path.join(MODELS_DIR, "architecture")
    )
    
    print("Applying weights...")
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # Clean up weight keys (from your notebook logic)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module._orig_mod."):
            clean_state_dict[k[len("module._orig_mod."):]] = v
        elif k.startswith("module."):
            clean_state_dict[k[len("module."):]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    model.to(DEVICE).float()
    model.eval()
    
    MODEL = model
    print(f"Model successfully loaded on {DEVICE}!")

# --- 4. Processing Function ---
def process_image(input_pil):
    if input_pil is None:
        return None
    
    if MODEL is None:
        load_toonout_model()

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    original_size = input_pil.size
    img_rgb = input_pil.convert('RGB')
    input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE).float()
    
    with torch.no_grad():
        preds = MODEL(input_tensor)[-1].sigmoid().cpu()
    
    mask = transforms.ToPILImage()(preds[0].squeeze())
    mask = mask.resize(original_size, Image.Resampling.LANCZOS)
    
    result = input_pil.convert("RGBA")
    result.putalpha(mask)
    return result

# --- 5. Gradio UI ---
with gr.Blocks(title="ToonOut Anime BG Remover") as demo:
    gr.Markdown("# 🎌 ToonOut: Anime Background Remover")
    gr.Markdown(f"Model and weights are saved in `./models/`. Running on: **{DEVICE.upper()}**")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Anime Image")
            submit_btn = gr.Button("Remove Background", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Result (Transparent)")

    submit_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image]
    )

if __name__ == "__main__":
    load_toonout_model()
    demo.launch(theme=gr.themes.Soft())