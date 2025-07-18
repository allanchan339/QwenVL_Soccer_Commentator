# models.py
"""
Handles all model loading and device setup for the app.
"""
import torch
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel
from musetalk.utils.utils import load_all_model

def load_all_models(use_float16: bool = False, unet_model_path: str = "./models/musetalkV15/unet.pth", vae_type: str = "sd-vae", unet_config: str = "./models/musetalkV15/musetalk.json", whisper_dir: str = "./models/whisper"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type=vae_type,
        unet_config=unet_config,
        device=device
    )
    if use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)
    timesteps = torch.tensor([0], device=device)
    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    whisper = WhisperModel.from_pretrained(whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    return device, vae, unet, pe, weight_dtype, audio_processor, whisper, timesteps 