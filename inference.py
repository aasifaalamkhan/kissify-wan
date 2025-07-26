import os
import torch
import uuid
import gc
from PIL import Image
import json
from huggingface_hub import snapshot_download
import importlib.util
from contextlib import nullcontext

# Import individual components, NOT the pipeline
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import export_to_video, pil_to_video
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from utils import (
    load_face_images, crop_face
)

# --- Define constants ---
OUTPUT_DIR = "/workspace/outputs"
device = "cuda"
torch_dtype = torch.float16

# --- Helper function ---
def image_to_ascii(image, width=100):
    ASCII_CHARS = "@%#*+=-:. "
    aspect_ratio = image.height / image.width
    new_height = int(aspect_ratio * width * 0.55)
    resized_image = image.resize((width, new_height)).convert("L")
    pixels = resized_image.getdata()
    ascii_str = "".join([ASCII_CHARS[pixel * (len(ASCII_CHARS) - 1) // 255] for pixel in pixels])
    final_art = "\n".join([ascii_str[i:i+width] for i in range(0, len(ascii_str), width)])
    return f"\n--- Composite Image ASCII Preview ---\n{final_art}\n-------------------------------------\n"

# --- Manually Load Each Model Component ---
print("[INFO] Manually loading all model components...")
base_model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# 1. Load a working VAE (the model's is broken/incompatible)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch_dtype)

# 2. Load Tokenizer and Text Encoder
tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="text_encoder")
text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder", torch_dtype=torch_dtype)

# 3. Load Image Processor and a placeholder Image Encoder (we'll use processor only)
image_processor = CLIPImageProcessor.from_pretrained(base_model_id, subfolder="image_encoder")

# 4. Load the custom Transformer
print("[INFO] Loading custom Transformer code...")
model_path = snapshot_download(base_model_id, cache_dir=os.getenv("HF_HOME"), allow_patterns=["*.json", "*.py", "*.safetensors"])
transformer_code_path = os.path.join(model_path, "transformer", "modeling_wan_transformer.py")
spec = importlib.util.spec_from_file_location("modeling_wan_transformer", transformer_code_path)
custom_transformer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_transformer_module)
WanTransformer3DModel = custom_transformer_module.WanTransformer3DModel
transformer = WanTransformer3DModel.from_pretrained(base_model_id, subfolder="transformer", torch_dtype=torch_dtype)

# 5. Create a Scheduler
scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")

# 6. Move all to GPU
vae.to(device)
text_encoder.to(device)
transformer.to(device)

print("✅ All components loaded manually and ready.", flush=True)


# ========= Video Generation Logic (Manually Implemented) =========
@torch.no_grad()
def generate_kissing_video(input_data):
    """
    Main function to generate a video using manual component orchestration.
    """
    try:
        # Prepare inputs
        unique_id = str(uuid.uuid4())
        final_filename = f"{unique_id}_final.mp4"
        final_video_path = os.path.join(OUTPUT_DIR, final_filename)

        yield "🧠 Step 1/4: Loading and preparing images..."
        face_images_b64 = [input_data['face_image1'], input_data['face_image2']]
        pil_images = load_face_images(face_images_b64)

        face1_cropped = crop_face(pil_images[0]).resize((224, 224))
        face2_cropped = crop_face(pil_images[1]).resize((224, 224))

        composite_image = Image.new('RGB', (448, 224))
        composite_image.paste(face1_cropped, (0, 0))
        composite_image.paste(face2_cropped, (224, 0))
        print(image_to_ascii(composite_image))
        
        composite_filename = f"{unique_id}_composite.jpg"
        composite_image_path = os.path.join(OUTPUT_DIR, composite_filename)
        composite_image.save(composite_image_path)
        yield {'composite_filename': composite_filename}

        # Generation parameters
        prompt = "A man and a woman are embracing near a lake with mountains in the background. They gaze into each other's eyes, then they share a tender and passionate kissing. masterpiece, best quality, realistic, high resolution, cinematic film still"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, nsfw, text, watermark, logo, deformed, distorted, disfigured, cartoon, anime"
        
        num_frames = 16 # NOTE: Reduced for manual pipeline to manage memory/speed. Increase if needed.
        num_inference_steps = 50
        guidance_scale = 7.5
        height = 480
        width = 832

        yield f"🎨 Step 2/4: Generating {num_frames} frames of video..."
        
        # 1. Encode prompts
        text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
        
        uncond_inputs = tokenizer(negative_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]
        
        prompt_embeds = torch.cat([uncond_embeddings, text_embeddings])

        # 2. Preprocess image
        image = image_processor(images=composite_image, return_tensors="pt").pixel_values
        image_embeddings = vae.encode(image.to(device, dtype=torch_dtype)).latent_dist.sample()
        image_embeddings = image_embeddings * vae.config.scaling_factor
        image_embeddings = image_embeddings.unsqueeze(1).repeat(1, num_frames, 1, 1, 1).squeeze(0)

        # 3. Prepare scheduler and latents
        scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn((1, transformer.config.in_channels, num_frames, height // 8, width // 8), device=device, dtype=torch_dtype)

        # 4. Denoising loop
        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # Predict the noise residual
            noise_pred = transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                image_hidden_states=image_embeddings,
                timestep=t
            ).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 5. Decode latents
        video_latents = 1.0 / vae.config.scaling_factor * latents
        video_tensor = vae.decode(video_latents, return_dict=False)[0]
        video = (video_tensor / 2 + 0.5).clamp(0, 1)
        video = video.cpu().permute(0, 2, 3, 1).float().numpy()
        video_frames = pil_to_video(video)

        yield f"✅ Step 3/4: Finished generation. Total frames: {len(video_frames)}"
        yield "🚀 Step 4/4: Exporting video..."

        export_to_video(video_frames, final_video_path, fps=8)

        yield "✅ Post-processing finished."
        yield "✅ Done!"

        yield {"filename": final_filename}

    finally:
        yield "🧹 Cleaning up GPU memory..."
        gc.collect()
        torch.cuda.empty_cache()