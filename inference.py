import os
import torch
import uuid
import gc
from PIL import Image
from diffusers import I2VTransformerPipeline
from diffusers.utils import export_to_video

from utils import (
    load_face_images, crop_face
)

# --- Define constants ---
OUTPUT_DIR = "/workspace/outputs"


# --- Load Models (NEW I2V PIPELINE) ---
print("[INFO] Initializing new I2V pipeline...", flush=True)
device = "cuda"

# The powerful Image-to-Video base model
base_model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

pipe = I2VTransformerPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

# --- Load BOTH compatible LoRAs ---
print("[INFO] Loading and combining two Kissing LoRAs...", flush=True)
# Load the first LoRA for motion dynamics
pipe.load_lora_weights("ighoshsubho/Wan-I2V-LoRA-Kiss", weight_name="i2v-custom-lora.safetensors", adapter_name="motion")
# Load the second LoRA for style and the specific trigger
pipe.load_lora_weights("Remade-AI/kissing", adapter_name="style")

# --- NEW: Set adapter weights based on model card recommendations ---
# 'motion' (ighoshsubho) at 0.9 and 'style' (Remade-AI) at 1.0
pipe.set_adapters(["motion", "style"], adapter_weights=[0.9, 1.0])


print("✅ All models and LoRAs are loaded and ready.", flush=True)


# ========= Video Generation Logic (REFINED WITH NEW SETTINGS) =========
def generate_kissing_video(input_data):
    """
    Main function to generate a video using the refined I2V pipeline with
    optimized settings based on model documentation.
    """
    try:
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

        composite_filename = f"{unique_id}_composite.jpg"
        composite_image_path = os.path.join(OUTPUT_DIR, composite_filename)
        composite_image.save(composite_image_path)
        yield {'composite_filename': composite_filename}

        # --- NEW: Highly descriptive prompt using recommended structure and trigger words ---
        prompt = "A man and a woman are embracing near a lake with mountains in the background. They gaze into each other's eyes, then they share a tender and passionate k144ing kissing. masterpiece, best quality, realistic, high resolution, cinematic film still"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, nsfw, text, watermark, logo, deformed, distorted, disfigured, cartoon, anime"
        
        # --- NEW: Generation parameters aligned with LoRA documentation ---
        num_frames = 65
        num_inference_steps = 50
        guidance_scale = 6.0 # Changed from 7.5 to recommended 6.0

        yield f"🎨 Step 2/4: Generating {num_frames} frames of video..."
        
        with torch.inference_mode():
            video_frames = pipe(
                prompt=prompt,
                image=composite_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                negative_prompt=negative_prompt,
                target_fps=24
            ).frames[0]

        yield f"✅ Step 3/4: Finished generation. Total frames: {len(video_frames)}"
        yield "🚀 Step 4/4: Exporting video..."

        export_to_video(video_frames, final_video_path, fps=24)

        yield "✅ Post-processing finished."
        yield "✅ Done!"

        yield {"filename": final_filename}

    finally:
        yield "🧹 Cleaning up GPU memory..."
        gc.collect()
        torch.cuda.empty_cache()