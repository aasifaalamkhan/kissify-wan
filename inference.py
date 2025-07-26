import os
import torch
import uuid
import gc
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from utils import (
    load_face_images, crop_face
)

# --- Define constants ---
OUTPUT_DIR = "/workspace/outputs"

# --- Helper function to display image as ASCII art in the terminal ---
def image_to_ascii(image, width=100):
    """Converts a PIL Image to an ASCII string representation."""
    ASCII_CHARS = "@%#*+=-:. "
    aspect_ratio = image.height / image.width
    new_height = int(aspect_ratio * width * 0.55)
    resized_image = image.resize((width, new_height)).convert("L")
    pixels = resized_image.getdata()
    ascii_str = "".join([ASCII_CHARS[pixel * (len(ASCII_CHARS) - 1) // 255] for pixel in pixels])
    final_art = "\n".join([ascii_str[i:i+width] for i in range(0, len(ascii_str), width)])
    return f"\n--- Composite Image ASCII Preview ---\n{final_art}\n-------------------------------------\n"

# --- Load Models ---
print("[INFO] Initializing I2V pipeline...", flush=True)
base_model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# Use DiffusionPipeline to automatically load the model's custom pipeline code
# Use device_map="balanced" and low_cpu_mem_usage=True to handle the large model size
pipe = DiffusionPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="balanced",
    low_cpu_mem_usage=True
)

# --- Load BOTH compatible LoRAs ---
print("[INFO] Loading and combining two Kissing LoRAs...", flush=True)
pipe.load_lora_weights("ighoshsubho/Wan-I2V-LoRA-Kiss", weight_name="i2v-custom-lora.safetensors", adapter_name="motion")
pipe.load_lora_weights("Remade-AI/kissing", adapter_name="style")

# --- Set adapter weights ---
pipe.set_adapters(["motion", "style"], adapter_weights=[0.9, 1.0])

print("✅ All models and LoRAs are loaded and ready.", flush=True)

# ========= Video Generation Logic =========
def generate_kissing_video(input_data):
    """
    Main function to generate a video.
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

        print(image_to_ascii(composite_image))

        composite_filename = f"{unique_id}_composite.jpg"
        composite_image_path = os.path.join(OUTPUT_DIR, composite_filename)
        composite_image.save(composite_image_path)
        yield {'composite_filename': composite_filename}

        prompt = "A man and a woman are embracing near a lake with mountains in the background. They gaze into each other's eyes, then they share a tender and passionate kissing. masterpiece, best quality, realistic, high resolution, cinematic film still"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, nsfw, text, watermark, logo, deformed, distorted, disfigured, cartoon, anime"
        
        num_frames = 65
        num_inference_steps = 50
        guidance_scale = 6.0

        yield f"🎨 Step 2/4: Generating {num_frames} frames of video..."

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