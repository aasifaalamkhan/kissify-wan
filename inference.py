import os
import torch
import uuid
import gc
from PIL import Image
# Corrected the import statement below
from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_video

from utils import (
    load_face_images, crop_face
)

# --- Define constants ---
OUTPUT_DIR = "/workspace/outputs"


# --- NEW: Helper function to display image as ASCII art in the terminal ---
def image_to_ascii(image, width=100):
    """Converts a PIL Image to an ASCII string representation."""
    # ASCII characters used to build the output text
    ASCII_CHARS = "@%#*+=-:. "

    # Resize the image and convert to grayscale
    aspect_ratio = image.height / image.width
    new_height = int(aspect_ratio * width * 0.55) # 0.55 corrects for non-square character cells
    resized_image = image.resize((width, new_height)).convert("L")

    # Get pixel data
    pixels = resized_image.getdata()

    # Map each pixel to an ASCII character
    ascii_str = ""
    for pixel_value in pixels:
        # Normalize pixel value to the range of ASCII_CHARS
        ascii_str += ASCII_CHARS[pixel_value * (len(ASCII_CHARS) - 1) // 255]

    # Format as a multi-line string
    final_art = ""
    for i in range(0, len(ascii_str), width):
        final_art += ascii_str[i:i+width] + "\n"

    return f"\n--- Composite Image ASCII Preview ---\n{final_art}-------------------------------------\n"


# --- Load Models (NEW I2V PIPELINE) ---
print("[INFO] Initializing new I2V pipeline...", flush=True)
device = "cuda"

# The powerful Image-to-Video base model
base_model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# Corrected the class name from I2VTransformerPipeline to I2VGenXLPipeline
pipe = I2VGenXLPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16
    # REMOVED: variant="fp16" - This was causing the error
).to(device)

# --- Load BOTH compatible LoRAs ---
print("[INFO] Loading and combining two Kissing LoRAs...", flush=True)
# Load the first LoRA for motion dynamics
pipe.load_lora_weights("ighoshsubho/Wan-I2V-LoRA-Kiss", weight_name="i2v-custom-lora.safetensors", adapter_name="motion")
# Load the second LoRA for style and the specific trigger
pipe.load_lora_weights("Remade-AI/kissing", adapter_name="style")

# --- Set adapter weights based on model card recommendations ---
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

        # --- NEW: Print the ASCII art preview to the SSH console log ---
        # This will not be sent to the GUI, it will only appear in the server log.
        print(image_to_ascii(composite_image))

        composite_filename = f"{unique_id}_composite.jpg"
        composite_image_path = os.path.join(OUTPUT_DIR, composite_filename)
        composite_image.save(composite_image_path)
        yield {'composite_filename': composite_filename}

        # --- Highly descriptive prompt using recommended structure and trigger words ---
        prompt = "A man and a woman are embracing near a lake with mountains in the background. They gaze into each other's eyes, then they share a tender and passionate k144ing kissing. masterpiece, best quality, realistic, high resolution, cinematic film still"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, nsfw, text, watermark, logo, deformed, distorted, disfigured, cartoon, anime"

        # --- Generation parameters aligned with LoRA documentation ---
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