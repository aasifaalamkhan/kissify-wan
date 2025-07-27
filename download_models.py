#!/usr/bin/env python3
"""
Pre-download models to reduce cold start time
"""
import os
import logging
from diffusers import WanPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Download and cache models"""
    try:
        logger.info("📥 Downloading Wan2.1 model...")
        
        # Download base model
        pipeline = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P",
            torch_dtype=torch.bfloat16,
            variant="fp16"
        )
        
        # Download LoRA
        logger.info("💋 Downloading kissing LoRA...")
        pipeline.load_lora_weights("Remade-AI/kissing")
        
        logger.info("✅ Models downloaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to download models: {str(e)}")
        raise e

if __name__ == "__main__":
    download_models()