import runpod
import torch
import logging
import traceback
from typing import Dict, Any
import asyncio
from inference_server import InferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global inference engine
inference_engine = None

def initialize_model():
    """Initialize the model on container startup"""
    global inference_engine
    
    try:
        logger.info("🚀 Initializing inference engine...")
        inference_engine = InferenceEngine()
        asyncio.run(inference_engine.initialize())
        logger.info("✅ Inference engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize inference engine: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function
    """
    try:
        logger.info(f"📥 Received job: {event.get('id', 'unknown')}")
        
        # Get input data
        input_data = event.get('input', {})
        
        # Validate required inputs
        required_fields = ['input_images', 'prompt']
        for field in required_fields:
            if field not in input_data:
                return {
                    "error": f"Missing required field: {field}",
                    "status": "failed"
                }
        
        # Run inference
        result = asyncio.run(inference_engine.generate_video(input_data))
        
        logger.info(f"✅ Job completed: {event.get('id', 'unknown')}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            "error": str(e),
            "status": "failed",
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    # Initialize model on startup
    if initialize_model():
        logger.info("🎬 Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("❌ Failed to start handler due to initialization error")
        exit(1)