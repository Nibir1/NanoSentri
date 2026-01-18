"""
inference_edge.py
-----------------
The Edge Inference Engine (FastAPI).

Purpose:
    Serves the Quantized (INT4) Phi-3 model as a REST API.
    Designed to run on the 'Field Laptop' (Local CPU).

Endpoints:
    POST /diagnose: Accepts a sensor log/query and returns a technical response.
    GET /health: Simple heartbeat check.

Configuration:
    - Uses 'optimum.onnxruntime' for inference.
    - Disables IO Binding/Caching for stability on MacOS/Consumer CPUs.
    - Force-loads slow tokenizer (use_fast=False) to prevent segmentation faults.

Usage:
    uvicorn src.inference_edge:app --reload --host 0.0.0.0 --port 8000
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, set_seed

# --- Configuration ---
MODEL_PATH = "./phi3_export/phi3_int4_final" # Path to the quantized model directory
MAX_TOKENS = 256
TEMPERATURE = 0.3  # Low temperature for technical precision

# --- Logging Setup ---
logging.basicConfig(
    format="%(asctime)s - [NanoSentri] - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Global State ---
# We store the model in a global dict to access it across requests
ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model on startup, unload on shutdown.
    Prevents reloading the model for every single request.
    """
    logger.info(f"BOOT SEQUENCE: Loading model from {MODEL_PATH}...")
    start_time = time.time()
    
    try:
        # CRITICAL FIX: Matching your working run_inference.py settings
        model = ORTModelForCausalLM.from_pretrained(
            MODEL_PATH,
            use_cache=False,
            use_io_binding=False
        )
        
        # CRITICAL FIX: use_fast=False for stability
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        
        ml_resources["model"] = model
        ml_resources["tokenizer"] = tokenizer
        
        logger.info(f"READY. Model loaded in {time.time() - start_time:.2f}s")
        yield
        
    except Exception as e:
        logger.critical(f"FATAL: Could not load model. Error: {e}")
        raise e
    finally:
        logger.info("SHUTDOWN: Cleaning up resources...")
        ml_resources.clear()

# --- API Definition ---
app = FastAPI(
    title="NanoSentri Edge API",
    description="Offline Industrial Diagnostics for Vaisala Sensors",
    version="1.0.0",
    lifespan=lifespan
)

# --- Data Models ---
class DiagnosticRequest(BaseModel):
    query: str
    context: Optional[str] = None # Optional raw log data
    max_tokens: int = 150

class DiagnosticResponse(BaseModel):
    diagnosis: str
    inference_time_sec: float
    model_version: str

# --- Endpoints ---

@app.get("/health")
def health_check():
    """Heartbeat endpoint for system monitoring."""
    if not ml_resources:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "online", "device": "cpu_edge"}

@app.post("/diagnose", response_model=DiagnosticResponse)
def run_diagnostic(request: DiagnosticRequest):
    """
    Main inference endpoint.
    Takes a technical question/log and returns an expert analysis.
    """
    model = ml_resources["model"]
    tokenizer = ml_resources["tokenizer"]
    
    # 1. Prompt Engineering (Phi-3 Format)
    if request.context:
        prompt_content = f"{request.query}\n\nTechnical Context:\n{request.context}"
    else:
        prompt_content = request.query
        
    messages = [{"role": "user", "content": prompt_content}]
    
    try:
        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # 2. Inference
        start_ts = time.time()
        
        # Set seed for reproducibility in diagnostics
        set_seed(42) 
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1 # Prevent looping on error codes
        )
        
        inference_time = time.time() - start_ts
        
        # 3. Decode
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's part
        # Phi-3 typically formats as: <|user|>...<|end|><|assistant|>...
        if "<|assistant|>" in full_response:
            clean_response = full_response.split("<|assistant|>")[-1].strip()
        else:
            # Fallback if specific tokens are stripped differently
            clean_response = full_response.replace(input_text, "").strip()

        return DiagnosticResponse(
            diagnosis=clean_response,
            inference_time_sec=round(inference_time, 2),
            model_version="phi3-int4-vaisala-v1"
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Inference Error")