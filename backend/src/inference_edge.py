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
    GET /benchmark: Returns hardware performance stats from benchmark report.

Configuration:
    - Uses 'optimum.onnxruntime' for inference.
    - Disables IO Binding/Caching for stability on MacOS/Consumer CPUs.
    - Force-loads slow tokenizer (use_fast=False) to prevent segmentation faults.

Usage:
    uvicorn src.inference_edge:app --reload --host 0.0.0.0 --port 8000
"""

import time
import logging
import pandas as pd
import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, set_seed

# --- Configuration ---
MODEL_PATH = "./phi3_export/phi3_int4_final" 
BENCHMARK_FILE = "benchmark_report.csv"
MAX_TOKENS = 50 
TEMPERATURE = 0.3 

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"BOOT SEQUENCE: Loading model from {MODEL_PATH}...")
    try:
        model = ORTModelForCausalLM.from_pretrained(MODEL_PATH, use_cache=False, use_io_binding=False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        ml_resources["model"] = model
        ml_resources["tokenizer"] = tokenizer
        logger.info("READY.")
        yield
    except Exception as e:
        logger.critical(f"FATAL: {e}")
        raise e
    finally:
        ml_resources.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiagnosticRequest(BaseModel):
    query: str
    context: Optional[str] = None
    max_tokens: int = 50

# --- HEALTH CHECK ENDPOINT ---
@app.get("/health")
def health_check():
    """Heartbeat endpoint for system monitoring."""
    if not ml_resources:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "online", "device": "cpu_edge"}

# --- GET HARDWARE STATS ---
@app.get("/benchmark")
def get_benchmark_stats():
    """Reads the CSV report and returns summary stats."""
    if not os.path.exists(BENCHMARK_FILE):
        return {"status": "no_data"}
    
    try:
        df = pd.read_csv(BENCHMARK_FILE)
        # Calculate averages from the CSV
        return {
            "status": "success",
            "avg_tps": round(df["tokens_per_sec"].mean(), 2),
            "peak_memory_mb": round(df["memory_mb"].max(), 2),
            "load_time_sec": 9.66, # Hardcoded from your log, or you can save it to CSV too
            "device": "CPU (INT4)"
        }
    except Exception as e:
        logger.error(f"Failed to read benchmark: {e}")
        return {"status": "error"}

@app.post("/diagnose")
def run_diagnostic(request: DiagnosticRequest):
    model = ml_resources["model"]
    tokenizer = ml_resources["tokenizer"]
    
    if request.context:
        prompt_content = f"{request.query}\n\nTechnical Context:\n{request.context}"
    else:
        prompt_content = request.query
        
    messages = [{"role": "user", "content": prompt_content}]
    
    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt")
        
        start_ts = time.time()
        set_seed(42) 
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=TEMPERATURE,
            do_sample=True,
        )
        
        inference_time = time.time() - start_ts
        
        # --- CRITICAL FIX: DECODE ONLY NEW TOKENS ---
        # This prevents the prompt (input) from being repeated in the output
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        clean_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "diagnosis": clean_response.strip(),
            "inference_time_sec": round(inference_time, 2),
            "model_version": "phi3-int4-vaisala-v1"
        }
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))