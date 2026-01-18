"""
benchmark.py
------------
EnergyOps & Performance Profiler for NanoSentri.

Purpose:
    Quantifies the "Cost" of running the model on Edge hardware.
    Measures Latency (Time-to-First-Token, Generation Speed) and 
    Memory Pressure (RAM Usage).

Usage:
    python src/benchmark.py

Author: NanoSentri Engineering
"""

import time
import psutil
import torch
import pandas as pd
import os
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, set_seed

# --- Config ---
MODEL_PATH = "./phi3_export/phi3_int4_final" # Path to the quantized model directory
PROMPT = "Error code 503 detected on WXT536 sensor. Input voltage 5V."
ITERATIONS = 3  # Run 3 times to get an average

def get_memory_usage():
    """Returns current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark():
    print(f"--- Starting Benchmark: {MODEL_PATH} ---")
    
    # 1. Measure Load Time & Memory
    mem_start = get_memory_usage()
    start_load = time.time()
    
    # Load Model (Same settings as Inference Server)
    model = ORTModelForCausalLM.from_pretrained(
        MODEL_PATH,
        use_cache=False,
        use_io_binding=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    
    load_time = time.time() - start_load
    mem_loaded = get_memory_usage()
    
    print(f"Model Load Time: {load_time:.2f}s")
    print(f"Memory Footprint: {mem_loaded - mem_start:.2f} MB (Total: {mem_loaded:.2f} MB)")
    
    # 2. Warmup
    print("Warming up...")
    inputs = tokenizer(PROMPT, return_tensors="pt")
    model.generate(**inputs, max_new_tokens=10)
    
    # 3. Latency Loops
    results = []
    
    print(f"Running {ITERATIONS} iterations...")
    for i in range(ITERATIONS):
        # Garbage collect to ensure clean slate
        import gc
        gc.collect()
        
        inputs = tokenizer(PROMPT, return_tensors="pt")
        input_tokens = inputs.input_ids.shape[1]
        
        start_gen = time.time()
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=50, # Generate exactly 50 tokens for consistent math
            min_new_tokens=50,
            do_sample=False,   # Greedy decoding for consistent speed tests
            temperature=1.0
        )
        
        end_gen = time.time()
        
        # Metrics
        total_time = end_gen - start_gen
        output_tokens = outputs.shape[1] - input_tokens
        tps = output_tokens / total_time
        
        print(f"  Iter {i+1}: {total_time:.2f}s | {output_tokens} tokens | {tps:.4f} tok/s")
        
        results.append({
            "iteration": i+1,
            "total_time_s": total_time,
            "tokens_generated": output_tokens,
            "tokens_per_sec": tps,
            "memory_mb": get_memory_usage()
        })

    # 4. Save Report
    df = pd.DataFrame(results)
    print("\n--- Summary ---")
    print(df.describe())
    
    report_path = "benchmark_report.csv"
    df.to_csv(report_path, index=False)
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    benchmark()