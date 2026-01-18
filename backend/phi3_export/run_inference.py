import time
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# 1. Configuration
model_path = "phi3_int4_final"
prompt_text = "What usually causes high humidity readings in Vaisala sensors?"

print(f">>> Loading model from '{model_path}'...")
start_time = time.time()

# 2. Load the Optimized Model
# We disable cache and IO binding to match the export settings
model = ORTModelForCausalLM.from_pretrained(
    model_path, 
    use_cache=False, 
    use_io_binding=False
)

# 3. Load Tokenizer (CRITICAL FIX: use_fast=False)
# This forces it to use the Python-based tokenizer which is more stable
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds.")

# 4. Format the Prompt (Phi-3 Chat Template)
messages = [
    {"role": "user", "content": prompt_text},
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt")

# 5. Generate Response
print(f"\n>>> Generating response for: '{prompt_text}'")
gen_start = time.time()

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
)

gen_time = time.time() - gen_start
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 6. Print Result
print("\n" + "="*40)
print(f"GENERATED OUTPUT ({gen_time:.2f}s):")
print("="*40)
clean_response = response.split("<|assistant|>")[-1].strip()
print(clean_response)
print("="*40)