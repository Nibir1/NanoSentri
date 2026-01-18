import os
import shutil
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 1. Define Paths
onnx_path = "onnx_output_dir"
output_path = "phi3_int4_final"

print(f"Quantizing model from '{onnx_path}'...")

# 2. Initialize Quantizer
quantizer = ORTQuantizer.from_pretrained(onnx_path)

# 3. Define Configuration (AVX2 for Mac)
# Note: We use the same config as before to ensure compatibility
qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

# 4. Run Quantization
# CRITICAL FIX: Set use_external_data_format=True because model > 2GB
quantizer.quantize(
    save_dir=output_path, 
    quantization_config=qconfig,
    use_external_data_format=True
)

print(f"✅ SUCCESS! Quantized model saved to: {output_path}")