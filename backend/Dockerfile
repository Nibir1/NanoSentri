# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# 1. Install System Dependencies (for compilation if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Source Code
COPY src/ ./src/

# 4. Copy Model Artifacts 
# We copy the 'phi3_export' folder from the host into /app/phi3_export
COPY phi3_export/ ./phi3_export/

# 5. Copy Benchmark Data (Optional, allows dashboard to show stats)
# We use a wildcard/true trick so it doesn't fail if file is missing
COPY benchmark_report.cs[v] ./benchmark_report.csv

# 6. Expose Port
EXPOSE 8000

# 7. Set Environment Variable for the script to find the model
ENV MODEL_PATH="/app/phi3_export/phi3_int4_final"

# 8. Start the Server
CMD ["uvicorn", "src.inference_edge:app", "--host", "0.0.0.0", "--port", "8000"]