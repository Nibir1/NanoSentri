# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Source Code
COPY src/ ./src/

# 4. Copy Model Artifacts 
COPY phi3_export/ ./phi3_export/

# 5. Copy Benchmark Data
COPY benchmark_report.cs[v] ./benchmark_report.csv

# --- CRITICAL FIX: COPY TESTS INTO CONTAINER ---
COPY tests/ ./tests/
# Copy pytest config if you have it (optional but good practice)
COPY pytest.in[i] ./pytest.ini

# 6. Expose Port
EXPOSE 8000

# 7. Set Environment Variable
ENV MODEL_PATH="/app/phi3_export/phi3_int4_final"
ENV PYTHONPATH=/app

# 8. Start the Server
CMD ["uvicorn", "src.inference_edge:app", "--host", "0.0.0.0", "--port", "8000"]