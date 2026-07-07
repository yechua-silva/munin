FROM rocm/vllm-dev:latest

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose API port (8000) and Streamlit port (8501)
EXPOSE 8000 8501

# Default: run FastAPI API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
