ARG CUSTOM_OCR_BASE_IMAGE
FROM ${CUSTOM_OCR_BASE_IMAGE}

# Set working directory
WORKDIR /app

# Install FastAPI, Uvicorn, and torch-related dependencies (if needed)
RUN pip install --upgrade pip && \
    pip install fastapi uvicorn

# Copy the FastAPI server script into the container
COPY start_server.py /app/start_server.py

# Expose the port FastAPI will run on
EXPOSE 8000

# Set the entrypoint to start your FastAPI server
CMD ["uvicorn", "start_server:app", "--host", "0.0.0.0", "--port", "8000"]
