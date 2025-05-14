FROM python:3.10-alpine

# Set the working directory in the container
WORKDIR /app

# Copy everything from the current directory to /app in the container
COPY . /app

# Install the required Python packages
RUN pip install --no-cache-dir fastapi uvicorn onnxruntime numpy pydantic

# Expose port 5000 to the outside
EXPOSE 5000

# Start the FastAPI app using Uvicorn, specifying the correct app module location
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]