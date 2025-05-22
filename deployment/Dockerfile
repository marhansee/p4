FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment and code
COPY environment.yml .
COPY . .

# Create Conda environment
RUN conda env create -f environment.yml

# Activate Conda env for all future RUNs and CMDs
SHELL ["conda", "run", "-n", "p4-deployment", "/bin/bash", "-c"]

# Expose FastAPI port
EXPOSE 8000

# Run your FastAPI app
CMD ["conda", "run", "--no-capture-output", "-n", "p4-deployment", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
