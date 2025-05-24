FROM continuumio/miniconda3

WORKDIR /app

# Copy env file early to cache this step
COPY environment.yml .

# Create environment
RUN conda env create -f environment.yml && conda clean -afy

# Use conda shell for all commands
SHELL ["conda", "run", "-n", "p4-deployment", "/bin/bash", "-c"]

# Copy entire contents of /deployment into container
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["conda", "run", "--no-capture-output", "-n", "p4-deployment", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
