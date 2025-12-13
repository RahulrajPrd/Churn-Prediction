# Stage 1: Build Stage - Install dependencies and perform any pre-compilation
# Use a full python image to ensure all build tools are present
FROM python:3.10-slim as builder

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY src/api/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image - A smaller base for the runtime environment
# Using a python-slim base is good practice for production
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy only the installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code and the trained model artifact
# NOTE: The model path needs to be correct relative to the root of the project build context.
# Assuming the Docker build is run from the project root:
# (You might need to adjust the paths depending on where you run 'docker build')
COPY src/api/main.py .
COPY src/model/model.pkl .

# Set environment variable for the model path (used by main.py)
ENV MODEL_PATH=/app/model.pkl
ENV PORT=80

# Expose the application port
EXPOSE 80

# Command to run the application using Uvicorn (a production-ready ASGI server)
# The `--host 0.0.0.0` is essential for running inside a container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]