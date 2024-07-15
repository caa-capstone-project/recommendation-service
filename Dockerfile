# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app.py .
COPY dataloader.py .
COPY model.py .
COPY hyperparameters.py .
COPY train.py .
COPY movie_ids.csv .
COPY model.pth model.pth

# Expose the port the app runs on
EXPOSE 4201

# Run the Flask app
CMD ["python", "app.py"]
