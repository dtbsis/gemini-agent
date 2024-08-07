# Use the official lightweight Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=True

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the app with Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
