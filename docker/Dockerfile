# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY .. /app

# Expose port 5000 
EXPOSE 5000

# Define the command to run your application
CMD ["python", "app.py"]
