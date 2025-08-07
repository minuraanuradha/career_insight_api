# Use official Python image with 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose default port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
