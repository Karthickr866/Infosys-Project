# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Copy the entire app code into the container
COPY . .

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
