# Using the official Python image
FROM python:3.10

# Setting the working directory in the container
WORKDIR ./

# Copying files
COPY . .

# Dependency installation
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Launching the application
CMD ["python", "main.py"]
