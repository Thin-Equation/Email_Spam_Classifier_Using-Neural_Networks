# Use an official Python runtime as a parent image
FROM python:3.11.7

# Set the working directory in the container
WORKDIR /spam_classifier_app

# Copy the current directory contents into the container at /app
COPY . /spam_classifier_app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "Spam_Classifier_App.py"]
