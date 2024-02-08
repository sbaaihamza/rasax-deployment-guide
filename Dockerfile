# Extend the official Rasa SDK image
FROM rasa/rasa-sdk:3.6.2

# Use subdirectory as working directory
WORKDIR /app

# Create a cache directory for Hugging Face transformers
RUN mkdir -p /app/huggingface_cache
ENV HF_HOME=/app/huggingface_cache

# Set permissions for the cache directory
RUN chown -R 1001 /app/huggingface_cache
RUN chmod -R 777 /app/huggingface_cache


# Change back to root user to install dependencies
USER root

RUN mkdir -p /.cache
RUN chown -R 1001:1001 /.cache
RUN chmod -R 777 /.cache

# Copy any additional custom requirements, if necessary
COPY actions/requirements.txt ./

# Install extra requirements for actions code, if necessary (uncomment next line)
RUN pip install --upgrade pip
RUN pip install --upgrade typing_extensions
RUN pip install -r requirements.txt


# Copy actions folder to working directory
COPY ./actions /app/actions
COPY ./mydata /app/mydata
COPY ./functions.py /app/functions.py
RUN chmod -R 777 /app/actions
RUN chmod -R 777 /app/actions/mydata
RUN chmod -R 777 /app/mydata

# By best practices, don't run the code with root user
USER 1001
