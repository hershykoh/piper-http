FROM  dustynv/piper-tts:r35.4.1


# Set the working directory
WORKDIR /opt

# Get the latest version of the code

# Update pip and install the required packages
# RUN pip install --upgrade pip
WORKDIR /opt/piper/src/python_run

# Install the package
#RUN pip install -e .

# Install the requirements
#RUN pip install -r requirements.txt

# Install http server
# RUN pip install -r requirements_http.txt

# Install wget pip package
RUN pip install wget

# Copy the run.py file into the container
COPY run.py /opt
# Copy the download folder into the container
COPY download /opt/download

# Expose the port 5000
EXPOSE 5000

# Create ENV that will be used in the run.py file to set the download link
ENV MODEL_DOWNLOAD_LINK="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/kusal/medium/en_US-kusal-medium.onnx?download=true"

# Create ENV that will be used in the run.py file to set the target folder
ENV MODEL_TARGET_FOLDER="/app/models"

# Create ENV that will be used in the run.py file to set the speaker
ENV SPEAKER="0"

# Run the webserver with python run.py
CMD python3 /opt/run.py $MODEL_DOWNLOAD_LINK $MODEL_TARGET_FOLDER $SPEAKER