# set base image
FROM tensorflow/tensorflow:latest

# set working directory
WORKDIR /app

# set environment variables
ENV PYTHONPATH "${PYTHONPATH}:/"

# install system dependencies
RUN apt-get update -qq && \
    apt-get install -qq libgomp1 build-essential python-dev git
RUN pip install --upgrade pip setuptools

# copy and install python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    python -m spacy download en

# copy app into working directory and grant read/write permissions to data folder
COPY . /app
RUN chmod -R 777 /app/data/

# configure entrypoint
# ENTRYPOINT [ "/bin/sh", "test.sh" ]
CMD [ "python", "-c", "chatbot/scripts/generate_text.py" ]
